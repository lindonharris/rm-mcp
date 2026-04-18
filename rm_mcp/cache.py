"""
Centralized caching for reMarkable MCP.

Consolidates all cache instances and their access functions from
api.py, extract.py, and tools.py into one module.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Collection cache (from api.py)
# =============================================================================

_cached_collection = None
_cached_root_hash: Optional[str] = None
_cache_timestamp: float = 0.0

try:
    _CACHE_TTL_SECONDS = int(os.environ.get("REMARKABLE_CACHE_TTL", "60"))
except ValueError:
    logger.warning("Invalid REMARKABLE_CACHE_TTL value, using default of 60 seconds")
    _CACHE_TTL_SECONDS = 60

# Disk cache for collection — survives process restarts.
# Prevents 429s when multiple Claude sessions start concurrently.
_DISK_CACHE_PATH = (
    Path(
        os.environ.get("REMARKABLE_INDEX_PATH", str(Path.home() / ".cache" / "rm-mcp" / "index.db"))
    ).parent
    / "collection-cache.json"
)


def _load_disk_collection_cache() -> bool:
    """
    Load collection cache from disk if it exists and is within TTL.
    Returns True if cache was loaded successfully.
    """
    global _cached_collection, _cached_root_hash, _cache_timestamp

    try:
        if not _DISK_CACHE_PATH.exists():
            return False
        mtime = _DISK_CACHE_PATH.stat().st_mtime
        if (time.time() - mtime) >= _CACHE_TTL_SECONDS:
            return False  # Stale

        data = json.loads(_DISK_CACHE_PATH.read_text())
        from rm_mcp.models import Document

        items = []
        for d in data.get("items", []):
            last_modified = d.get("last_modified")
            if last_modified is not None:
                from datetime import datetime

                try:
                    last_modified = datetime.fromisoformat(last_modified)
                except (ValueError, TypeError):
                    last_modified = None
            items.append(
                Document(
                    id=d["id"],
                    hash=d["hash"],
                    name=d["name"],
                    doc_type=d["doc_type"],
                    parent=d.get("parent", ""),
                    deleted=d.get("deleted", False),
                    pinned=d.get("pinned", False),
                    last_modified=last_modified,
                    size=d.get("size", 0),
                    files=d.get("files", []),
                    synced=d.get("synced", True),
                )
            )
        _cached_collection = items
        _cached_root_hash = data.get("root_hash")
        _cache_timestamp = mtime
        logger.debug(f"Loaded {len(items)} items from disk collection cache")
        return True
    except Exception as e:
        logger.debug(f"Failed to load disk collection cache: {e}")
        return False


def _save_disk_collection_cache(collection: list, root_hash: Optional[str]) -> None:
    """Persist collection to disk so the next process start can skip the API call."""
    try:
        items = []
        for doc in collection:
            lm = doc.last_modified
            items.append(
                {
                    "id": doc.id,
                    "hash": doc.hash,
                    "name": doc.name,
                    "doc_type": doc.doc_type,
                    "parent": doc.parent,
                    "deleted": doc.deleted,
                    "pinned": doc.pinned,
                    "last_modified": lm.isoformat() if lm is not None else None,
                    "size": doc.size,
                    "files": doc.files,
                    "synced": doc.synced,
                }
            )
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DISK_CACHE_PATH.write_text(json.dumps({"root_hash": root_hash, "items": items}))
    except Exception as e:
        logger.debug(f"Failed to save disk collection cache: {e}")


def get_cached_collection() -> Tuple[Any, List]:
    """
    Get the client and document collection, using a cache to avoid redundant fetches.

    Cache strategy:
    - If cache is < 60s old, return it immediately (0 requests)
    - Else fetch root hash (1 request), compare to cached hash
    - If unchanged, refresh timestamp and return cache (1 request total)
    - If changed, do a full re-fetch

    Returns:
        Tuple of (client, collection)
    """
    global _cached_collection, _cached_root_hash, _cache_timestamp

    from rm_mcp.api import get_rmapi

    client = get_rmapi()
    if client is None:
        raise RuntimeError("Not authenticated. Run: uvx rm-mcp --setup")
    now = time.time()

    # If we have a valid in-memory cache within TTL, return immediately
    if _cached_collection is not None and (now - _cache_timestamp) < _CACHE_TTL_SECONDS:
        logger.debug("Collection cache hit (within TTL)")
        return client, _cached_collection

    # Try disk cache before making any network requests
    if _cached_collection is None and _load_disk_collection_cache():
        age = time.time() - _cache_timestamp
        if _cached_collection is not None and age < _CACHE_TTL_SECONDS:
            logger.debug("Collection cache hit (disk)")
            return client, _cached_collection

    # Check if client supports root hash (for change detection)
    if not hasattr(client, "get_root_hash"):
        collection = client.get_meta_items()
        _cached_collection = collection
        _cache_timestamp = time.time()
        return client, collection

    # Cloud mode: check root hash to see if anything changed
    try:
        current_hash = client.get_root_hash()
    except Exception:
        # If root hash fetch fails, do a full re-fetch
        collection = client.get_meta_items()
        _cached_collection = collection
        _cache_timestamp = time.time()
        return client, collection

    if _cached_collection is not None and current_hash == _cached_root_hash:
        # Nothing changed, refresh timestamp
        logger.debug("Collection cache hit (root hash unchanged)")
        _cache_timestamp = time.time()
        return client, _cached_collection

    # Root hash changed or no cache — full re-fetch
    logger.debug("Collection cache miss — fetching full collection")
    collection = client.get_meta_items(root_hash=current_hash)
    _cached_collection = collection
    _cached_root_hash = current_hash
    _cache_timestamp = time.time()
    _save_disk_collection_cache(collection, current_hash)
    return client, collection


def set_cached_collection(client, collection, root_hash: Optional[str] = None) -> None:
    """
    Populate the collection cache from an external source (e.g., background loader).

    This allows the resource loader to share its fetched data with tools,
    so the first tool call after startup doesn't need to re-fetch.

    Args:
        client: The reMarkable API client
        collection: List of document/folder items
        root_hash: Optional root hash to avoid an extra network call.
                   If not provided, will attempt to fetch from client.
    """
    global _cached_collection, _cached_root_hash, _cache_timestamp

    # Also set the client singleton in api.py
    import rm_mcp.api as api_mod

    api_mod._client_singleton = client

    _cached_collection = collection
    # Use provided root hash or try to get one for future comparisons
    if root_hash is not None:
        _cached_root_hash = root_hash
    elif hasattr(client, "get_root_hash"):
        try:
            _cached_root_hash = client.get_root_hash()
        except Exception:
            pass
    _cache_timestamp = time.time()
    _save_disk_collection_cache(collection, _cached_root_hash)


def invalidate_collection_cache() -> None:
    """Force the next get_cached_collection() call to re-fetch."""
    global _cached_collection, _cached_root_hash, _cache_timestamp
    _cached_collection = None
    _cached_root_hash = None
    _cache_timestamp = 0.0


# =============================================================================
# Extraction cache (from extract.py)
# =============================================================================

# Cache TTL in seconds (5 minutes)
EXTRACTION_CACHE_TTL_SECONDS = 300

# Maximum cache sizes to prevent unbounded memory growth
_MAX_EXTRACTION_CACHE_SIZE = 50
_MAX_PAGE_OCR_CACHE_SIZE = 200

# Module-level cache for OCR results (full document)
# Key: doc_id
# Value: {"result": extraction_result, "include_ocr": bool, "timestamp": float}
_extraction_cache: Dict[str, Dict[str, Any]] = {}

# Per-page cache for sampling OCR results
# Key: (doc_id, page_number, backend)
# Value: {"text": str, "timestamp": float}
_page_ocr_cache: Dict[tuple, Dict[str, Any]] = {}


def _is_cache_valid(cached: Dict[str, Any]) -> bool:
    """Check if a cached entry is still valid based on TTL."""
    if "timestamp" not in cached:
        return False  # Unknown age = stale
    return (time.time() - cached["timestamp"]) < EXTRACTION_CACHE_TTL_SECONDS


def clear_extraction_cache(doc_id: Optional[str] = None) -> None:
    """
    Clear the extraction cache.

    Args:
        doc_id: If provided, only clear cache for this document.
                If None, clear the entire cache.
    """
    if doc_id:
        _extraction_cache.pop(doc_id, None)
        # Also clear per-page cache entries for this document
        keys_to_remove = [k for k in _page_ocr_cache if k[0] == doc_id]
        for key in keys_to_remove:
            _page_ocr_cache.pop(key, None)
    else:
        _extraction_cache.clear()
        _page_ocr_cache.clear()


def get_cached_page_ocr(
    doc_id: str,
    page: int,
    backend: str,
) -> Optional[str]:
    """
    Get cached OCR result for a specific page.

    Checks L1 (in-memory) first, then falls back to L2 (SQLite index).
    On L2 hit, promotes the result back to L1.

    Args:
        doc_id: Document ID
        page: Page number (1-indexed)
        backend: OCR backend used ("sampling")

    Returns:
        Cached OCR text or None if not cached/expired
    """
    # L1: in-memory cache
    cache_key = (doc_id, page, backend)
    if cache_key in _page_ocr_cache:
        cached = _page_ocr_cache[cache_key]
        if _is_cache_valid(cached):
            return cached["text"]
        # Expired, remove it
        _page_ocr_cache.pop(cache_key, None)

    # L2: SQLite index
    try:
        from rm_mcp.index import get_instance

        index = get_instance()
        if index is not None:
            text = index.get_page_ocr(doc_id, page, backend)
            if text is not None:
                # Promote to L1
                _page_ocr_cache[cache_key] = {
                    "text": text,
                    "timestamp": time.time(),
                }
                logger.debug(f"L2 cache hit for page OCR: {doc_id} p{page}")
                return text
    except Exception:
        logger.debug("L2 read failed for page OCR", exc_info=True)

    return None


def cache_page_ocr(
    doc_id: str,
    page: int,
    backend: str,
    text: str,
) -> None:
    """
    Cache OCR result for a specific page.

    Writes to both L1 (in-memory) and L2 (SQLite index).

    Args:
        doc_id: Document ID
        page: Page number (1-indexed)
        backend: OCR backend used ("sampling")
        text: OCR text result
    """
    # L1: in-memory cache
    cache_key = (doc_id, page, backend)
    _page_ocr_cache[cache_key] = {
        "text": text,
        "timestamp": time.time(),
    }
    if len(_page_ocr_cache) > _MAX_PAGE_OCR_CACHE_SIZE:
        excess = len(_page_ocr_cache) - _MAX_PAGE_OCR_CACHE_SIZE
        for key in list(_page_ocr_cache.keys())[:excess]:
            del _page_ocr_cache[key]

    # L2: write-through to SQLite index
    try:
        from rm_mcp.index import get_instance

        index = get_instance()
        if index is not None:
            index.upsert_page(doc_id, page, text, "ocr", backend)
    except Exception:
        logger.debug("L2 write failed for page OCR", exc_info=True)


def get_cached_ocr_result(
    doc_id: str,
    include_ocr: bool = True,
    ocr_backend: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get cached OCR result for a document if available and valid.

    Args:
        doc_id: Document ID to look up
        include_ocr: Whether OCR content is required
        ocr_backend: If specified, only return cache if it was produced by this backend.
                     Use "sampling". None accepts any backend.

    Returns:
        Cached result dict or None if not cached/expired/wrong backend
    """
    if doc_id in _extraction_cache:
        cached = _extraction_cache[doc_id]
        if (cached["include_ocr"] or not include_ocr) and _is_cache_valid(cached):
            # Check backend match if specified
            if ocr_backend is not None:
                cached_backend = cached["result"].get("ocr_backend")
                if cached_backend != ocr_backend:
                    return None
            return cached["result"]
    return None


def cache_ocr_result(
    doc_id: str,
    result: Dict[str, Any],
    include_ocr: bool = True,
) -> None:
    """
    Cache an OCR result for a document.

    Writes to both L1 (in-memory) and L2 (SQLite index).

    Args:
        doc_id: Document ID
        result: Extraction result dict with keys: typed_text, highlights,
                handwritten_text, pages, page_ids, ocr_backend
        include_ocr: Whether this result includes OCR content
    """
    _extraction_cache[doc_id] = {
        "result": result,
        "include_ocr": include_ocr,
        "timestamp": time.time(),
    }
    if len(_extraction_cache) > _MAX_EXTRACTION_CACHE_SIZE:
        excess = len(_extraction_cache) - _MAX_EXTRACTION_CACHE_SIZE
        for key in list(_extraction_cache.keys())[:excess]:
            del _extraction_cache[key]

    # L2: write-through to SQLite index
    try:
        from rm_mcp.index import get_instance

        index = get_instance()
        if index is not None:
            index.store_extraction_result(doc_id, result)
    except Exception:
        logger.debug("L2 write failed for extraction result", exc_info=True)


# =============================================================================
# File type cache (from tools.py)
# =============================================================================

_file_type_cache: Dict[str, str] = {}

# =============================================================================
# Rendered image cache (from tools.py)
# =============================================================================

_rendered_image_cache: Dict[str, str] = {}  # key: f"{doc_id}:{page}" -> base64 PNG


# =============================================================================
# Document zip cache (content-addressed by doc.hash)
# =============================================================================

# doc.hash is content-addressed — safe to cache indefinitely within a session.
# Keyed by hash so a modified document automatically gets a different cache slot.
_MAX_ZIP_CACHE_SIZE = 10  # Keep last 10 documents in memory (~50-500MB depending on sizes)
_zip_cache: Dict[str, bytes] = {}  # key: doc.hash -> zip bytes


def get_cached_zip(doc_hash: str) -> Optional[bytes]:
    """Return cached zip bytes for a document by its content hash, or None."""
    return _zip_cache.get(doc_hash)


def cache_zip(doc_hash: str, zip_bytes: bytes) -> None:
    """Cache zip bytes for a document by its content hash (FIFO eviction)."""
    if len(_zip_cache) >= _MAX_ZIP_CACHE_SIZE:
        oldest = next(iter(_zip_cache))
        del _zip_cache[oldest]
    _zip_cache[doc_hash] = zip_bytes
