"""
Background PNG pre-fetch pipeline for reMarkable MCP.

Polls for recently-modified documents every REMARKABLE_PREFETCH_INTERVAL seconds
(default 60s), downloads and renders pages 1-3 of changed notebooks to a disk
PNG cache. When a tool call comes in for a pre-fetched page, the render step
(~1-2s) is already done.

Architectural constraint: sampling OCR requires a live MCP Context, so this
pipeline can only pre-render PNG images — not pre-run OCR. The OCR result cache
(L1/L2 in cache.py) handles the OCR side after the first on-demand read.

Enable: set REMARKABLE_PREFETCH_ENABLED=1 (default off to avoid
  unexpected background network activity).
Disable: set REMARKABLE_PREFETCH_ENABLED=0 (or leave unset).

Configuration:
  REMARKABLE_PREFETCH_ENABLED   — 1 to enable, 0/unset to disable
  REMARKABLE_PREFETCH_INTERVAL  — seconds between root-hash polls (default 60)
  REMARKABLE_PREFETCH_MAX_DOCS  — max documents rendered per cycle (default 10)
  REMARKABLE_PNG_CACHE_PATH     — disk cache directory
                                  (default ~/.cache/rm-mcp/png-cache)
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _is_prefetch_enabled() -> bool:
    return os.environ.get("REMARKABLE_PREFETCH_ENABLED", "0").lower() in ("1", "true", "yes")


def _get_interval() -> int:
    try:
        return int(os.environ.get("REMARKABLE_PREFETCH_INTERVAL", "60"))
    except ValueError:
        return 60


def _get_max_docs() -> int:
    try:
        return int(os.environ.get("REMARKABLE_PREFETCH_MAX_DOCS", "10"))
    except ValueError:
        return 10


def _get_png_cache_dir() -> Path:
    custom = os.environ.get("REMARKABLE_PNG_CACHE_PATH", "")
    if custom:
        return Path(custom)
    return Path.home() / ".cache" / "rm-mcp" / "png-cache"


# ---------------------------------------------------------------------------
# Public helpers — used by tool call sites to read pre-rendered PNGs
# ---------------------------------------------------------------------------


def get_prefetched_png(doc_id: str, page: int) -> Optional[bytes]:
    """Return pre-rendered PNG bytes from disk cache, or None if not ready."""
    path = _get_png_cache_dir() / doc_id / f"{page:04d}.png"
    try:
        if path.exists():
            return path.read_bytes()
    except OSError:
        pass
    return None


def _store_prefetched_png(doc_id: str, page: int, png_data: bytes) -> None:
    cache_dir = _get_png_cache_dir() / doc_id
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{page:04d}.png").write_bytes(png_data)
    except OSError as e:
        logger.debug(f"PNG cache write failed for {doc_id} p{page}: {e}")


def invalidate_doc_png_cache(doc_id: str) -> None:
    """Delete all cached PNGs for a document (called when doc hash changes)."""
    cache_dir = _get_png_cache_dir() / doc_id
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.debug(f"Invalidated PNG cache for {doc_id}")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Background prefetch loop
# ---------------------------------------------------------------------------

# Pages to pre-render per document (first N pages)
_PAGES_TO_PREFETCH = 3


async def _prefetch_loop(shutdown_event: asyncio.Event) -> None:
    """Background task: poll for changes, pre-render notebook pages to disk."""
    from rm_mcp.api import get_rmapi
    from rm_mcp.cache import set_cached_collection
    from rm_mcp.extract.render import get_background_color, render_page_from_document_zip
    from rm_mcp.index import get_instance

    interval = _get_interval()
    max_docs = _get_max_docs()
    loop = asyncio.get_event_loop()
    consecutive_errors = 0
    last_root_hash: Optional[str] = None

    client = get_rmapi()
    if client is None:
        logger.info("Prefetch: not authenticated, skipping")
        return

    if not hasattr(client, "get_root_hash"):
        logger.info("Prefetch: client does not support root hash, skipping")
        return

    logger.info(f"Prefetch pipeline started (interval={interval}s, max_docs={max_docs})")

    while not shutdown_event.is_set():
        try:
            # ----------------------------------------------------------------
            # Step 1: Cheap change detection via root hash
            # ----------------------------------------------------------------
            current_hash = await loop.run_in_executor(None, client.get_root_hash)

            if current_hash == last_root_hash:
                consecutive_errors = 0
                await asyncio.sleep(interval)
                continue

            last_root_hash = current_hash

            # ----------------------------------------------------------------
            # Step 2: Fetch updated collection (shared with tool-call cache)
            # ----------------------------------------------------------------
            items = await loop.run_in_executor(
                None, lambda h=current_hash: client.get_meta_items(root_hash=h)
            )
            set_cached_collection(client, items, root_hash=current_hash)
            consecutive_errors = 0

            # ----------------------------------------------------------------
            # Step 3: Find documents that need pre-rendering
            # ----------------------------------------------------------------
            index = get_instance()
            docs_to_render = []

            for item in items:
                if item.is_folder:
                    continue
                name_lower = item.VissibleName.lower()
                if name_lower.endswith(".pdf") or name_lower.endswith(".epub"):
                    continue  # Only pre-render notebooks

                # Invalidate stale PNGs when document hash changes
                if index is not None and item.hash:
                    if index.needs_reindex(item.ID, item.hash):
                        invalidate_doc_png_cache(item.ID)
                        docs_to_render.append(item)
                        continue

                # Also queue if page 1 PNG is missing entirely
                if not (_get_png_cache_dir() / item.ID / "0001.png").exists():
                    docs_to_render.append(item)

            # Sort: most recently modified first
            docs_to_render.sort(
                key=lambda d: d.last_modified or "",
                reverse=True,
            )
            docs_to_render = docs_to_render[:max_docs]

            if not docs_to_render:
                logger.debug("Prefetch: no documents need rendering")
                await asyncio.sleep(interval)
                continue

            logger.info(f"Prefetch: pre-rendering {len(docs_to_render)} document(s)")
            bg_color = get_background_color()

            # ----------------------------------------------------------------
            # Step 4: Download + render pages 1-N for each document
            # ----------------------------------------------------------------
            for doc in docs_to_render:
                if shutdown_event.is_set():
                    break

                try:
                    # Download in executor (blocking network IO).
                    # Populate the zip cache so the tool-call hot path skips the download.
                    from rm_mcp.cache import cache_zip, get_cached_zip

                    doc_hash = getattr(doc, "hash", None)
                    raw = doc_hash and get_cached_zip(doc_hash)
                    if not raw:
                        raw = await loop.run_in_executor(None, lambda d=doc: client.download(d))
                        if doc_hash:
                            cache_zip(doc_hash, raw)

                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                        tmp.write(raw)
                        tmp_path = Path(tmp.name)

                    try:
                        import zipfile

                        from rm_mcp.extract.notebook import _get_ordered_rm_files

                        with zipfile.ZipFile(tmp_path, "r") as zf:
                            import tempfile as _tf

                            with _tf.TemporaryDirectory() as td:
                                zf.extractall(td)
                                rm_files = _get_ordered_rm_files(Path(td))
                                page_count = len(rm_files)

                        pages_to_render = range(1, min(page_count + 1, _PAGES_TO_PREFETCH + 1))

                        for page_num in pages_to_render:
                            if shutdown_event.is_set():
                                break

                            # Skip already-cached pages
                            if get_prefetched_png(doc.ID, page_num) is not None:
                                continue

                            # Render in executor (CPU-bound subprocess)
                            png = await loop.run_in_executor(
                                None,
                                lambda p=page_num: render_page_from_document_zip(
                                    tmp_path, p, background_color=bg_color
                                ),
                            )
                            if png:
                                _store_prefetched_png(doc.ID, page_num, png)
                                logger.debug(
                                    f"Prefetch: rendered '{doc.VissibleName}' "
                                    f"p{page_num} ({len(png) // 1024}KB)"
                                )

                            # Yield between pages so tool calls can run
                            await asyncio.sleep(0.05)

                        # Store page count in index for cache-hit path
                        if index is not None and page_count:
                            index.upsert_document(
                                doc_id=doc.ID,
                                doc_hash=doc.hash or None,
                                name=doc.VissibleName,
                                file_type="notebook",
                                page_count=page_count,
                            )

                    finally:
                        tmp_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.debug(f"Prefetch: failed for '{doc.VissibleName}': {e}")
                    continue

            logger.info(f"Prefetch: cycle complete — sleeping {interval}s")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            consecutive_errors += 1
            backoff = min(2**consecutive_errors * 10, 300)
            logger.debug(f"Prefetch: error #{consecutive_errors}: {e} — backoff {backoff}s")
            await asyncio.sleep(backoff)
            continue

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Lifecycle management
# ---------------------------------------------------------------------------


def start_prefetch_pipeline() -> Optional[asyncio.Task]:
    """Start the background prefetch task. Returns task or None if disabled."""
    if not _is_prefetch_enabled():
        return None
    shutdown_event = asyncio.Event()
    try:
        task = asyncio.create_task(_prefetch_loop(shutdown_event))
        task._shutdown_event = shutdown_event  # type: ignore[attr-defined]
        logger.info("Prefetch pipeline started")
        return task
    except Exception as e:
        logger.warning(f"Could not start prefetch pipeline: {e}")
        return None


async def stop_prefetch_pipeline(task: Optional[asyncio.Task]) -> None:
    """Stop the background prefetch task gracefully."""
    if task is None:
        return
    if hasattr(task, "_shutdown_event"):
        task._shutdown_event.set()
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("Prefetch pipeline stopped")
