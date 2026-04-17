"""remarkable_status tool — check connection and authentication."""

import os

from rm_mcp.server import mcp
from rm_mcp.tools import _helpers


@mcp.tool(annotations=_helpers.STATUS_ANNOTATIONS)
def remarkable_status(compact_output: bool = False) -> str:
    """
    <usecase>Check connection status and authentication with reMarkable Cloud.</usecase>
    <instructions>
    Returns authentication status and diagnostic information.
    Use this to verify your connection or troubleshoot issues.
    Includes index statistics, SSH mode status, and prefetch pipeline status when available.
    </instructions>
    <examples>
    - remarkable_status()
    - remarkable_status(compact_output=True)  # Omit hints
    </examples>
    """
    compact = _helpers.is_compact(compact_output)

    # Detect transport mode
    ssh_host = os.environ.get("REMARKABLE_SSH_HOST", "").strip()
    if ssh_host:
        transport = "ssh"
        connection_info = f"SSH to {ssh_host}"
    else:
        transport = "cloud"
        connection_info = "environment variable" if _helpers.REMARKABLE_TOKEN else "file (~/.rmapi)"

    try:
        client, collection = _helpers.get_cached_collection()
        items_by_id = _helpers.get_items_by_id(collection)

        root = _helpers._get_root_path()

        # Count documents (not folders, filtered by root)
        doc_count = 0
        for item in collection:
            if item.is_folder:
                continue
            item_path = _helpers.get_item_path(item, items_by_id)
            if _helpers._is_within_root(item_path, root):
                doc_count += 1

        result = {
            "authenticated": True,
            "transport": transport,
            "connection": connection_info,
            "status": "connected",
            "document_count": doc_count,
        }

        # Add root path info if configured
        if root != "/":
            result["root_path"] = root

        # Add configuration details
        from rm_mcp.cache import _CACHE_TTL_SECONDS

        result["config"] = {
            "ocr_backend": _helpers.get_ocr_backend(),
            "root_path": root,
            "background_color": _helpers.get_background_color(),
            "cache_ttl_seconds": _CACHE_TTL_SECONDS,
            "compact_mode": _helpers.is_compact(),
        }

        # Add OCR model preferences summary
        try:
            from rm_mcp.ocr.sampling import OCR_MODEL_PREFERENCES

            hints = [h.name for h in (OCR_MODEL_PREFERENCES.hints or []) if h.name]
            result["ocr_model_priority"] = hints[:3] if hints else []
            result["ocr_speed_priority"] = OCR_MODEL_PREFERENCES.speedPriority
        except Exception:
            pass

        # Add prefetch pipeline status
        prefetch_enabled = os.environ.get("REMARKABLE_PREFETCH_ENABLED", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        prefetch_info: dict = {"enabled": prefetch_enabled}
        if prefetch_enabled:
            try:
                interval = int(os.environ.get("REMARKABLE_PREFETCH_INTERVAL", "60"))
                max_docs = int(os.environ.get("REMARKABLE_PREFETCH_MAX_DOCS", "10"))
                from rm_mcp.prefetch import _get_png_cache_dir

                cache_dir = _get_png_cache_dir()
                cached_docs = len(list(cache_dir.glob("*"))) if cache_dir.exists() else 0
                prefetch_info.update(
                    {
                        "interval_seconds": interval,
                        "max_docs_per_cycle": max_docs,
                        "cache_dir": str(cache_dir),
                        "cached_documents": cached_docs,
                    }
                )
            except Exception:
                pass
        result["prefetch"] = prefetch_info

        # Add index stats if available
        try:
            from rm_mcp.index import get_instance

            index = get_instance()
            if index is not None:
                stats = index.get_stats()
                result.update(stats)
                # Add coverage percentage
                indexed = stats.get("index_documents", 0)
                if doc_count > 0:
                    pct = int(indexed / doc_count * 100)
                    result["index_coverage"] = f"{indexed}/{doc_count} documents indexed ({pct}%)"
        except Exception:
            pass

        hint_parts = [f"Connected successfully via {transport}. Found {doc_count} documents."]
        if root != "/":
            hint_parts.append(f"Filtered to root: {root}")
        if transport == "ssh":
            hint_parts.append(
                f"Using SSH direct access ({ssh_host}) — bypasses cloud for low latency."
            )
        if prefetch_enabled:
            cached = result.get("prefetch", {}).get("cached_documents", 0)
            hint_parts.append(f"Prefetch pipeline active: {cached} document(s) pre-rendered.")
        if "index_coverage" in result:
            hint_parts.append(f"Index coverage: {result['index_coverage']}.")
        hint_parts.append(
            "Use remarkable_browse() to see your files, "
            "or remarkable_recent() for recent documents."
        )

        return _helpers.make_response(result, " ".join(hint_parts), compact=compact)

    except Exception as e:
        error_msg = str(e)

        result = {
            "authenticated": False,
            "transport": transport,
            "connection": connection_info,
            "error": error_msg,
        }

        hint = "To authenticate: run 'uvx rm-mcp --setup' and follow the instructions."

        return _helpers.make_response(result, hint, compact=compact)
