"""
SSH-based reMarkable client.

Bypasses the reMarkable Cloud API entirely by pulling .rm files directly from
the tablet over SSH/scp. Eliminates the N-round-trip cloud download (~3-15s)
and replaces it with a single scp call (~0.5-1.5s).

Enable by setting REMARKABLE_SSH_HOST (default: 10.11.99.1 for USB connection).

The zip produced by download() is byte-compatible with the cloud client's output,
so the existing render pipeline (rmc + cairosvg) requires zero changes.

SSH ControlMaster is used for connection reuse — after the first connect (~0.3s),
subsequent calls share the same socket with ~0.05s overhead.
"""

import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

from rm_mcp.models import Document

logger = logging.getLogger(__name__)

XOCHITL_PATH = "/home/root/.local/share/remarkable/xochitl"
_SSH_TIMEOUT = int(os.environ.get("REMARKABLE_SSH_TIMEOUT", "3"))


class SSHRemarkableClient:
    """reMarkable client using direct SSH access instead of the Cloud API.

    Connects to the tablet at the configured host (default 10.11.99.1 for USB)
    and pulls .rm annotation files via scp. ControlMaster keeps the connection
    alive between calls to amortize SSH handshake overhead.
    """

    def __init__(self, host: str = "10.11.99.1", user: str = "root"):
        self.host = host
        self.user = user
        self._controlpath = f"/tmp/rm-ssh-ctl-{user}-{host}"
        self._ssh_base = [
            "ssh",
            "-o",
            f"ControlPath={self._controlpath}",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=120",  # Keep socket alive for 120s of inactivity
            "-o",
            f"ConnectTimeout={_SSH_TIMEOUT}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",  # Never prompt for passwords
        ]

    def _ssh(self, cmd: str) -> str:
        """Run a command on the tablet via SSH, return stdout. Raises on failure."""
        result = subprocess.run(
            self._ssh_base + [f"{self.user}@{self.host}", cmd],
            capture_output=True,
            text=True,
            timeout=_SSH_TIMEOUT + 2,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed: {result.stderr.strip()}")
        return result.stdout

    def _scp(self, remote_path: str, local_path: str) -> None:
        """Copy a file from the tablet via scp."""
        scp_cmd = [
            "scp",
            "-q",
            "-o",
            f"ControlPath={self._controlpath}",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ConnectTimeout={_SSH_TIMEOUT}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",
            f"{self.user}@{self.host}:{remote_path}",
            local_path,
        ]
        result = subprocess.run(scp_cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"scp failed for {remote_path}: {result.stderr.decode().strip()}")

    def is_available(self) -> bool:
        """Check if the tablet is reachable over SSH."""
        try:
            self._ssh("echo ok")
            return True
        except Exception:
            return False

    def get_root_hash(self) -> str:
        """Return a hash representing the current state of the xochitl directory.

        Uses the combined mtime of all .metadata files as a cheap change indicator.
        Changes within ~1s of a modification (mtime resolution).
        """
        try:
            out = self._ssh(
                f"find {XOCHITL_PATH} -maxdepth 1 -name '*.metadata' -printf '%T@\\n' "
                f"| sort | md5sum"
            )
            return out.strip().split()[0]
        except Exception:
            # Fall back to directory mtime
            out = self._ssh(f"stat -c %Y {XOCHITL_PATH} 2>/dev/null || echo 0")
            return out.strip()

    def get_meta_items(self, limit: Optional[int] = None, **kwargs) -> List[Document]:
        """List all documents on the tablet by reading .metadata files via SSH."""
        try:
            # Read all .metadata files in one SSH call
            out = self._ssh(
                f"for f in {XOCHITL_PATH}/*.metadata; do "
                f'[ -f "$f" ] && printf "%s\\t" "$(basename "$f" .metadata)" && cat "$f" && echo; '
                f"done"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to list tablet documents: {e}")

        documents: List[Document] = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            try:
                tab_idx = line.index("\t")
                doc_id = line[:tab_idx]
                meta = json.loads(line[tab_idx + 1 :])
            except (ValueError, json.JSONDecodeError):
                continue

            if meta.get("deleted", False):
                continue
            if meta.get("parent") == "trash":
                continue

            # Use mtime as a proxy for content hash (good enough for change detection)
            try:
                mtime = self._ssh(
                    f"stat -c %Y {XOCHITL_PATH}/{doc_id}.metadata 2>/dev/null || echo 0"
                ).strip()
                doc_hash = hashlib.md5(f"{doc_id}:{mtime}".encode()).hexdigest()
            except Exception:
                doc_hash = doc_id

            documents.append(
                Document(
                    id=doc_id,
                    hash=doc_hash,
                    name=meta.get("visibleName", doc_id),
                    doc_type=meta.get("type", "DocumentType"),
                    parent=meta.get("parent", ""),
                    deleted=False,
                    pinned=meta.get("pinned", False),
                    last_modified=None,
                    size=0,
                )
            )
            if limit and len(documents) >= limit:
                break

        return documents

    def download(self, doc) -> bytes:
        """Pull document files via scp and package into a zip for the render pipeline.

        The zip format matches exactly what cloud.py's download() produces, so
        render_page_from_document_zip() and extract_text_from_document_zip()
        require zero changes.
        """
        uuid = doc.ID

        # Fetch .content and .metadata in one SSH call
        try:
            content_json = self._ssh(
                f"cat {XOCHITL_PATH}/{uuid}.content 2>/dev/null || echo '{{}}'"
            )
            metadata_json = self._ssh(
                f"cat {XOCHITL_PATH}/{uuid}.metadata 2>/dev/null || echo '{{}}'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata for {uuid}: {e}")

        # Get list of .rm files (one per page)
        try:
            rm_list_out = self._ssh(f"ls {XOCHITL_PATH}/{uuid}/*.rm 2>/dev/null || true")
            rm_paths = [p.strip() for p in rm_list_out.strip().splitlines() if p.strip()]
        except Exception:
            rm_paths = []

        # Package into zip — structure mirrors the cloud zip format
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(f"{uuid}.content", content_json)
            zf.writestr(f"{uuid}.metadata", metadata_json)

            for rm_path in rm_paths:
                page_uuid = Path(rm_path).stem
                with tempfile.NamedTemporaryFile(suffix=".rm", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    self._scp(rm_path, tmp_path)
                    zf.write(tmp_path, f"{uuid}/{page_uuid}.rm")
                except Exception as e:
                    logger.warning(f"Failed to pull {rm_path}: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        buf.seek(0)
        return buf.read()
