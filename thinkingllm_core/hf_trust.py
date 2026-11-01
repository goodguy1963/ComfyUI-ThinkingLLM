"""Operator-controlled, revision-pinned trust for Hugging Face repository code."""

import json
import os
import re
import tempfile
from pathlib import Path


TRUSTED_HF_MODELS_ENV = "THINKINGLLM_TRUSTED_HF_MODELS"
_COMMIT_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")


def get_trusted_hf_revision(repo_id: str) -> str | None:
    """Return an explicitly configured immutable commit for a repository."""
    if os.environ.get("THINKINGLLM_COMMERCIAL_RELEASE") == "1" or not repo_id:
        return None

    raw = os.environ.get(TRUSTED_HF_MODELS_ENV, "").strip()
    if not raw:
        return None

    trusted: dict[str, str] = {}
    for item in raw.split(","):
        entry = item.strip()
        configured_repo, separator, revision = entry.rpartition("@")
        if not separator or not configured_repo or not _COMMIT_REVISION.fullmatch(revision):
            raise ValueError(
                f"{TRUSTED_HF_MODELS_ENV} entries must use repo_id@<40-character commit SHA>"
            )
        normalized_revision = revision.lower()
        existing = trusted.get(configured_repo)
        if existing is not None and existing != normalized_revision:
            raise ValueError(f"Conflicting pinned revisions configured for Hugging Face repository {configured_repo}")
        trusted[configured_repo] = normalized_revision

    return trusted.get(repo_id)


def _trust_marker_path(model_path) -> Path:
    path = Path(model_path)
    return path.parent / f".{path.name}.thinkingllm-trust.json"


def mark_trusted_hf_snapshot(model_path, repo_id: str, revision: str) -> None:
    """Record that this isolated snapshot was downloaded from the pinned commit."""
    if not _COMMIT_REVISION.fullmatch(str(revision)):
        raise ValueError("A 40-character commit SHA is required to mark a trusted snapshot")

    marker = _trust_marker_path(model_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"repo_id": repo_id, "revision": revision.lower()}, sort_keys=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent,
            prefix=f".{marker.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, marker)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def trust_remote_code_for_snapshot(model_path, repo_id: str | None) -> bool:
    """Enable Transformers custom Python only for a verified pinned snapshot."""
    if not repo_id:
        return False
    revision = get_trusted_hf_revision(repo_id)
    if revision is None:
        return False

    try:
        marker = json.loads(_trust_marker_path(model_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return marker == {"repo_id": repo_id, "revision": revision}
