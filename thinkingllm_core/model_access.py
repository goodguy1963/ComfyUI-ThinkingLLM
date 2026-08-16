"""Model catalog policy and authenticated Hugging Face download helpers."""

import inspect
import os
import shutil
import time
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    from huggingface_hub import snapshot_download
    hf_hub_download = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None
try:
    from server import PromptServer
except Exception:
    PromptServer = None

COMMERCIAL_RELEASE = os.environ.get("THINKINGLLM_COMMERCIAL_RELEASE") == "1"
_DOWNLOAD_STATUS_INTERVAL_SECONDS = 0.25
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
VALID_COMMERCIAL_STATUSES = {"cleared", "external_gated", "unclear", "noncommercial"}
MODEL_STATUS_PRESENTATION = {
    "cleared": (0, "✅ Commercial | "),
    "external_gated": (1, "🔑 External / gated | "),
    "local": (2, "📁 Local / user-supplied | "),
    "unclear": (3, "⚠ Rights unclear | "),
    "noncommercial": (4, "🚫 Non-commercial | "),
}
MODEL_INSTALLED_SUFFIX = " [installed]"
MODEL_LOCAL_PREFIX = "[local] "


def _format_download_size(num_bytes) -> str:
    if num_bytes is None:
        return "? B"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


class _DownloadProgressReporter:
    def __init__(self, node_id=None, label: str = "QwenVL Download", repo_id: str | None = None):
        self.node_id = node_id
        self.label = label
        self.repo_id = repo_id
        self.progress_bar = ProgressBar(1, node_id=node_id) if ProgressBar is not None else None
        self._last_text_at = 0.0
        self._last_terminal_at = 0.0
        self._last_percent = None

    def _emit_ui_text(self, text: str, *, force: bool = False) -> None:
        if PromptServer is None or self.node_id is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_text_at) < _DOWNLOAD_STATUS_INTERVAL_SECONDS:
            return
        PromptServer.instance.send_progress_text(text, self.node_id)
        self._last_text_at = now

    def _emit_terminal_text(self, text: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_terminal_at) < _DOWNLOAD_STATUS_INTERVAL_SECONDS:
            return
        print(text)
        self._last_terminal_at = now

    def _build_message(self, stage: str, detail: str | None = None, current: int | float | None = None, total: int | float | None = None) -> str:
        lines = [self.label, f"Stage: {stage}"]
        if self.repo_id:
            lines.append(f"Source: {self.repo_id}")
        if detail:
            lines.append(f"File: {detail}")
        if total and total > 0:
            percent = int(min(100, max(0, round((float(current or 0) / float(total)) * 100))))
            lines.append(f"Progress: {percent}% ({_format_download_size(current)} / {_format_download_size(total)})")
        elif current is not None:
            lines.append(f"Downloaded: {_format_download_size(current)}")
        return "\n".join(lines)

    def stage(self, stage: str, *, detail: str | None = None, force: bool = True) -> None:
        message = self._build_message(stage, detail=detail)
        self._emit_ui_text(message, force=force)
        self._emit_terminal_text(f"[{self.label}] {stage}{f' - {detail}' if detail else ''}", force=force)

    def progress(self, stage: str, *, detail: str | None = None, current: int | float | None = None, total: int | float | None = None, force: bool = False) -> None:
        if self.progress_bar is not None and total is not None and total > 0:
            self.progress_bar.update_absolute(min(float(current or 0), float(total)), total=float(total))
        percent = None
        if total and total > 0:
            percent = int(min(100, max(0, round((float(current or 0) / float(total)) * 100))))
        message = self._build_message(stage, detail=detail, current=current, total=total)
        self._emit_ui_text(message, force=force or percent != self._last_percent)
        suffix = ""
        if percent is not None:
            suffix = f" {percent}% ({_format_download_size(current)} / {_format_download_size(total)})"
        elif current is not None:
            suffix = f" {_format_download_size(current)}"
        self._emit_terminal_text(f"[{self.label}] {stage}{f' - {detail}' if detail else ''}{suffix}", force=force or percent != self._last_percent)
        self._last_percent = percent

    def finish(self, *, detail: str | None = None, force: bool = True) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update_absolute(1, total=1)
        message = self._build_message("Completed", detail=detail, current=1, total=1)
        self._emit_ui_text(message, force=force)
        self._emit_terminal_text(f"[{self.label}] Completed{f' - {detail}' if detail else ''}", force=force)

    def fail(self, detail: str) -> None:
        message = self._build_message("Failed", detail=detail)
        self._emit_ui_text(message, force=True)
        self._emit_terminal_text(f"[{self.label}] Failed - {detail}", force=True)

    def make_tqdm_class(self):
        if tqdm is None:
            return None
        reporter = self

        class _NodeDownloadTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                    force=True,
                )

            def update(self, n=1):
                result = super().update(n)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                )
                return result

            def refresh(self, *args, **kwargs):
                result = super().refresh(*args, **kwargs)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                )
                return result

            def close(self):
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=(getattr(self, "total", None) or getattr(self, "n", 0)),
                    total=(getattr(self, "total", None) or getattr(self, "n", 0) or None),
                    force=True,
                )
                return super().close()

        return _NodeDownloadTqdm


def _iter_hf_token_candidates(hf_token: str | None):
    token = str(hf_token or "").strip()
    if token:
        yield token
    for env_name in _HF_TOKEN_ENV_VARS:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            yield token


def _clean_hf_token(hf_token: str | None) -> str | None:
    return next(_iter_hf_token_candidates(hf_token), None)


def normalize_commercial_status(info: dict | None) -> str:
    info = info if isinstance(info, dict) else {}
    if info.get("is_local"):
        return "local"
    status = str(info.get("commercial_status") or "unclear").strip().lower()
    if status == "local":
        return "local"
    return status if status in VALID_COMMERCIAL_STATUSES else "unclear"


def _model_display_name(name: str, info: dict | None) -> str:
    info = info if isinstance(info, dict) else {}
    local_display_name = str(info.get("local_display_name") or "").strip()
    if info.get("is_local") and local_display_name:
        label = f"{MODEL_LOCAL_PREFIX}{local_display_name}"
        storage_label = str(info.get("storage_label") or "").strip()
        if storage_label:
            label = f"{label} ({storage_label})"
        return label
    if info.get("is_installed") and not info.get("is_local") and not name.endswith(MODEL_INSTALLED_SUFFIX):
        return f"{name}{MODEL_INSTALLED_SUFFIX}"
    return name


def model_catalog_label(name: str, info: dict | None) -> str:
    display_name = _model_display_name(name, info)
    if not COMMERCIAL_RELEASE:
        return display_name
    return f"{MODEL_STATUS_PRESENTATION[normalize_commercial_status(info)][1]}{display_name}"


def model_catalog_options(models: dict[str, dict], predicate=None) -> list[str]:
    entries = [
        (name, info)
        for name, info in models.items()
        if predicate is None or predicate(name, info or {})
    ]
    if COMMERCIAL_RELEASE:
        entries.sort(
            key=lambda item: (
                MODEL_STATUS_PRESENTATION[normalize_commercial_status(item[1])][0],
                item[0].casefold(),
            ),
        )
    return [model_catalog_label(name, info) for name, info in entries]


def resolve_model_catalog_name(models: dict[str, dict], selected: str) -> str:
    if selected in models:
        return selected
    for name, info in models.items():
        prefix = MODEL_STATUS_PRESENTATION[normalize_commercial_status(info)][1]
        if selected in {model_catalog_label(name, info), f"{prefix}{name}"}:
            return name
    undecorated = selected
    for _order, prefix in MODEL_STATUS_PRESENTATION.values():
        if undecorated.startswith(prefix):
            undecorated = undecorated[len(prefix):]
            break
    if undecorated.endswith(MODEL_INSTALLED_SUFFIX):
        legacy_name = undecorated[:-len(MODEL_INSTALLED_SUFFIX)]
        if legacy_name in models:
            return legacy_name
    return selected


def _catalog_filename_candidates(info: dict) -> set[str]:
    candidates: set[str] = set()
    for key in ("filename", "local_filenames", "local_model_files"):
        value = info.get(key)
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = value
        elif isinstance(value, dict):
            values = [*value.keys(), *value.values()]
        else:
            values = []
        for candidate in values:
            if isinstance(candidate, (list, tuple, set)):
                nested = candidate
            else:
                nested = [candidate]
            for item in nested:
                filename = Path(str(item)).name.casefold()
                if filename:
                    candidates.add(filename)
    return candidates


def mark_catalog_files_installed(models: dict[str, dict], search_dirs) -> None:
    """Mark catalog entries found in any configured directory without changing their keys."""
    installed_files: dict[str, Path] = {}
    for directory in search_dirs:
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            for path in directory.rglob("*.gguf"):
                if path.is_file():
                    installed_files.setdefault(path.name.casefold(), path)
        except (OSError, PermissionError):
            continue

    for info in models.values():
        if not isinstance(info, dict) or info.get("is_local"):
            continue
        installed_path = next(
            (installed_files[name] for name in _catalog_filename_candidates(info) if name in installed_files),
            None,
        )
        if installed_path is None:
            info.pop("is_installed", None)
            info.pop("installed_path", None)
            continue
        info["is_installed"] = True
        info["installed_path"] = str(installed_path)


def enforce_model_access(info: dict | None, model_name: str, *, local_exists: bool, hf_token: str | None = None) -> None:
    status = normalize_commercial_status(info)
    if COMMERCIAL_RELEASE:
        if status == "cleared":
            return
        raise PermissionError(f"[QwenVL] Commercial mode rejects model '{model_name}' with status '{status}'.")
    if not local_exists and status == "external_gated" and not _clean_hf_token(hf_token):
        raise PermissionError(
            f"[QwenVL] Cannot download {model_name}: no Hugging Face access token was supplied. "
            "Accept the model's access terms on Hugging Face, then add a read token to the hf_token field "
            "or set the HF_TOKEN environment variable."
        )
    if not local_exists and status == "local":
        raise FileNotFoundError(f"[QwenVL] Local user-supplied model not found: {model_name}")


def _redact_hf_token(text: str, hf_token: str | None) -> str:
    redacted = str(text)
    for token in _iter_hf_token_candidates(hf_token):
        redacted = redacted.replace(token, "<redacted HF token>")
    return redacted


def _hf_download_error_message(exc: Exception, *, repo_id: str, hf_token: str | None) -> str:
    message = _redact_hf_token(str(exc), hf_token)
    lower = message.lower()
    access_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "gated",
        "private",
        "repository not found",
        "requires authentication",
    )
    if any(marker in lower for marker in access_markers):
        if _clean_hf_token(hf_token):
            message += (
                f"\n[QwenVL] Hugging Face access hint for {repo_id}: a token was supplied but access was rejected. "
                "Confirm the token has read permission and that you accepted the model license/gate on Hugging Face."
            )
        else:
            message += (
                f"\n[QwenVL] Cannot download {repo_id}: no Hugging Face access token was supplied. "
                "Accept the model's access terms on Hugging Face, then add a read token to the hf_token field "
                "or set the HF_TOKEN environment variable."
            )
    return message


def _call_hf_hub_download(*, repo_id: str, filename: str, local_dir: Path, token: str | None, reporter: _DownloadProgressReporter) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub.hf_hub_download is unavailable in this environment")
    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "filename": filename,
        "local_dir": str(local_dir),
    }
    if token:
        download_kwargs["token"] = token
    tqdm_cls = reporter.make_tqdm_class()
    if tqdm_cls is not None:
        try:
            if "tqdm_class" in inspect.signature(hf_hub_download).parameters:
                download_kwargs["tqdm_class"] = tqdm_cls
        except (TypeError, ValueError):
            pass
    try:
        return Path(hf_hub_download(**download_kwargs))
    finally:
        download_kwargs.pop("token", None)



def _model_snapshot_has_weights(target: Path) -> bool:
    return any(target.glob("*.safetensors")) or any(target.glob("*.bin"))


def _cleanup_unneeded_snapshot_weights(target: Path) -> None:
    unwanted_markers = ("fp32", "bf16", "fp8", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0", "f32")
    for weight_file in target.rglob("*.safetensors"):
        name = weight_file.name.lower()
        if any(marker in name for marker in unwanted_markers):
            print(f"[QwenVL] Removing unneeded quantized weight: {weight_file.name}")
            weight_file.unlink(missing_ok=True)


def _model_snapshot_has_required_files(target: Path, require_processor: bool = False) -> bool:
    if not target.exists() or not target.is_dir():
        return False
    if not _model_snapshot_has_weights(target):
        return False
    if not (target / "config.json").exists():
        return False
    if require_processor:
        processor_files = (
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
            "video_preprocessor_config.json",
        )
        if not any((target / name).exists() for name in processor_files):
            return False
    return True


def _download_model_snapshot(repo_id: str, target: Path, *, force_clean_target: bool = False, node_id=None, progress_label: str | None = None, hf_token: str | None = None) -> None:
    if COMMERCIAL_RELEASE:
        raise RuntimeError("[QwenVL] Commercial release mode only permits preinstalled, hash-locked models.")
    reporter = _DownloadProgressReporter(node_id=node_id, label=progress_label or "QwenVL HF Download", repo_id=repo_id)
    token_for_download = _clean_hf_token(hf_token)
    download_kwargs = {}
    if force_clean_target and target.exists() and target.is_dir():
        print(f"[QwenVL] Removing incomplete model snapshot before retry: {target}")
        reporter.stage("Cleaning incomplete snapshot", detail=target.name)
        shutil.rmtree(target, ignore_errors=False)
    reporter.stage("Preparing download", detail=target.name)
    try:
        download_kwargs = {
            "repo_id": repo_id,
            "local_dir": str(target),
            "ignore_patterns": ["*.md", ".git*"],
            "force_download": force_clean_target,
        }
        if token_for_download:
            download_kwargs["token"] = token_for_download
        tqdm_cls = reporter.make_tqdm_class()
        if tqdm_cls is not None:
            download_kwargs["tqdm_class"] = tqdm_cls
        snapshot_download(**download_kwargs)
        _cleanup_unneeded_snapshot_weights(target)
    except Exception as exc:
        reporter.fail(_hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download))
        raise RuntimeError(_hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)) from exc
    finally:
        download_kwargs.pop("token", None)
        token_for_download = None
    reporter.finish(detail=target.name)


def download_hf_file_to_path(repo_ids: list[str], filename: str, target_path: Path, *, node_id=None, progress_label: str = "QwenVL GGUF Download", hf_token: str | None = None) -> None:
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return
    if COMMERCIAL_RELEASE:
        raise RuntimeError("[QwenVL] Commercial release mode only permits preinstalled, hash-locked models.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    wanted_name = Path(filename).name
    token_for_download = _clean_hf_token(hf_token)
    exact_candidates = []
    for candidate in (filename, wanted_name):
        if candidate and candidate not in exact_candidates:
            exact_candidates.append(candidate)

    try:
        for repo_id in repo_ids:
            reporter = _DownloadProgressReporter(node_id=node_id, label=progress_label, repo_id=repo_id)
            auth_label = "Preparing authenticated download" if token_for_download else "Preparing download"
            reporter.stage(auth_label, detail=wanted_name)
            if hf_hub_download is not None:
                for candidate in exact_candidates:
                    try:
                        reporter.stage("Resolving exact file", detail=candidate, force=False)
                        downloaded_path = _call_hf_hub_download(
                            repo_id=repo_id,
                            filename=candidate,
                            local_dir=target_path.parent,
                            token=token_for_download,
                            reporter=reporter,
                        )
                        if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            downloaded_path.replace(target_path)
                        if target_path.exists():
                            reporter.finish(detail=wanted_name)
                            return
                    except Exception as exc:
                        hint = _hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)
                        last_exc = RuntimeError(hint)
                        print(f"[QwenVL] exact file download failed from {repo_id}/{candidate}: {hint}")
            reporter.stage("Falling back to snapshot scan", detail=wanted_name, force=False)
            download_kwargs = {}
            try:
                download_kwargs = {
                    "repo_id": repo_id,
                    "repo_type": "model",
                    "local_dir": str(target_path.parent),
                    "allow_patterns": [filename, wanted_name, f"**/{wanted_name}"],
                }
                if token_for_download:
                    download_kwargs["token"] = token_for_download
                tqdm_cls = reporter.make_tqdm_class()
                if tqdm_cls is not None:
                    download_kwargs["tqdm_class"] = tqdm_cls
                snapshot_download(**download_kwargs)
                found = list(target_path.parent.rglob(wanted_name))
                if found:
                    downloaded_path = found[0]
                    if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        downloaded_path.replace(target_path)
                if target_path.exists():
                    reporter.finish(detail=wanted_name)
                    return
                reporter.fail(f"{wanted_name} was not found after snapshot download")
            except Exception as exc:
                hint = _hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)
                last_exc = RuntimeError(hint)
                reporter.fail(hint)
                print(f"[QwenVL] snapshot_download failed from {repo_id}: {hint}")
            finally:
                download_kwargs.pop("token", None)
    finally:
        token_for_download = None

    raise FileNotFoundError(f"[QwenVL] Download failed for {wanted_name}: {last_exc}")
