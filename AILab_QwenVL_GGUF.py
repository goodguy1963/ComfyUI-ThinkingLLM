# ComfyUI-QwenVL (GGUF)
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python and configured through gguf_models.json.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import base64
import gc
import hashlib
import io
import inspect
import json
import os
import struct
import sys
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from AILab_StreamDisplay import TerminalStreamDisplay
from AILab_LlamaCppInstaller import ensure_llama_cpp_backend
from comfy.model_management import throw_exception_if_processing_interrupted

# Import cache functions from main module
sys.path.append(str(Path(__file__).parent))
from AILab_QwenVL import (
    PROMPT_CACHE,
    apply_qwen_soft_thinking_directive,
    build_node_input_signature,
    ensure_cuda_vram_headroom,
    estimate_qwen_text_tokens,
    get_cache_key,
    get_alternative_cache_key,
    get_image_hash,
    get_video_hash,
    save_prompt_cache,
    get_node_saved_prompt,
    get_node_saved_prompt_with_seed,
    resolve_qwen_thinking_mode,
    set_node_saved_prompt,
    load_node_prompt_state,
    _make_node_state_key,
    NODE_PROMPT_STATE,
)

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output

# DEPRECATED: per-node state via set_node_saved_prompt / get_node_saved_prompt is used instead.
LAST_SAVED_PROMPT = None  # kept only to avoid ImportError in legacy importers

# Load per-node state at startup so keep_last_prompt works across restarts
load_node_prompt_state()


def read_gguf_architecture(filepath: Path) -> str | None:
    """Read general.architecture from a GGUF file header without loading the model.

    Returns the architecture string (e.g. 'qwen3', 'qwen2vl', 'llama') or None on failure.
    """
    # GGUF value type enum
    _VTYPE_SIZE = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
    }
    _VTYPE_STRING = 8
    _VTYPE_ARRAY = 9

    def _read_string(f):
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _skip_value(f, vtype):
        if vtype in _VTYPE_SIZE:
            f.seek(_VTYPE_SIZE[vtype], 1)
        elif vtype == _VTYPE_STRING:
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, 1)
        elif vtype == _VTYPE_ARRAY:
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            for _ in range(arr_len):
                _skip_value(f, arr_type)
        else:
            return False  # unknown type, bail
        return True

    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture":
                    if vtype == _VTYPE_STRING:
                        return _read_string(f)
                    else:
                        return None
                # Skip this value and continue searching
                if not _skip_value(f, vtype):
                    return None
    except Exception:
        return None
    return None


def _parse_repo_quant_sizes(repo_key: str) -> dict[str, str] | None:
    """Parse bracket info from repo key like 'Gemma-4-E4B-it-GGUF [Q4:5.4GB|Q8:8GB|VRAM:~7GB]'.
    Returns a dict mapping quant names to size strings, or None if no bracket info."""
    match = re.search(r'\[([^\]]+)\]', repo_key)
    if not match:
        return None
    sizes: dict[str, str] = {}
    for part in match.group(1).split('|'):
        part = part.strip()
        if ':' in part:
            k, v = part.split(':', 1)
            sizes[k.strip()] = v.strip()
    return sizes if sizes else None


def _quant_from_filename(filename: str) -> str | None:
    """Extract quantization level from a GGUF filename stem.
    E.g. 'gemma-4-E4B-it-Q4_K_M.gguf' -> 'Q4'."""
    name = Path(filename).stem.upper()
    for q in ('BF16', 'F16', 'F32', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q3_K_M', 'Q2_K'):
        if q in name:
            return _QUANT_CANONICAL.get(q, q)
    return None

_QUANT_CANONICAL = {
    'BF16': 'BF16', 'F16': 'F16', 'F32': 'F32',
    'Q8_0': 'Q8', 'Q6_K': 'Q6',
    'Q5_K_M': 'Q5', 'Q5_K_S': 'Q5',
    'Q4_K_M': 'Q4', 'Q4_K_S': 'Q4',
    'Q3_K_M': 'Q3', 'Q2_K': 'Q2',
}

import re

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
_STREAM_HEARTBEAT_INTERVAL_SECONDS = 3.0
_GGUF_MAX_FINALIZATION_ATTEMPTS = 3
_ACTIVE_GGUF_LOADERS: "weakref.WeakSet[object]" = weakref.WeakSet()
_REASONING_ONLY_RE = re.compile(
    r"(?is)^\s*(?:<think[^>]*>.*?</think>\s*)?(?:okay[,.:]?\s*)?(?:"
    r"let'?s\s+(?:think|reason)|"
    r"i\s+(?:should|need|must|will|am\s+going\s+to|have\s+to)|"
    r"wait[,.:]?|"
    r"hmm[,.:]?|"
    r"the\s+user\s+(?:is\s+asking|wants)"
    r")"
)


def _looks_like_reasoning_only_answer(text: str) -> bool:
    if not text:
        return False
    normalized = " ".join(str(text).split())
    return bool(_REASONING_ONLY_RE.search(normalized))


def _answer_output_is_usable(cleaned_text: str) -> bool:
    return bool(cleaned_text and not _looks_like_reasoning_only_answer(cleaned_text))


def _maybe_emit_answer_stream_heartbeat(stage_label: str, started_at: float, last_status_at: float, full_text: str) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    cleaned_preview = clean_model_output(full_text, OutputCleanConfig(mode="text")).strip()
    if _answer_output_is_usable(cleaned_preview):
        return now
    elapsed = max(0.0, now - started_at)
    print(f"[QwenVL GGUF] {stage_label}: reasoning hidden; still generating... ({len(full_text)} chars, {elapsed:.1f}s)")
    return now


def register_active_gguf_loader(loader: object) -> None:
    if loader is not None:
        _ACTIVE_GGUF_LOADERS.add(loader)


def release_other_gguf_loaders(current_loader: object, next_model_label: str) -> None:
    register_active_gguf_loader(current_loader)
    for loader in list(_ACTIVE_GGUF_LOADERS):
        if loader is None or loader is current_loader:
            continue
        llm = getattr(loader, "llm", None)
        chat_handler = getattr(loader, "chat_handler", None)
        if llm is None and chat_handler is None:
            continue
        clear_fn = getattr(loader, "clear", None)
        if not callable(clear_fn):
            continue
        try:
            print(f"[QwenVL] Releasing another GGUF node's loaded model before loading {next_model_label}.")
            clear_fn()
        except Exception as exc:
            print(f"[QwenVL] Warning: failed to release another GGUF node before loading {next_model_label}: {exc}")


def _load_prompt_config():
    preset_prompts = ["🚫 No preset (image-only)", "🖼️ Detailed Description"]
    system_prompts: dict[str, str] = {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        preset_prompts = data.get("_preset_prompts") or preset_prompts
        system_prompts = data.get("_system_prompts") or system_prompts
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")

    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            system_prompts = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            preset_prompts = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    if "🚫 No preset (image-only)" not in preset_prompts:
        preset_prompts = ["🚫 No preset (image-only)", *preset_prompts]
    if isinstance(system_prompts, dict):
        system_prompts.setdefault("🚫 No preset (image-only)", "")

    return preset_prompts, system_prompts


PRESET_PROMPTS, SYSTEM_PROMPTS = _load_prompt_config()


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    repo_id: str | None
    alt_repo_ids: list[str]
    author: str | None
    repo_dirname: str
    model_filename: str
    mmproj_filename: str | None
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _model_name_to_filename_candidates(model_name: str) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()
    candidates = {raw, f"{raw}.gguf"}
    if " / " in raw:
        tail = raw.split(" / ", 1)[1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    if "/" in raw:
        tail = raw.rsplit("/", 1)[-1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    return candidates


def _scan_local_gguf_models(base_dir: Path, existing_filenames: set[str]) -> dict[str, dict]:
    """Scan the GGUF base directory for locally available .gguf files not already in the JSON catalog."""
    local_models: dict[str, dict] = {}
    if not base_dir.exists() or not base_dir.is_dir():
        return local_models

    # Walk all subdirectories and collect .gguf files grouped by parent directory
    dirs_with_gguf: dict[Path, list[Path]] = {}
    try:
        for gguf_file in base_dir.rglob("*.gguf"):
            if gguf_file.is_file():
                parent = gguf_file.parent
                dirs_with_gguf.setdefault(parent, []).append(gguf_file)
    except PermissionError:
        pass

    for dir_path, gguf_files in dirs_with_gguf.items():
        # Separate mmproj files from model files
        mmproj_files = [f for f in gguf_files if "mmproj" in f.name.lower()]
        model_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]

        # Pick the first mmproj file in this directory (if any)
        mmproj_path = mmproj_files[0] if mmproj_files else None

        for model_file in model_files:
            # Skip if this filename is already in the JSON catalog
            if model_file.name in existing_filenames:
                continue

            display = f"[local] {model_file.name}"
            local_models[display] = {
                "filename": str(model_file),
                "mmproj_filename": str(mmproj_path) if mmproj_path else None,
                "is_local": True,
                "repo_id": None,
                "alt_repo_ids": [],
                "author": None,
                "repo_dirname": dir_path.name,
                "context_length": 32768,
                "image_max_tokens": 4096,
                "n_batch": 512,
                "gpu_layers": -1,
                "top_k": 20,
                "pool_size": 4194304,
            }

    if local_models:
        print(f"[QwenVL] Discovered {len(local_models)} local GGUF model(s) on disk")

    return local_models


def _load_gguf_vl_catalog():
    data = {}
    if GGUF_CONFIG_PATH.exists():
        try:
            with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception as exc:
            print(f"[QwenVL] gguf_models.json load failed: {exc}")

    base_dir = data.get("base_dir") or "LLM/GGUF"

    flattened: dict[str, dict] = {}

    repos = data.get("qwenVL_model") or data.get("vl_repos") or data.get("repos") or {}
    seen_display_names: set[str] = set()
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt_repo_ids = repo.get("alt_repo_ids") or []

        defaults = repo.get("defaults") or {}
        mmproj_file = repo.get("mmproj_file")
        model_files = repo.get("model_files") or []

        quant_sizes = _parse_repo_quant_sizes(repo_key)
        for model_file in model_files:
            display = Path(model_file).name
            if quant_sizes:
                q = _quant_from_filename(model_file)
                if q and q in quant_sizes:
                    display = f"{display} [~{quant_sizes[q]}]"
            if display in seen_display_names:
                display = f"{display} ({repo_key})"
            seen_display_names.add(display)
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "mmproj_filename": mmproj_file,
            }

    legacy_models = data.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            flattened[name] = entry

    # Scan filesystem for locally available models not in JSON config
    # Collect all directories to scan: the configured base_dir + any extra LLM paths from ComfyUI
    existing_filenames = {Path(e.get("filename", "")).name for e in flattened.values() if e.get("filename")}
    scan_dirs: list[Path] = [_resolve_base_dir(base_dir)]
    try:
        if "LLM" in folder_paths.folder_names_and_paths:
            for llm_path in folder_paths.get_folder_paths("LLM"):
                gguf_dir = Path(llm_path) / "GGUF"
                if gguf_dir not in scan_dirs:
                    scan_dirs.append(gguf_dir)
                # Also scan the LLM path itself (users may put GGUFs directly there)
                llm_p = Path(llm_path)
                if llm_p not in scan_dirs:
                    scan_dirs.append(llm_p)
    except Exception:
        pass
    for scan_dir in scan_dirs:
        local_models = _scan_local_gguf_models(scan_dir, existing_filenames)
        flattened.update(local_models)
        # Update existing_filenames so we don't add duplicates across directories
        existing_filenames.update(Path(e.get("filename", "")).name for e in local_models.values() if e.get("filename"))

    return {"base_dir": base_dir, "models": flattened}


GGUF_VL_CATALOG = _load_gguf_vl_catalog()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _tensor_to_base64_png(tensor) -> str | None:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    if array.ndim == 2:
        pil_img = Image.fromarray(array, mode="L")
    elif array.shape[-1] == 4:
        pil_img = Image.fromarray(array, mode="RGBA")
    else:
        pil_img = Image.fromarray(array[..., :3], mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _sample_video_frames(video, frame_count: int):
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    if total <= frame_count:
        return [video[i] for i in range(total)]
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]


def _is_media_capacity_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    markers = (
        "media evaluation failed",
        "failed to find a memory slot",
        "invalid input batch",
        "fatal decode error",
        "llama_decode failed",
    )
    return any(marker in message for marker in markers)


def _build_video_retry_settings(frame_count: int, n_batch: int | None, image_max_tokens: int | None):
    requested_frames = max(int(frame_count), 1)
    current_n_batch = int(n_batch) if n_batch is not None else 512
    current_img_tokens = int(image_max_tokens) if image_max_tokens is not None else 4096

    attempts: list[tuple[int, int, int]] = []
    candidates = [
        (min(requested_frames, 8), min(current_n_batch, 256), min(current_img_tokens, 2048)),
        (min(requested_frames, 4), min(current_n_batch, 128), min(current_img_tokens, 1024)),
    ]
    seen: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        if candidate not in seen:
            attempts.append(candidate)
            seen.add(candidate)
    return attempts


def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path):
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for repo_id in repo_ids:
        print(f"[QwenVL] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
    else:
        raise FileNotFoundError(f"[QwenVL] Download failed for {filename}: {last_exc}")

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


def _find_existing_local_file(base_dir: Path, filename: str) -> Path | None:
    """Reuse an already-downloaded GGUF file anywhere under the shared base dir."""
    wanted = Path(filename).name
    if not wanted:
        return None
    try:
        for candidate in base_dir.rglob(wanted):
            if candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    all_models = GGUF_VL_CATALOG.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = _model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break

    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []

    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name)

    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_vl_models.json entry missing 'filename' for: {model_name}")

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    return GGUFVLResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=_safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 32768),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 20),
        pool_size=_int("pool_size", 4194304),
    )


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        register_active_gguf_loader(self)

    def clear(self):
        print(f"[QwenVL GGUF DEBUG] Starting VRAM cleanup...")

        # Force cleanup of chat handler first
        if self.chat_handler is not None:
            try:
                # Try to explicitly close the chat handler if it has a close method
                if hasattr(self.chat_handler, 'close'):
                    self.chat_handler.close()
                elif hasattr(self.chat_handler, '__del__'):
                    self.chat_handler.__del__()
            except Exception as e:
                print(f"[QwenVL GGUF DEBUG] Error closing chat_handler: {e}")
            finally:
                self.chat_handler = None

        # Force cleanup of LLM model
        if self.llm is not None:
            try:
                # Try to explicitly close the LLM if it has a close method
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                elif hasattr(self.llm, '__del__'):
                    self.llm.__del__()
                # Force garbage collection of the model
                del self.llm
            except Exception as e:
                print(f"[QwenVL GGUF DEBUG] Error closing LLM: {e}")
            finally:
                self.llm = None

        # Clear signature
        self.current_signature = None

        # Aggressive garbage collection
        gc.collect()

        # Force CUDA cache cleanup multiple times
        if torch.cuda.is_available():
            print(f"[QwenVL GGUF DEBUG] Clearing CUDA cache...")
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
            # Additional cleanup
            torch.cuda.empty_cache()

        print(f"[QwenVL GGUF DEBUG] VRAM cleanup completed")

    def _load_backend(self):
        return ensure_llama_cpp_backend(require_vision_handlers=True)

    def _load_model(
        self,
        model_name: str,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        enable_thinking: bool = True,
    ):
        Llama = self._load_backend()

        resolved = _resolve_model_entry(model_name)

        # Local models store absolute paths — use them directly, skip download logic
        if Path(resolved.model_filename).is_absolute():
            model_path = Path(resolved.model_filename)
            mmproj_path = Path(resolved.mmproj_filename) if resolved.mmproj_filename else None
            if not model_path.exists():
                raise FileNotFoundError(f"[QwenVL] Local GGUF model not found: {model_path}")
            if mmproj_path is not None and not mmproj_path.exists():
                raise FileNotFoundError(f"[QwenVL] Local mmproj not found: {mmproj_path}")
        else:
            base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "llm/GGUF")

            author_dir = _safe_dirname(resolved.author or "")
            repo_dir = _safe_dirname(resolved.repo_dirname)
            target_dir = base_dir / author_dir / repo_dir

            model_path = target_dir / Path(resolved.model_filename).name
            mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

            existing_model = _find_existing_local_file(base_dir, resolved.model_filename)
            if existing_model is not None:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if not model_path.exists():
                    try:
                        model_path.hardlink_to(existing_model)
                    except Exception:
                        model_path = existing_model
                else:
                    model_path = model_path

            if mmproj_path is not None:
                existing_mmproj = _find_existing_local_file(base_dir, resolved.mmproj_filename)
                if existing_mmproj is not None:
                    mmproj_path.parent.mkdir(parents=True, exist_ok=True)
                    if not mmproj_path.exists():
                        try:
                            mmproj_path.hardlink_to(existing_mmproj)
                        except Exception:
                            mmproj_path = existing_mmproj
                    else:
                        mmproj_path = mmproj_path

            repo_ids: list[str] = []
            if resolved.repo_id:
                repo_ids.append(resolved.repo_id)
            repo_ids.extend(resolved.alt_repo_ids)

            if not model_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] GGUF model not found locally and no repo_id provided: {model_path}")
                _download_single_file(repo_ids, resolved.model_filename, model_path)

            if mmproj_path is not None and not mmproj_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] mmproj not found locally and no repo_id provided: {mmproj_path}")
                _download_single_file(repo_ids, resolved.mmproj_filename, mmproj_path)

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_gpu_layers,
            n_batch_val,
            device_kind,
            img_max,
            top_k_val,
            pool_size_val,
            bool(enable_thinking),
        )
        if self.llm is not None and self.current_signature == signature:
            ensure_cuda_vram_headroom("QwenVL GGUF", min_free_gb=1.0, min_free_ratio=0.08)
            return

        release_other_gguf_loaders(self, Path(resolved.model_filename).name)

        # Force aggressive cleanup before loading new model (especially for same model conflicts)
        print(f"[QwenVL GGUF DEBUG] Forcing cleanup before model loading...")
        self.clear()

        # Additional wait for CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(0.1)  # Brief pause for cleanup to complete

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                    handler_cls = Qwen25VLChatHandler
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md"
                    )

            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }

        # Detect architecture from GGUF metadata instead of relying on model name
        arch = read_gguf_architecture(model_path)
        self.gguf_arch = arch
        is_qwen35 = arch in ("qwen35", "qwen35moe") if arch else "qwen3.5-" in model_name.lower()
        self.supports_qwen_soft_think = arch == "qwen3" if arch else "qwen3-" in model_name.lower()
        # Thinking toggle: for Qwen3.5/4 we control enable_thinking via the chat template kwarg.
        # For non-Qwen backends (Gemma, LLaMA) the flag is advisory and may be silently ignored.
        if is_qwen35 or arch == "qwen3" or arch == "qwen":
            thinking_enabled = bool(enable_thinking)
            llm_kwargs["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
            state_label = "enabled" if thinking_enabled else "disabled"
            print(f"[QwenVL] Qwen architecture detected (arch={arch}): Thinking {state_label} via chat template.")

        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max

        self.current_context_length = n_ctx
        self.current_image_token_budget = img_max

        print(f"[QwenVL] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")

        self.llm = Llama(**llm_kwargs_filtered)
        self.current_signature = signature

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        model_name: str = "",
        stream_to_terminal: bool = False,
        enable_thinking: bool = True,
    ):
        """Returns (cleaned_text, raw_text) tuple."""
        ensure_cuda_vram_headroom("QwenVL GGUF", min_free_gb=1.0, min_free_ratio=0.08)
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        if supports_soft_think:
            directive = "/think" if enable_thinking else "/no_think"
            print(f"[QwenVL] Qwen3 GGUF detected: Thinking {'enabled' if enable_thinking else 'disabled'} via chat template and {directive}.")
        if self.llm is not None and hasattr(self.llm, "reset"):
            try:
                self.llm.reset()
            except Exception as exc:
                print(f"[QwenVL GGUF DEBUG] llama context reset skipped: {exc}")

        def _run_completion(
            system_prompt_text: str,
            user_prompt_text: str,
            images_for_call: list[str],
            *,
            stage_label: str,
            seed_value: int,
            attempt_enable_thinking: bool | None = None,
        ) -> tuple[str, str]:
            current_enable_thinking = enable_thinking if attempt_enable_thinking is None else bool(attempt_enable_thinking)
            effective_user_prompt = apply_qwen_soft_thinking_directive(
                user_prompt_text,
                current_enable_thinking,
                supports_soft_switch=supports_soft_think,
            )
            if images_for_call:
                content = [{"type": "text", "text": effective_user_prompt}]
                for img in images_for_call:
                    if not img:
                        continue
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": effective_user_prompt},
                ]

            start = time.perf_counter()
            if stream_to_terminal:
                stream_display = TerminalStreamDisplay("QwenVL GGUF", suppress_planning=True, compact=False)
                stream_display.start_stage(stage_label)
                stage_started_at = time.monotonic()
                last_status_at = stage_started_at
                full_text = ""
                result = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repeat_penalty=float(repetition_penalty),
                    seed=int(seed_value),
                    stop=["<|im_end|>", "<|im_start|>"],
                    stream=True,
                )
                for chunk in result:
                    throw_exception_if_processing_interrupted()
                    token = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if token:
                        full_text += token
                        stream_display.push(token)
                        last_status_at = _maybe_emit_answer_stream_heartbeat(stage_label, stage_started_at, last_status_at, full_text)
                stream_display.end_stage()
                raw_full = full_text.strip()
                cleaned = clean_model_output(raw_full, OutputCleanConfig(mode="text"))
                return cleaned.strip(), raw_full

            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repeat_penalty=float(repetition_penalty),
                seed=int(seed_value),
                stop=["<|im_end|>", "<|im_start|>"]
            )
            throw_exception_if_processing_interrupted()
            elapsed = max(time.perf_counter() - start, 1e-6)

            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int) and completion_tokens > 0:
                tok_s = completion_tokens / elapsed
                if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                    print(
                        f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                        f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                    )
                else:
                    print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

            content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
            raw_content = str(content or "").strip()
            cleaned = clean_model_output(raw_content, OutputCleanConfig(mode="text"))
            return cleaned.strip(), raw_content

        cleaned_text, raw_text = _run_completion(
            system_prompt,
            user_prompt,
            images_b64,
            stage_label="STREAMING",
            seed_value=seed,
        )
        raw_trace_parts = [f"[STREAMING]\n{raw_text}"]
        best_cleaned = cleaned_text
        if _answer_output_is_usable(best_cleaned):
            return best_cleaned, "\n\n".join(raw_trace_parts)

        current_raw = raw_text
        for attempt_number in range(2, _GGUF_MAX_FINALIZATION_ATTEMPTS + 1):
            if stream_to_terminal:
                print(
                    "[QwenVL GGUF] "
                    f"Finalization attempt {attempt_number}/{_GGUF_MAX_FINALIZATION_ATTEMPTS} "
                    "— prior output was empty or reasoning-only"
                )
            retry_system = (
                "You are a helpful vision-language assistant.\n"
                "Return ONLY the final direct answer to the user.\n"
                "No analysis, no planning steps, no first-person, and no <think>."
            )
            retry_user = (
                "Rewrite the following into the final direct answer for the user:\n\n"
                f"{current_raw}\n"
            )
            stage_label = f"FINALIZATION ATTEMPT {attempt_number}/{_GGUF_MAX_FINALIZATION_ATTEMPTS}"
            cleaned_retry, raw_retry = _run_completion(
                retry_system,
                retry_user,
                [],
                stage_label=stage_label,
                seed_value=int(seed) + 999 + attempt_number,
                attempt_enable_thinking=False,
            )
            raw_trace_parts.append(f"[{stage_label}]\n{raw_retry}")
            if cleaned_retry and len(cleaned_retry) >= len(best_cleaned):
                best_cleaned = cleaned_retry
            if _answer_output_is_usable(cleaned_retry):
                return cleaned_retry, "\n\n".join(raw_trace_parts)
            current_raw = raw_retry

        if stream_to_terminal:
            print(
                "[QwenVL GGUF] "
                f"Finalization limit reached ({_GGUF_MAX_FINALIZATION_ATTEMPTS}/{_GGUF_MAX_FINALIZATION_ATTEMPTS}) "
                "— returning the best cleaned answer available"
            )
        return best_cleaned or "", "\n\n".join(raw_trace_parts)

    def _encode_media(self, image, video, frame_count: int) -> list[str]:
        images_b64: list[str] = []
        if image is not None:
            print(f"[QwenVL GGUF DEBUG] Processing image...")
            print(f"[QwenVL GGUF DEBUG] Image shape before processing: {image.shape}")

            if len(image.shape) == 4:
                print(f"[QwenVL GGUF DEBUG] Detected batch image with shape: {image.shape}")
                frame_img = image[0]
                if image.shape[0] > 1:
                    print(f"[QwenVL GGUF DEBUG] IMAGE input contains {image.shape[0]} items; using the first item only. Use the video input for multi-frame analysis.")
                print(f"[QwenVL GGUF DEBUG] Single image from batch, shape: {frame_img.shape}")
                img = _tensor_to_base64_png(frame_img)
                if img:
                    images_b64.append(img)
            else:
                print(f"[QwenVL GGUF DEBUG] Regular single image, shape: {image.shape}")
                img = _tensor_to_base64_png(image)
                if img:
                    images_b64.append(img)

        if video is not None:
            sampled_frames = _sample_video_frames(video, int(frame_count))
            if video.ndim == 4:
                print(f"[QwenVL GGUF DEBUG] Sampled {len(sampled_frames)} frame(s) from {int(video.shape[0])} total video frames")
            for frame in sampled_frames:
                img = _tensor_to_base64_png(frame)
                if img:
                    images_b64.append(img)

        return images_b64

    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
        keep_model_loaded,
        device,
        ctx=None,
        n_batch=None,
        gpu_layers=None,
        image_max_tokens=None,
        top_k=None,
        pool_size=None,
        keep_last_prompt=False,
        unique_id=None,
        extra_pnginfo=None,
        node_class="QwenVLGGUF",
        stream_to_terminal=False,
        enable_thinking=True,
    ):
        print(f"[QwenVL GGUF DEBUG] Starting run with seed={seed}, keep_last_prompt={keep_last_prompt}")
        image_hash = get_image_hash(image)
        video_hash = get_video_hash(video)
        input_signature = build_node_input_signature(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image_hash=image_hash,
            video_hash=video_hash,
            frame_count=frame_count,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
            device=device,
            enable_thinking=bool(enable_thinking),
        )

        # Auto-retrieve saved prompt when seed is fixed
        saved = get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=int(seed), max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        if keep_last_prompt:
            print(f"[QwenVL GGUF] Keep last prompt enabled — looking up per-node state")
            if saved:
                print(f"[QwenVL GGUF] Using per-node prompt: {saved[:50]}...")
                return (saved,)
            else:
                print(f"[QwenVL GGUF] No per-node prompt found, returning empty")
                return ("",)

        # Always generate unless keep_last_prompt was requested
        print(f"[QwenVL GGUF] Generating new prompt")

        prompt_template = "" if preset_prompt == "🚫 No preset (image-only)" else SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)

        # Generate cache key with all inputs including seed
        cache_key = get_cache_key(model_name, preset_prompt, custom_prompt, image_hash, video_hash, int(seed), max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)

        # TEMPORARILY DISABLED CACHE FOR DEBUGGING
        # Check cache first (only for random mode)
        # if cache_key in PROMPT_CACHE:
        #     cached_text = PROMPT_CACHE[cache_key].get("text", "")
        #     if cached_text:
        #         print(f"[QwenVL GGUF] Using cached prompt for seed {seed}: {cache_key[:8]}...")
        #         return cached_text.strip()

        print(f"[QwenVL GGUF DEBUG] Cache disabled - proceeding with generation")

        if custom_prompt and custom_prompt.strip():
            # Combine user input with template - custom prompt first for priority
            prompt = f"{custom_prompt.strip()}\n\n{prompt_template}" if prompt_template else custom_prompt.strip()
        else:
            prompt = prompt_template

        print(f"[QwenVL GGUF DEBUG] Final prompt: {prompt[:100]}...")

        attempt_settings: list[tuple[int, int | None, int | None]] = [
            (int(frame_count), n_batch, image_max_tokens),
        ]
        if video is not None:
            for retry_frame_count, retry_n_batch, retry_image_max_tokens in _build_video_retry_settings(
                int(frame_count),
                n_batch,
                image_max_tokens,
            ):
                candidate = (retry_frame_count, retry_n_batch, retry_image_max_tokens)
                if candidate not in attempt_settings:
                    attempt_settings.append(candidate)

        # Debug VRAM before model loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            print(f"[QwenVL GGUF DEBUG] VRAM before loading: {allocated/1024**3:.2f}GB / {total/1024**3:.2f}GB")

        text = None
        last_exc: Exception | None = None
        resolved = _resolve_model_entry(model_name)
        context_window = int(ctx) if ctx is not None else resolved.context_length

        try:
            for attempt_index, (attempt_frame_count, attempt_n_batch, attempt_image_max_tokens) in enumerate(attempt_settings):
                images_b64 = self._encode_media(image, video, attempt_frame_count)

                print(f"[QwenVL GGUF DEBUG] Images processed: {len(images_b64)} images/videos")
                if video is not None:
                    print(f"[QwenVL GGUF DEBUG] Video shape: {video.shape}")
                    print(f"[QwenVL GGUF DEBUG] Frame count requested: {frame_count}")
                    print(
                        f"[QwenVL GGUF DEBUG] Attempt {attempt_index + 1}: "
                        f"frame_count={attempt_frame_count}, n_batch={attempt_n_batch}, image_max_tokens={attempt_image_max_tokens}"
                    )
                if image is not None:
                    print(f"[QwenVL GGUF DEBUG] Image shape: {image.shape}")

                try:
                    estimated_prompt_tokens = estimate_qwen_text_tokens(
                        "You are a helpful vision-language assistant.",
                        prompt,
                    ) + len(images_b64) * max(1024, int(attempt_image_max_tokens or 1024))
                    effective_thinking = resolve_qwen_thinking_mode(
                        enable_thinking,
                        max_tokens,
                        label="QwenVL GGUF",
                        prompt_tokens=estimated_prompt_tokens,
                        context_window=context_window,
                    )
                    print(f"[QwenVL GGUF DEBUG] Loading model...")
                    self._load_model(
                        model_name=model_name,
                        device=device,
                        ctx=ctx,
                        n_batch=attempt_n_batch,
                        gpu_layers=gpu_layers,
                        image_max_tokens=attempt_image_max_tokens,
                        top_k=top_k,
                        pool_size=pool_size,
                        enable_thinking=effective_thinking,
                    )
                    print(f"[QwenVL GGUF DEBUG] Model loaded successfully")
                    if images_b64 and self.chat_handler is None:
                        print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
                    print(f"[QwenVL GGUF DEBUG] Starting generation...")
                    text, raw_trace = self._invoke(
                        system_prompt=(
                            "You are a helpful vision-language assistant. "
                            "Answer directly with the final answer only. No <think> and no reasoning."
                        ) if not effective_thinking else (
                            "You are a helpful vision-language assistant."
                        ),
                        user_prompt=prompt,
                        images_b64=images_b64 if self.chat_handler is not None else [],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                        model_name=model_name,
                        stream_to_terminal=stream_to_terminal,
                        enable_thinking=effective_thinking,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    is_last_attempt = attempt_index == len(attempt_settings) - 1
                    if video is None or not _is_media_capacity_error(exc) or is_last_attempt:
                        raise
                    print(
                        "[QwenVL GGUF] Video media capacity limit reached; retrying with reduced frame/token budget. "
                        f"Cause: {exc}"
                    )
                    self.clear()

            if text is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("[QwenVL GGUF] Generation failed without returning text")

            print(f"[QwenVL GGUF DEBUG] Generation completed. Text length: {len(text) if text else 0}")
            print(f"[QwenVL GGUF DEBUG] Generated text: {text[:100] if text else 'EMPTY'}...")

            # Cache the generated text
            PROMPT_CACHE[cache_key] = {
                "text": text,
                "timestamp": None,  # GGUF doesn't have CUDA events
                "model": model_name,
                "preset": preset_prompt,
                "seed": int(seed),
                "image_hash": image_hash,
                "video_hash": video_hash
            }
            save_prompt_cache()  # Save cache to file

            print(f"[QwenVL GGUF] Cached new prompt for seed {seed}: {cache_key[:8]}...")

            print(f"[QwenVL GGUF DEBUG] Returning tuple with text...")

            # Save the generated prompt for future per-node keep-last-prompt
            set_node_saved_prompt(node_class, unique_id, extra_pnginfo, text, raw_trace=raw_trace, seed=int(seed), max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
            print(f"[QwenVL GGUF] Saved per-node prompt: {text[:50]}...")

            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(no GGUF VL models found)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["🚫 No preset (image-only)", "🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": "Select 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional user input that gets combined with the preset template. Leave empty to use only the template."}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Print every generated token live to the ComfyUI terminal/console"}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen GGUF models this is advisory and may not be honored by the backend."}),
                            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def process(
        self,
        model_name,
        preset_prompt,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        image=None,
        video=None,
        unique_id=None,
        extra_pnginfo=None,
        stream_tokens_to_terminal=False,
        enable_thinking=True,
    ):
        result = self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.0,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            gpu_layers=None,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
            keep_last_prompt=keep_last_prompt,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="AILab_QwenVL_GGUF",
            stream_to_terminal=stream_tokens_to_terminal,
            enable_thinking=enable_thinking,
        )
        # Read back raw_trace from just-saved per-node state
        key = _make_node_state_key("AILab_QwenVL_GGUF", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        return (result[0], raw_trace)


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(no GGUF VL models found)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["🚫 No preset (image-only)", "🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "device": (device_options, {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": "Select 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional user input that gets combined with the preset template. Leave empty to use only the template."}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Print every generated token live to the ComfyUI terminal/console"}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen GGUF models this is advisory."}),
                            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def process(
        self,
        model_name,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        top_k,
        pool_size,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        image=None,
        video=None,
        unique_id=None,
        extra_pnginfo=None,
        stream_tokens_to_terminal=False,
        enable_thinking=True,
    ):
        result = self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=frame_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
            keep_last_prompt=keep_last_prompt,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="AILab_QwenVL_GGUF_Advanced",
            stream_to_terminal=stream_tokens_to_terminal,
            enable_thinking=enable_thinking,
        )
        # Read back raw_trace from just-saved per-node state
        key = _make_node_state_key("AILab_QwenVL_GGUF_Advanced", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        return (result[0], raw_trace)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF": AILab_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF": "ThinkingLLM (GGUF)",
    "AILab_QwenVL_GGUF_Advanced": "ThinkingLLM Advanced (GGUF)",
}
