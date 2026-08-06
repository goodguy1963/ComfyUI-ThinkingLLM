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
import re
import shutil
import struct
import subprocess
import sys
import time
import weakref
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from AILab_StreamDisplay import (
    StreamDegenerationError,
    StreamDegenerationGuard,
    TerminalStreamDisplay,
    extract_stream_token,
    strip_degenerate_repetition,
)
from AILab_LlamaCppInstaller import (
    ensure_llama_cpp_backend,
    format_llama_cpp_backend_info,
    get_last_llama_cpp_backend_info,
    relax_windows_dll_directory_for_long_paths,
)
import comfy.model_management as comfy_model_management
from comfy.model_management import throw_exception_if_processing_interrupted

# Import cache functions from main module
sys.path.append(str(Path(__file__).parent))
from AILab_QwenVL import (
    PROMPT_CACHE,
    MASK_FOCUS_INSTRUCTION,
    apply_mask_highlight,
    apply_qwen_soft_thinking_directive,
    build_node_input_signature,
    download_hf_file_to_path,
    ensure_cuda_vram_headroom,
    estimate_qwen_text_tokens,
    get_cache_key,
    get_alternative_cache_key,
    get_image_hash,
    get_video_hash,
    log_llm_input,
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

# Load per-node state at startup for per-node prompt and trace access
load_node_prompt_state()

_GEMMA4_MULTIMODAL_MAX_CTX = 32768
_GEMMA4_MULTIMODAL_IMAGE_TOKENS = 1120
_GEMMA4_MULTIMODAL_MAX_BATCH = 2048


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


def construct_llama_safely(Llama, kwargs: dict, label: str):
    original_del = getattr(Llama, "__del__", None)
    patched_del = False

    def guarded_del(instance):
        if not getattr(instance, "_thinkingllm_llama_init_complete", False):
            return
        if callable(original_del):
            try:
                original_del(instance)
            except BaseException as exc:
                print(f"[{label}] Warning: llama.cpp cleanup failed: {exc}")

    try:
        if callable(original_del):
            setattr(Llama, "__del__", guarded_del)
            patched_del = True
        llm = Llama(**kwargs)
        try:
            setattr(llm, "_thinkingllm_llama_init_complete", True)
        except Exception:
            pass
        return llm
    except Exception as exc:
        raise RuntimeError(
            f"[{label}] llama.cpp failed to create a context for this GGUF model. "
            "Free VRAM, reduce context length, select a smaller quant, or disable GPU offload."
        ) from exc
    finally:
        if patched_del:
            try:
                setattr(Llama, "__del__", original_del)
            except Exception:
                pass


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
    local_filenames: list[str] = field(default_factory=list)
    mmproj_local_filenames: list[str] = field(default_factory=list)


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


def _gguf_search_dirs(base_dir_value: str) -> list[Path]:
    search_dirs: list[Path] = [_resolve_base_dir(base_dir_value)]
    try:
        if "LLM" in folder_paths.folder_names_and_paths:
            for llm_path in folder_paths.get_folder_paths("LLM"):
                gguf_dir = Path(llm_path) / "GGUF"
                if gguf_dir not in search_dirs:
                    search_dirs.append(gguf_dir)

                llm_dir = Path(llm_path)
                if llm_dir not in search_dirs:
                    search_dirs.append(llm_dir)
    except Exception:
        pass
    return search_dirs


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


def _as_filename_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []

    filenames: list[str] = []
    for item in values:
        name = Path(str(item)).name
        if name and name not in filenames:
            filenames.append(name)
    return filenames


def _filename_aliases_for(payload, key: str | None) -> list[str]:
    if not isinstance(payload, dict) or not key:
        return []
    return _as_filename_list(payload.get(key) or payload.get(Path(key).name))


def _filename_search_candidates(filename: str, aliases: list[str] | None = None) -> list[str]:
    return _as_filename_list([filename, *(aliases or [])])


def _is_removed_qwen3_asr_filename(filename: str) -> bool:
    return "qwen3-asr" in str(filename or "").lower()


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
            if _is_removed_qwen3_asr_filename(model_file.name):
                continue
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
        mmproj_files = repo.get("mmproj_files") or {}
        local_filenames = repo.get("local_filenames") or repo.get("local_model_files") or {}
        mmproj_local_filenames = repo.get("mmproj_local_filenames") or repo.get("local_mmproj_files") or {}
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
            resolved_mmproj_file = mmproj_files.get(model_file) or mmproj_file
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "local_filenames": _filename_aliases_for(local_filenames, model_file),
                "mmproj_filename": resolved_mmproj_file,
                "mmproj_local_filenames": _filename_aliases_for(mmproj_local_filenames, resolved_mmproj_file)
                + _filename_aliases_for(mmproj_local_filenames, model_file),
            }

    legacy_models = data.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            model_files = entry.get("model_files") or []
            if model_files and not entry.get("filename"):
                defaults = entry.get("defaults") or {}
                mmproj_file = entry.get("mmproj_file")
                mmproj_files = entry.get("mmproj_files") or {}
                local_filenames = entry.get("local_filenames") or entry.get("local_model_files") or {}
                mmproj_local_filenames = entry.get("mmproj_local_filenames") or entry.get("local_mmproj_files") or {}
                author = entry.get("author") or entry.get("publisher")
                repo_name = entry.get("repo_name") or entry.get("repo_name_override") or name
                repo_id = entry.get("repo_id")
                alt_repo_ids = entry.get("alt_repo_ids") or []
                quant_sizes = _parse_repo_quant_sizes(name)
                for model_file in model_files:
                    display = Path(model_file).name
                    q = _quant_from_filename(model_file)
                    if q and q in quant_sizes:
                        display = f"{display} [~{quant_sizes[q]}]"
                    if display in flattened:
                        display = f"{display} ({name})"
                    resolved_mmproj_file = mmproj_files.get(model_file) or mmproj_file
                    flattened[display] = {
                        **defaults,
                        "author": author,
                        "repo_dirname": repo_name,
                        "repo_id": repo_id,
                        "alt_repo_ids": alt_repo_ids,
                        "filename": model_file,
                        "local_filenames": _filename_aliases_for(local_filenames, model_file),
                        "mmproj_filename": resolved_mmproj_file,
                        "mmproj_local_filenames": _filename_aliases_for(mmproj_local_filenames, resolved_mmproj_file)
                        + _filename_aliases_for(mmproj_local_filenames, model_file),
                    }
            else:
                flattened[name] = entry

    # Mark catalog entries already present on disk, then add uncatalogued local models.
    search_dirs = _gguf_search_dirs(base_dir)
    installed_filenames: set[str] = set()
    for directory in search_dirs:
        try:
            installed_filenames.update(path.name for path in directory.rglob("*.gguf") if path.is_file())
        except (OSError, PermissionError):
            pass
    for display, entry in list(flattened.items()):
        filenames = _filename_search_candidates(entry.get("filename", ""), entry.get("local_filenames"))
        if any(filename in installed_filenames for filename in filenames):
            flattened.pop(display)
            flattened[f"{display} [installed]"] = {**entry, "catalog_display_name": display}

    existing_filenames = {Path(e.get("filename", "")).name for e in flattened.values() if e.get("filename")}
    for scan_dir in search_dirs:
        local_models = _scan_local_gguf_models(scan_dir, existing_filenames)
        flattened.update(local_models)
        # Update existing_filenames so we don't add duplicates across directories
        existing_filenames.update(Path(e.get("filename", "")).name for e in local_models.values() if e.get("filename"))

    return {"base_dir": base_dir, "models": flattened}


GGUF_VL_CATALOG = _load_gguf_vl_catalog()

GGUF_TOOLTIPS = {
    "model_name": "GGUF vision model from gguf_models.json or auto-detected local files. [installed] means the catalog model file was found on disk; [local] means an uncatalogued local model. Missing GGUF or mmproj files are downloaded on first use.",
    "audio_model_name": "Gemma 4 audio-capable GGUF model. Only Gemma 4 E2B, E4B, and 12B are listed; 26B/31B variants are image/text-only for this purpose.",
    "audio_file_path": "Optional local audio file path. M4A, MP3, WAV, FLAC, and other FFmpeg-readable files are decoded to 16 kHz mono WAV before inference.",
    "device": "auto prefers CUDA when PyTorch sees an NVIDIA GPU. If RAW_TRACE says GPU offload is no or unknown, verify your llama-cpp-python CUDA wheel before blaming the model.",
    "max_tokens": "Maximum new tokens to generate. Larger values give more room for reasoning but increase runtime and memory use.",
    "keep_model_loaded": "Keep the GGUF model in RAM/VRAM after the run so repeated prompts skip model loading. Disable if you need memory back for other nodes.",
    "seed": "Sampling seed. The node also uses fixed-seed prompt persistence, so identical inputs can reuse the saved result.",
    "frame_count": "Number of video frames to sample. More frames improve video context but raise image-token, batch, and context pressure.",
    "ctx": "llama.cpp context window. Too large can reduce speed and increase KV-cache memory even on strong GPUs.",
    "n_batch": "Prompt processing batch size. Higher can improve prompt ingestion but may raise memory use or fail with image/video inputs.",
    "gpu_layers": "Number of model layers to offload to GPU. -1 asks llama.cpp to offload all possible layers; 0 is CPU-only.",
    "image_max_tokens": "Upper token budget for each image/video frame. Lower it if multimodal decode fails or VRAM use is too high.",
    "top_k": "llama.cpp sampler top-k. 0 disables top-k filtering; 20 is a conservative default.",
    "pool_size": "llama.cpp memory pool size for multimodal work. Increase only when backend errors point at pool/context capacity.",
    "n_ubatch": "Physical batch size. Keep at or below n_batch. Lower values can improve stability; 0 uses min(n_batch, 512).",
    "n_threads": "CPU generation threads. On high-core/NUMA servers, auto can be slower; try 8-16 if GPU utilization is low.",
    "n_threads_batch": "CPU prompt/batch threads. Tune separately from generation threads on server CPUs.",
    "flash_attn": "Enable llama.cpp flash attention when the installed backend accepts and supports it. RAW_TRACE reports if the kwarg was dropped.",
    "offload_kqv": "Keep K/Q/V and KV-cache related work on GPU when supported. RAW_TRACE warns when the backend drops this kwarg.",
    "ctx_checkpoints": "Checkpoint count for multimodal context handling. JamePeng builds usually recommend 0 for single-turn ComfyUI runs.",
    "stream_tokens_to_terminal": "Print generated tokens live in the ComfyUI terminal. Useful for long runs and backend troubleshooting.",
    "hf_token": "Optional Hugging Face access token for private or gated GGUF/mmproj downloads. It is passed only to the download call, never logged or cached, and the in-memory copy is dropped after the download attempt. Clear this field before saving or sharing workflows.",
}

GEMMA4_AUDIO_DEFAULT_PROMPT = (
    "Audio analysis: transcribe the speech in the original language, then summarize the important points. "
    "If there is no clear speech, describe the audible scene and relevant sounds."
)

GGUF_AUDIO_TARGET_SAMPLE_RATE = 16000


def _is_gemma4_audio_model_name(model_name: str) -> bool:
    lowered = str(model_name or "").lower()
    if "gemma-4" not in lowered and "gemma4" not in lowered:
        return False
    if "26b" in lowered or "31b" in lowered or "a4b" in lowered:
        return False
    return any(token in lowered for token in ("12b", "e2b", "e4b"))


def _gguf_audio_model_keys() -> list[str]:
    all_models = GGUF_VL_CATALOG.get("models") or {}
    keys = [
        key
        for key, entry in all_models.items()
        if (entry or {}).get("mmproj_filename") and _is_gemma4_audio_model_name(key)
    ]
    return sorted(keys)


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


def _unexpected_kwarg_names_from_type_error(exc: TypeError) -> list[str]:
    message = str(exc or "")
    matches = re.findall(r"unexpected keyword argument(?:\(s\))?\s+'([^']+)'", message)
    names: list[str] = []
    for match in matches:
        for name in str(match).split(","):
            cleaned = name.strip()
            if cleaned:
                names.append(cleaned)
    return names


def _filter_mmproj_handler_kwargs(handler_cls, kwargs: dict) -> dict:
    filtered = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), kwargs)
    handler_name = getattr(handler_cls, "__name__", "")
    if "Qwen" not in handler_name:
        filtered.pop("force_reasoning", None)
    return filtered


def _select_mmproj_handler_class(model_name: str, arch: str | None):
    """Pick the llama-cpp-python multimodal chat handler for the selected model."""
    try:
        from llama_cpp import llama_chat_format
    except ImportError as exc:
        raise RuntimeError(
            "[QwenVL] Missing llama_cpp chat handlers. Install a multimodal-capable llama-cpp-python build. "
            "See docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md"
        ) from exc

    name = (model_name or "").lower()
    architecture = (arch or "").lower()
    if "gemma-4" in name or "gemma4" in name or architecture == "gemma4":
        preferred = ("Gemma4ChatHandler",)
    elif "qwen2.5" in name or "qwen25" in name or architecture in ("qwen2vl", "qwen25vl"):
        preferred = ("Qwen25VLChatHandler", "Qwen3VLChatHandler")
    else:
        preferred = ("Qwen3VLChatHandler", "Qwen25VLChatHandler")

    for handler_name in preferred:
        handler_cls = getattr(llama_chat_format, handler_name, None)
        if handler_cls is not None:
            return handler_cls

    raise RuntimeError(
        "[QwenVL] Missing multimodal chat handler for this model. "
        f"Tried: {', '.join(preferred)}. Install/update llama-cpp-python."
    )


def _resample_mono_audio(samples, source_rate: int, target_rate: int = GGUF_AUDIO_TARGET_SAMPLE_RATE):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0 or source_rate == target_rate:
        return samples
    target_length = max(1, int(round(float(samples.size) * float(target_rate) / float(source_rate))))
    if target_length == samples.size:
        return samples
    if samples.size == 1:
        return np.full((target_length,), float(samples[0]), dtype=np.float32)
    old_positions = np.linspace(0.0, float(samples.size - 1), num=samples.size, dtype=np.float64)
    new_positions = np.linspace(0.0, float(samples.size - 1), num=target_length, dtype=np.float64)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def _clean_audio_file_path(audio_file_path) -> Path | None:
    if audio_file_path is None:
        return None
    raw_path = str(audio_file_path).strip().strip("\"'")
    if not raw_path:
        return None
    expanded = os.path.expandvars(raw_path)
    return Path(expanded).expanduser()


def _audio_file_path_to_wav_base64(audio_file_path) -> list[str]:
    path = _clean_audio_file_path(audio_file_path)
    if path is None:
        return []
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"audio_file_path does not exist or is not a file: {path}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH; install FFmpeg or connect a decoded ComfyUI AUDIO input")

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(GGUF_AUDIO_TARGET_SAMPLE_RATE),
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio_file_path: {stderr or 'unknown ffmpeg error'}")
    if not result.stdout:
        raise RuntimeError("ffmpeg decoded no audio samples from audio_file_path")

    if len(result.stdout) % 2 != 0:
        raise RuntimeError("ffmpeg produced malformed 16-bit PCM audio")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(GGUF_AUDIO_TARGET_SAMPLE_RATE)
        wav.writeframes(result.stdout)
    return [base64.b64encode(buffer.getvalue()).decode("ascii")]


def _audio_to_wav_base64(audio) -> list[str]:
    if audio is None:
        return []
    try:
        if not isinstance(audio, dict):
            raise ValueError("expected ComfyUI AUDIO dict with waveform/sample_rate")
        waveform = audio.get("waveform")
        if waveform is None:
            raise ValueError("missing waveform")
        sample_rate = int(audio.get("sample_rate") or 0)
        if sample_rate <= 0:
            raise ValueError(f"invalid sample_rate={audio.get('sample_rate')!r}")
        tensor = waveform.detach().cpu() if hasattr(waveform, "detach") else waveform
        tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        array = np.asarray(array, dtype=np.float32)
        if array.size == 0:
            raise ValueError("empty waveform")
        if array.ndim == 3:
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"unsupported waveform shape={getattr(array, 'shape', None)}")
        if array.shape[0] > array.shape[-1]:
            array = array.T
        if not np.all(np.isfinite(array)):
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        mono = np.mean(array, axis=0)
        mono = _resample_mono_audio(mono, sample_rate, GGUF_AUDIO_TARGET_SAMPLE_RATE)
        pcm = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(GGUF_AUDIO_TARGET_SAMPLE_RATE)
            wav.writeframes(pcm16.tobytes())
        return [base64.b64encode(buffer.getvalue()).decode("ascii")]
    except Exception as exc:
        print(f"[QwenVL] Warning: failed to encode AUDIO input as WAV; audio will be ignored. Cause: {exc}")
        return []


def _audio_inputs_to_wav_base64(audio=None, audio_file_path="") -> tuple[list[str], list[str]]:
    items: list[str] = []
    notes: list[str] = []
    if audio is not None:
        converted = _audio_to_wav_base64(audio)
        if converted:
            items.extend(converted)
        else:
            notes.append("Connected ComfyUI AUDIO input could not be converted to 16 kHz WAV.")

    clean_path = _clean_audio_file_path(audio_file_path)
    if clean_path is not None:
        try:
            converted = _audio_file_path_to_wav_base64(str(clean_path))
            if converted:
                items.extend(converted)
                print(f"[QwenVL] Audio file decoded: {clean_path}")
        except Exception as exc:
            note = f"audio_file_path failed: {exc}"
            notes.append(note)
            print(f"[QwenVL] Warning: {note}")

    return items, notes


def _get_audio_hash(audio) -> str | None:
    if audio is None or not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None:
        return None
    try:
        shape = tuple(getattr(waveform, "shape", ()))
        tensor = waveform.detach().cpu() if hasattr(waveform, "detach") else waveform
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        sample = array.reshape(-1)[:32].tolist()
        content = json.dumps({"shape": shape, "sample_rate": sample_rate, "sample": sample}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.md5(f"{sample_rate}|{getattr(waveform, 'shape', '')}".encode()).hexdigest()[:16]


def _get_audio_file_hash(audio_file_path) -> str | None:
    path = _clean_audio_file_path(audio_file_path)
    if path is None:
        return None
    try:
        stat = path.stat()
        content = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:16]
    except Exception:
        return hashlib.md5(str(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def _missing_audio_result(notes: list[str] | None = None) -> tuple[str, str]:
    message = (
        "[QwenVL GGUF] No usable audio input. Connect a decoded ComfyUI AUDIO output "
        "or set audio_file_path to an existing M4A/MP3/WAV/FLAC file."
    )
    trace_lines = ["[AUDIO]", message]
    if notes:
        trace_lines.extend(str(note) for note in notes if note)
    return message, "\n".join(trace_lines)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _current_cuda_device_index() -> int | None:
    try:
        if not torch.cuda.is_available():
            return None
        try:
            comfy_device = comfy_model_management.get_torch_device()
            if getattr(comfy_device, "type", None) == "cuda" and getattr(comfy_device, "index", None) is not None:
                return int(comfy_device.index)
        except Exception:
            pass
        return int(torch.cuda.current_device())
    except Exception:
        return None


def _cuda_device_label(index: int | None = None) -> str:
    if index is None:
        index = _current_cuda_device_index()
    if index is None:
        return "cuda:unknown"
    try:
        props = torch.cuda.get_device_properties(index)
        name = getattr(props, "name", "") or torch.cuda.get_device_name(index)
        total_gb = float(getattr(props, "total_memory", 0) or 0) / 1024**3
        return f"cuda:{index} {name} ({total_gb:.2f}GB)"
    except Exception:
        return f"cuda:{index}"


def _format_backend_trace(
    backend_info: dict | None,
    *,
    model_path: Path,
    device_kind: str,
    n_gpu_layers: int,
    main_gpu: int | None,
    n_ctx: int,
    n_batch: int,
    n_ubatch: int,
    n_threads: int | None,
    n_threads_batch: int | None,
    flash_attn: bool,
    offload_kqv: bool,
    accepted_kwargs: list[str],
    dropped_perf_kwargs: list[str],
    warnings: list[str],
) -> str:
    lines = [
        "[BACKEND]",
        format_llama_cpp_backend_info(backend_info),
        (
            f"model={model_path.name}; device={device_kind}; gpu_layers={n_gpu_layers}; "
            f"main_gpu={main_gpu if main_gpu is not None else 'auto'}; "
            f"ctx={n_ctx}; batch={n_batch}; ubatch={n_ubatch}"
        ),
        (
            f"threads={n_threads if n_threads is not None else 'auto'}; "
            f"threads_batch={n_threads_batch if n_threads_batch is not None else 'auto'}; "
            f"flash_attn={bool(flash_attn)}; offload_kqv={bool(offload_kqv)}"
        ),
        "accepted_kwargs=" + (", ".join(accepted_kwargs) if accepted_kwargs else "unknown"),
        "dropped_performance_kwargs=" + (", ".join(dropped_perf_kwargs) if dropped_perf_kwargs else "none"),
    ]
    if warnings:
        lines.append("warnings=" + " | ".join(warnings))
        advice: list[str] = []
        warning_text = " ".join(warnings).lower()
        if "no gpu offload" in warning_text or "cpu-only" in warning_text:
            advice.append("Install a CUDA-enabled llama-cpp-python vision wheel or rebuild with GGML_CUDA=on, then rerun tools/check_llama_backend.py --strict-gpu.")
        if "unknown" in warning_text:
            advice.append("Set THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD=1 and check llama.cpp logs for offloaded layers, CUDA buffers, KV buffers, and flash_attn status.")
        if "performance kwargs" in warning_text:
            advice.append("Update to a llama-cpp-python/JamePeng build that accepts flash_attn, offload_kqv, n_ubatch, and thread kwargs.")
        if "automatic thread counts" in warning_text:
            advice.append("On large Linux servers, start with n_threads=8-16 and tune n_threads_batch while watching token/s and GPU utilization.")
        if "chat_handler" in warning_text:
            advice.append("Install a multimodal llama-cpp-python build with Qwen vision chat handlers so image/video inputs are used.")
        if advice:
            lines.append("recommended_actions=" + " | ".join(dict.fromkeys(advice)))
    return "\n".join(lines)


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


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path, *, node_id=None, progress_label: str = "QwenVL GGUF Download", hf_token: str | None = None):
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    download_hf_file_to_path(
        repo_ids,
        filename,
        target_path,
        node_id=node_id,
        progress_label=progress_label,
        hf_token=hf_token,
    )
    hf_token = None

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


def _find_existing_local_file(base_dir: Path, filename: str, aliases: list[str] | None = None) -> Path | None:
    """Reuse an already-downloaded GGUF file anywhere under the shared base dir."""
    for wanted in _filename_search_candidates(filename, aliases):
        try:
            for candidate in base_dir.rglob(wanted):
                if candidate.is_file():
                    return candidate
        except Exception:
            continue
    return None


def _find_existing_local_file_in_dirs(search_dirs: list[Path], filename: str, aliases: list[str] | None = None) -> Path | None:
    for search_dir in search_dirs:
        found = _find_existing_local_file(search_dir, filename, aliases)
        if found is not None:
            return found
    return None


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    all_models = GGUF_VL_CATALOG.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = _model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            if candidate.get("catalog_display_name") == model_name:
                entry = candidate
                break
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
        local_filenames=_as_filename_list(entry.get("local_filenames") or entry.get("local_model_files")),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        mmproj_local_filenames=_as_filename_list(entry.get("mmproj_local_filenames") or entry.get("local_mmproj_files")),
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
        self.last_backend_trace = ""
        self.uses_qwen_template_thinking = False
        self.current_backend_gpu_offload = None
        self.current_batch_size = None
        self.current_ubatch_size = None
        self.current_image_token_budget = None
        self.current_context_length = None
        self.gguf_arch = None
        register_active_gguf_loader(self)

    def _uses_chat_template_thinking(self) -> bool:
        return bool(getattr(self, "uses_qwen_template_thinking", False))

    def _sync_live_chat_template_kwargs(self, enable_thinking: bool) -> None:
        if self.llm is None or not self._uses_chat_template_thinking():
            return
        template_kwargs = {"enable_thinking": bool(enable_thinking)}
        for attr_name in ("chat_template_kwargs", "_chat_template_kwargs"):
            try:
                current = getattr(self.llm, attr_name, None)
                if isinstance(current, dict):
                    current.update(template_kwargs)
                else:
                    setattr(self.llm, attr_name, dict(template_kwargs))
            except Exception:
                continue

    def _create_chat_completion(self, *, enable_thinking: bool, **kwargs):
        self._sync_live_chat_template_kwargs(enable_thinking)
        kwargs = dict(kwargs)
        if self._uses_chat_template_thinking():
            kwargs["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        completion_fn = getattr(self.llm, "create_chat_completion")
        kwargs = _filter_kwargs_for_callable(completion_fn, kwargs)

        for _ in range(4):
            try:
                return completion_fn(**kwargs)
            except TypeError as exc:
                unexpected = [name for name in _unexpected_kwarg_names_from_type_error(exc) if name in kwargs]
                if not unexpected:
                    raise
                for name in unexpected:
                    kwargs.pop(name, None)
                print(
                    "[QwenVL GGUF] llama-cpp-python rejected chat completion kwarg(s): "
                    f"{', '.join(unexpected)}; retrying without them."
                )

        return completion_fn(**kwargs)

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
        n_ubatch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        n_threads: int | None,
        n_threads_batch: int | None,
        flash_attn: bool,
        offload_kqv: bool,
        ctx_checkpoints: int | None,
        enable_thinking: bool = True,
        hf_token: str | None = None,
        unique_id=None,
    ):
        Llama = self._load_backend()
        backend_info = get_last_llama_cpp_backend_info()

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
            base_dir_value = GGUF_VL_CATALOG.get("base_dir") or "llm/GGUF"
            search_dirs = _gguf_search_dirs(base_dir_value)
            base_dir = search_dirs[0]

            author_dir = _safe_dirname(resolved.author or "")
            repo_dir = _safe_dirname(resolved.repo_dirname)
            target_dir = base_dir / author_dir / repo_dir

            model_path = target_dir / Path(resolved.model_filename).name
            mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

            existing_model = _find_existing_local_file_in_dirs(search_dirs, resolved.model_filename, resolved.local_filenames)
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
                existing_mmproj = _find_existing_local_file_in_dirs(
                    search_dirs,
                    resolved.mmproj_filename,
                    resolved.mmproj_local_filenames,
                )
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
                _download_single_file(
                    repo_ids,
                    resolved.model_filename,
                    model_path,
                    node_id=unique_id,
                    progress_label=f"QwenVL GGUF Download: {Path(resolved.model_filename).name}",
                    hf_token=hf_token,
                )

            if mmproj_path is not None and not mmproj_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] mmproj not found locally and no repo_id provided: {mmproj_path}")
                _download_single_file(
                    repo_ids,
                    resolved.mmproj_filename,
                    mmproj_path,
                    node_id=unique_id,
                    progress_label=f"QwenVL GGUF Download: {Path(resolved.mmproj_filename).name}",
                    hf_token=hf_token,
                )
        hf_token = None

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        n_ubatch_val = int(n_ubatch) if n_ubatch is not None else 0
        if n_ubatch_val <= 0:
            n_ubatch_val = min(n_batch_val, 512)
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size
        n_threads_val = int(n_threads) if n_threads is not None and int(n_threads) > 0 else None
        n_threads_batch_val = int(n_threads_batch) if n_threads_batch is not None and int(n_threads_batch) > 0 else None
        ctx_checkpoints_val = int(ctx_checkpoints) if ctx_checkpoints is not None and int(ctx_checkpoints) >= 0 else 0
        load_warnings: list[str] = []

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0
        llama_main_gpu = _current_cuda_device_index() if device_kind == "cuda" and n_gpu_layers != 0 else None

        logical_cpus = os.cpu_count() or 0
        if logical_cpus > 32 and n_threads_val is None and n_threads_batch_val is None:
            load_warnings.append(
                f"automatic thread counts on {logical_cpus} logical CPUs may be slower on NUMA/server systems; try n_threads=8-16 and tune n_threads_batch"
            )

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()
        gpu_offload = None if backend_info is None else backend_info.get("gpu_offload")

        # Detect architecture from GGUF metadata before selecting a multimodal handler.
        arch = read_gguf_architecture(model_path)
        self.gguf_arch = arch
        is_qwen35 = arch in ("qwen35", "qwen35moe") if arch else "qwen3.5-" in model_name.lower()
        self.supports_qwen_soft_think = arch == "qwen3" if arch else "qwen3-" in model_name.lower()
        self.uses_qwen_template_thinking = bool(is_qwen35 or arch == "qwen3" or arch == "qwen")

        if arch == "gemma4" and has_mmproj:
            original_settings = (n_ctx, n_batch_val, n_ubatch_val, img_max)
            n_ctx = min(n_ctx, _GEMMA4_MULTIMODAL_MAX_CTX)
            img_max = min(img_max, _GEMMA4_MULTIMODAL_IMAGE_TOKENS)
            required_batch = max(img_max, 1)
            n_batch_val = max(n_batch_val, required_batch)
            n_batch_val = min(n_batch_val, _GEMMA4_MULTIMODAL_MAX_BATCH)
            n_ubatch_val = max(n_ubatch_val, required_batch)
            n_ubatch_val = min(n_ubatch_val, n_batch_val, _GEMMA4_MULTIMODAL_MAX_BATCH)
            downgraded_settings = (n_ctx, n_batch_val, n_ubatch_val, img_max)
            if downgraded_settings != original_settings:
                load_warnings.append(
                    "applied Gemma 4 multimodal compatibility settings: "
                    f"ctx {original_settings[0]}->{n_ctx}, "
                    f"n_batch {original_settings[1]}->{n_batch_val}, "
                    f"n_ubatch {original_settings[2]}->{n_ubatch_val}, "
                    f"image_max_tokens {original_settings[3]}->{img_max}"
                )

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_gpu_layers,
            n_batch_val,
            n_ubatch_val,
            device_kind,
            llama_main_gpu,
            img_max,
            top_k_val,
            pool_size_val,
            n_threads_val,
            n_threads_batch_val,
            bool(flash_attn),
            bool(offload_kqv),
            ctx_checkpoints_val,
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
            handler_cls = _select_mmproj_handler_class(model_name, arch)

            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = _filter_mmproj_handler_kwargs(handler_cls, mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            with relax_windows_dll_directory_for_long_paths():
                self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "n_ubatch": n_ubatch_val,
            "swa_full": True,
            "verbose": _env_bool("THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD", False),
            "pool_size": pool_size_val,
            "top_k": top_k_val,
            "n_threads": n_threads_val,
            "n_threads_batch": n_threads_batch_val,
            "flash_attn": bool(flash_attn),
            "offload_kqv": bool(offload_kqv),
            "ctx_checkpoints": ctx_checkpoints_val,
        }
        if llama_main_gpu is not None:
            llm_kwargs["main_gpu"] = llama_main_gpu

        # Thinking toggle: for Qwen3.5/4 we control enable_thinking via the chat template kwarg.
        # For non-Qwen backends (Gemma, LLaMA) the flag is advisory and may be silently ignored.
        if self.uses_qwen_template_thinking:
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
        self.current_batch_size = n_batch_val
        self.current_ubatch_size = n_ubatch_val

        print(
            f"[QwenVL] Loading GGUF: {model_path.name} "
            f"(device={device_kind}, gpu_layers={n_gpu_layers}, "
            f"main_gpu={llama_main_gpu if llama_main_gpu is not None else 'auto'}, "
            f"ctx={n_ctx}, batch={n_batch_val}, ubatch={n_ubatch_val})"
        )
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            load_warnings.append("installed llama_cpp Llama() does not accept chat_handler; images will be ignored")
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        dropped_perf_kwargs = [
            key
            for key in ("n_ubatch", "n_threads", "n_threads_batch", "flash_attn", "offload_kqv", "ctx_checkpoints")
            if key in llm_kwargs and key not in llm_kwargs_filtered
        ]
        if dropped_perf_kwargs:
            load_warnings.append(
                "installed llama_cpp Llama() does not accept performance kwargs: " + ", ".join(dropped_perf_kwargs)
            )
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept performance kwargs: "
                f"{', '.join(dropped_perf_kwargs)}"
            )
        if llama_main_gpu is not None and "main_gpu" not in llm_kwargs_filtered:
            load_warnings.append("installed llama_cpp Llama() does not accept main_gpu; cannot pin selected CUDA device")
            print("[QwenVL] Warning: installed llama_cpp Llama() does not accept main_gpu; cannot pin selected CUDA device.")
        if device_kind == "cuda" and n_gpu_layers == 0:
            load_warnings.append("device=cuda selected but n_gpu_layers=0; model will run on CPU")
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        self.current_backend_gpu_offload = gpu_offload
        if device_kind == "cuda" and n_gpu_layers != 0 and gpu_offload is False:
            load_warnings.append("backend reports no GPU offload support; CUDA request is likely CPU-only")
            print("[QwenVL] Warning: llama.cpp backend reports no GPU offload support; CUDA request is likely CPU-only.")
        elif device_kind == "cuda" and n_gpu_layers != 0 and gpu_offload is None:
            load_warnings.append("backend GPU offload support is unknown; enable THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD=1 to inspect llama.cpp load logs")

        if load_warnings:
            for warning in load_warnings:
                print(f"[QwenVL] Backend warning: {warning}")

        accepted_kwargs = sorted(llm_kwargs_filtered.keys())
        self.last_backend_trace = _format_backend_trace(
            backend_info,
            model_path=model_path,
            device_kind=device_kind,
            n_gpu_layers=n_gpu_layers,
            main_gpu=llama_main_gpu,
            n_ctx=n_ctx,
            n_batch=n_batch_val,
            n_ubatch=n_ubatch_val,
            n_threads=n_threads_val,
            n_threads_batch=n_threads_batch_val,
            flash_attn=bool(flash_attn),
            offload_kqv=bool(offload_kqv),
            accepted_kwargs=accepted_kwargs,
            dropped_perf_kwargs=dropped_perf_kwargs,
            warnings=load_warnings,
        )

        with relax_windows_dll_directory_for_long_paths():
            self.llm = construct_llama_safely(Llama, llm_kwargs_filtered, "QwenVL GGUF")
        self.current_signature = signature

    def _raise_if_unsafe_native_multimodal_path(self, images_b64: list[str], audio_b64: list[str] | None):
        if not images_b64 and not audio_b64:
            return
        arch = str(getattr(self, "gguf_arch", "") or "").lower()
        if arch != "gemma4":
            return
        if audio_b64:
            raise RuntimeError(
                "Gemma 4 GGUF audio input is disabled because the installed llama.cpp backend can abort the "
                "ComfyUI process for Gemma 4 audio. Use image/text input with Gemma 4 GGUF, or use a backend "
                "that explicitly supports Gemma 4 audio."
            )
        context_length = int(getattr(self, "current_context_length", 0) or 0)
        batch_size = int(getattr(self, "current_batch_size", 0) or 0)
        ubatch_size = int(getattr(self, "current_ubatch_size", 0) or 0)
        image_tokens = int(getattr(self, "current_image_token_budget", 0) or 0)
        if (
            context_length <= _GEMMA4_MULTIMODAL_MAX_CTX
            and image_tokens <= _GEMMA4_MULTIMODAL_IMAGE_TOKENS
            and batch_size >= max(image_tokens, 1)
            and ubatch_size >= max(image_tokens, 1)
            and batch_size <= _GEMMA4_MULTIMODAL_MAX_BATCH
            and ubatch_size <= _GEMMA4_MULTIMODAL_MAX_BATCH
        ):
            return
        raise RuntimeError(
            "Gemma 4 GGUF multimodal generation was blocked before llama.cpp could abort ComfyUI. "
            "The installed llama-cpp-python backend can abort with unsafe Gemma 4 image settings; this run uses "
            f"ctx={context_length}, batch={batch_size}, ubatch={ubatch_size}, image_max_tokens={image_tokens}. "
            f"Retry with ctx<={_GEMMA4_MULTIMODAL_MAX_CTX}, "
            f"image_max_tokens<={_GEMMA4_MULTIMODAL_IMAGE_TOKENS}, and n_batch/n_ubatch large enough "
            "to hold the image token chunk."
        )

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        audio_b64: list[str] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        model_name: str = "",
        stream_to_terminal: bool = False,
        enable_thinking: bool = True,
        auto_finalization_retry: bool = False,
        top_k: int | None = None,
        preset_name: str = "",
        media_summary: dict[str, int] | None = None,
    ):
        """Returns (cleaned_text, raw_text) tuple."""
        ensure_cuda_vram_headroom("QwenVL GGUF", min_free_gb=1.0, min_free_ratio=0.08)
        self._raise_if_unsafe_native_multimodal_path(images_b64, audio_b64 or [])
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        if supports_soft_think:
            directive = "/think" if enable_thinking else "/no_think"
            if not stream_to_terminal:
                print(f"[QwenVL] Qwen3 GGUF detected: Thinking {'enabled' if enable_thinking else 'disabled'} via chat template and {directive}.")
        def _run_completion(
            system_prompt_text: str,
            user_prompt_text: str,
            images_for_call: list[str],
            audio_for_call: list[str],
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
            if self.llm is not None and hasattr(self.llm, "reset"):
                try:
                    self.llm.reset()
                except Exception as exc:
                    print(f"[QwenVL GGUF DEBUG] llama context reset skipped: {exc}")
            if effective_user_prompt is None:
                effective_user_prompt = user_prompt_text
            if images_for_call or audio_for_call:
                content = [{"type": "text", "text": effective_user_prompt}]
                for img in images_for_call:
                    if not img:
                        continue
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                for aud in audio_for_call:
                    if not aud:
                        continue
                    content.append({"type": "input_audio", "input_audio": {"data": aud, "format": "wav"}})
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": effective_user_prompt},
                ]

            log_llm_input(
                "QwenVL GGUF",
                stage_label,
                preset_name,
                effective_user_prompt,
                system_text=system_prompt_text,
                media=(media_summary or {
                    "visual_items": sum(bool(item) for item in images_for_call),
                    "audio": sum(bool(item) for item in audio_for_call),
                }) if images_for_call or audio_for_call else {},
            )

            start = time.perf_counter()
            stop_tokens = ["<|im_end|>", "<|im_start|>"]
            sampler_kwargs = {
                "top_k": int(top_k) if top_k is not None else 60,
                "min_p": 0.01,
                "repeat_last_n": 48,
                "presence_penalty": max(0.0, min(1.5, float(repetition_penalty) - 1.0 + 1.2)),
                "frequency_penalty": 0.2,
            }
            if stream_to_terminal:
                stream_display = TerminalStreamDisplay("QwenVL GGUF", suppress_planning=True, compact=True)
                stream_display.start_stage(stage_label)
                full_text = ""
                degeneration_guard = StreamDegenerationGuard()
                stopped_for_degenerate_stream = ""
                result = self._create_chat_completion(
                    enable_thinking=current_enable_thinking,
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repeat_penalty=float(repetition_penalty),
                    seed=int(seed_value),
                    stop=stop_tokens,
                    stream=True,
                    **sampler_kwargs,
                )
                for chunk in result:
                    throw_exception_if_processing_interrupted()
                    token = extract_stream_token(chunk)
                    reasoning_token = token.get("reasoning", "")
                    content_token = token.get("content", "")
                    display_token = reasoning_token + content_token
                    if reasoning_token:
                        full_text += reasoning_token
                    if content_token:
                        full_text += content_token
                    if display_token:
                        try:
                            degeneration_guard.push(display_token)
                        except StreamDegenerationError as exc:
                            stopped_for_degenerate_stream = exc.reason
                            close_result = getattr(result, "close", None)
                            if callable(close_result):
                                close_result()
                            break
                        stream_display.push_compact(display_token)
                stream_display.end_compact()
                stream_display.end_stage()
                raw_full = strip_degenerate_repetition(full_text).strip()
                if stopped_for_degenerate_stream:
                    print(f"[QwenVL GGUF] stopped repeated-token loop: {stopped_for_degenerate_stream}")
                cleaned = clean_model_output(raw_full, OutputCleanConfig(mode="text"))
                raw_for_trace = raw_full
                if stopped_for_degenerate_stream:
                    raw_for_trace = (
                        f"[stopped repeated-token loop: {stopped_for_degenerate_stream}]\n"
                        f"{raw_full}"
                    ).strip()
                return cleaned.strip(), raw_for_trace

            # Thread-based interrupt: always stream, poll interrupt on main thread
            import threading, queue as qmod
            abort_event = threading.Event()
            chunk_queue: qmod.Queue = qmod.Queue()
            full_text_acc = ""
            worker_error: Exception | None = None
            degeneration_guard = StreamDegenerationGuard()
            stopped_for_degenerate_stream = ""

            def _stream_worker():
                nonlocal worker_error
                try:
                    result = self._create_chat_completion(
                        enable_thinking=current_enable_thinking,
                        messages=messages,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        repeat_penalty=float(repetition_penalty),
                        seed=int(seed_value),
                        stop=stop_tokens,
                        stream=True,
                        **sampler_kwargs,
                    )
                    for chunk in result:
                        if abort_event.is_set():
                            return
                        chunk_queue.put(chunk)
                    chunk_queue.put(None)
                except Exception as exc:
                    worker_error = exc
                    chunk_queue.put(None)

            worker = threading.Thread(target=_stream_worker, daemon=True)
            worker.start()
            try:
                while True:
                    try:
                        chunk = chunk_queue.get(timeout=0.25)
                    except qmod.Empty:
                        throw_exception_if_processing_interrupted()
                        continue
                    if chunk is None:
                        break
                    token = extract_stream_token(chunk)
                    reasoning_token = token.get("reasoning", "")
                    content_token = token.get("content", "")
                    if reasoning_token:
                        full_text_acc += reasoning_token
                    if content_token:
                        full_text_acc += content_token
                    display_token = reasoning_token + content_token
                    if display_token:
                        try:
                            degeneration_guard.push(display_token)
                        except StreamDegenerationError as exc:
                            stopped_for_degenerate_stream = exc.reason
                            abort_event.set()
                            break
            finally:
                abort_event.set()
                worker.join(timeout=5.0)
            if worker_error:
                raise worker_error
            throw_exception_if_processing_interrupted()
            elapsed = max(time.perf_counter() - start, 1e-6)
            raw_content = strip_degenerate_repetition(full_text_acc).strip()
            if stopped_for_degenerate_stream:
                print(f"[QwenVL GGUF] stopped repeated-token loop: {stopped_for_degenerate_stream}")
            cleaned = clean_model_output(raw_content, OutputCleanConfig(mode="text"))
            raw_for_trace = raw_content
            if stopped_for_degenerate_stream:
                raw_for_trace = (
                    f"[stopped repeated-token loop: {stopped_for_degenerate_stream}]\n"
                    f"{raw_content}"
                ).strip()
            return cleaned.strip(), raw_for_trace

        cleaned_text, raw_text = _run_completion(
            system_prompt,
            user_prompt,
            images_b64,
            audio_b64 or [],
            stage_label="STREAMING",
            seed_value=seed,
        )
        raw_trace_parts = [f"[STREAMING]\n{raw_text}"]
        best_cleaned = cleaned_text
        if _answer_output_is_usable(best_cleaned):
            return best_cleaned, "\n\n".join(raw_trace_parts)
        if not auto_finalization_retry:
            return best_cleaned or "", "\n\n".join(raw_trace_parts)

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
        mask=None,
        audio=None,
        audio_file_path="",
        ctx=None,
        n_batch=None,
        n_ubatch=None,
        gpu_layers=None,
        image_max_tokens=None,
        top_k=None,
        pool_size=None,
        n_threads=None,
        n_threads_batch=None,
        flash_attn=True,
        offload_kqv=True,
        ctx_checkpoints=0,
        unique_id=None,
        extra_pnginfo=None,
        node_class="QwenVLGGUF",
        stream_to_terminal=False,
        enable_thinking=True,
        auto_finalization_retry=False,
        hf_token="",
    ):
        print(f"[QwenVL GGUF DEBUG] Starting run with seed={seed}")
        image, mask_hash = apply_mask_highlight(image, mask)
        image_hash = get_image_hash(image)
        if mask_hash:
            image_hash = f"{image_hash}:{mask_hash}"
        video_hash = get_video_hash(video)
        audio_hash = _get_audio_hash(audio)
        audio_file_hash = _get_audio_file_hash(audio_file_path)
        input_signature = build_node_input_signature(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image_hash=image_hash,
            video_hash=video_hash,
            audio_hash=audio_hash,
            audio_file_hash=audio_file_hash,
            frame_count=frame_count,
            ctx=ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            device=device,
            flash_attn=bool(flash_attn),
            offload_kqv=bool(offload_kqv),
            ctx_checkpoints=ctx_checkpoints,
            enable_thinking=bool(enable_thinking),
            auto_finalization_retry=bool(auto_finalization_retry),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        requires_audio_input = node_class in {"ThinkingLLM_Gemma4_Audio_GGUF"}
        skip_prompt_persistence = requires_audio_input
        audio_b64, audio_notes = _audio_inputs_to_wav_base64(audio=audio, audio_file_path=audio_file_path)
        if requires_audio_input and not audio_b64:
            return _missing_audio_result(audio_notes)
        if audio_b64:
            print(f"[QwenVL GGUF DEBUG] Audio processed: {len(audio_b64)} audio item(s)")

        if not stream_to_terminal and not skip_prompt_persistence:
            saved = get_node_saved_prompt_with_seed(
                node_class,
                unique_id,
                extra_pnginfo,
                seed=int(seed),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                input_signature=input_signature,
            )
            if saved:
                print(f"[QwenVL GGUF] Preset {preset_prompt}: fixed seed {seed} matched; no LLM call was made.")
                return (saved, "")
        if not stream_to_terminal:
            print(f"[QwenVL GGUF] Generating new prompt")

        prompt_template = "" if preset_prompt == "🚫 No preset (image-only)" else SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)

        # Generate cache key with all inputs including seed
        cache_key = get_cache_key(model_name, preset_prompt, custom_prompt, image_hash, video_hash, int(seed), max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, enable_thinking=enable_thinking)

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
        if mask_hash:
            prompt = f"{prompt}\n\n{MASK_FOCUS_INSTRUCTION}".strip()

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
            cuda_index = _current_cuda_device_index()
            allocated = torch.cuda.memory_allocated(cuda_index) if cuda_index is not None else torch.cuda.memory_allocated()
            total = 0
            try:
                if cuda_index is not None:
                    total = int(torch.cuda.get_device_properties(cuda_index).total_memory)
            except Exception:
                total = 0
            if total:
                print(
                    f"[QwenVL GGUF DEBUG] VRAM before loading on {_cuda_device_label(cuda_index)}: "
                    f"{allocated/1024**3:.2f}GB / {total/1024**3:.2f}GB"
                )
            else:
                print(f"[QwenVL GGUF DEBUG] VRAM before loading on {_cuda_device_label(cuda_index)}: {allocated/1024**3:.2f}GB")

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
                    ) + len(images_b64) * max(1024, int(attempt_image_max_tokens or 1024)) + len(audio_b64) * 2048
                    effective_thinking = resolve_qwen_thinking_mode(
                        enable_thinking,
                        max_tokens,
                        label="QwenVL GGUF",
                        prompt_tokens=estimated_prompt_tokens,
                        context_window=context_window,
                        quiet=stream_to_terminal,
                    )
                    print(f"[QwenVL GGUF DEBUG] Loading model...")
                    self._load_model(
                        model_name=model_name,
                        device=device,
                        ctx=ctx,
                        n_batch=attempt_n_batch,
                        n_ubatch=n_ubatch,
                        gpu_layers=gpu_layers,
                        image_max_tokens=attempt_image_max_tokens,
                        top_k=top_k,
                        pool_size=pool_size,
                        n_threads=n_threads,
                        n_threads_batch=n_threads_batch,
                        flash_attn=flash_attn,
                        offload_kqv=offload_kqv,
                        ctx_checkpoints=ctx_checkpoints,
                        enable_thinking=effective_thinking,
                        hf_token=hf_token,
                        unique_id=unique_id,
                    )
                    hf_token = ""
                    print(f"[QwenVL GGUF DEBUG] Model loaded successfully")
                    if (images_b64 or audio_b64) and self.chat_handler is None:
                        print("[QwenVL] Warning: media provided but this model entry has no multimodal handler; media will be ignored")
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
                        audio_b64=audio_b64 if self.chat_handler is not None else [],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                        model_name=model_name,
                        stream_to_terminal=stream_to_terminal,
                        enable_thinking=effective_thinking,
                        auto_finalization_retry=auto_finalization_retry,
                        top_k=top_k,
                        preset_name=preset_prompt,
                        media_summary={
                            "image_input": 1 if image is not None else 0,
                            "video_input": 1 if video is not None else 0,
                            "visual_items": len(images_b64),
                            "audio": len(audio_b64),
                        },
                    )
                    if self.last_backend_trace:
                        raw_trace = f"{self.last_backend_trace}\n\n{raw_trace}" if raw_trace else self.last_backend_trace
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

            if not skip_prompt_persistence:
                # Cache the generated text
                PROMPT_CACHE[cache_key] = {
                    "text": text,
                    "timestamp": None,  # GGUF doesn't have CUDA events
                    "model": model_name,
                    "preset": preset_prompt,
                    "seed": int(seed),
                    "image_hash": image_hash,
                    "mask_hash": mask_hash,
                    "video_hash": video_hash
                }
                save_prompt_cache()  # Save cache to file

                print(f"[QwenVL GGUF] Cached new prompt for seed {seed}: {cache_key[:8]}...")
            else:
                print("[QwenVL GGUF] Audio node result was not stored in the prompt cache")

            print(f"[QwenVL GGUF DEBUG] Returning tuple with text...")

            if not skip_prompt_persistence:
                # Save the generated prompt for future per-node keep-last-prompt
                set_node_saved_prompt(node_class, unique_id, extra_pnginfo, text, raw_trace=raw_trace, seed=int(seed), max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
                print(f"[QwenVL GGUF] Saved per-node prompt: {text[:50]}...")
            else:
                print("[QwenVL GGUF] Audio node result was not stored as a saved prompt")

            return (text, raw_trace)
        finally:
            if not keep_model_loaded:
                self.clear()


class ThinkingLLM_QwenVL_GGUF(QwenVLGGUFBase):
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
                "model_name": (model_keys, {"default": default_model, "tooltip": GGUF_TOOLTIPS["model_name"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": "Select 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional user input that gets combined with the preset template. Leave empty to use only the template."}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 32768, "tooltip": GGUF_TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": GGUF_TOOLTIPS["seed"]}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["stream_tokens_to_terminal"]}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen GGUF models this is advisory and may not be honored by the backend."}),
                "auto_finalization_retry": ("BOOLEAN", {"default": False, "tooltip": "If enabled, runs an extra LLM completion when the first output is empty or reasoning-only. Disabled by default so one node execution performs one generation pass."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["hf_token"]}),
                            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "mask": ("MASK",),
                "audio": ("AUDIO",),
                "audio_file_path": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["audio_file_path"]}),
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
        image=None,
        video=None,
        mask=None,
        audio=None,
        audio_file_path="",
        unique_id=None,
        extra_pnginfo=None,
        stream_tokens_to_terminal=False,
        enable_thinking=True,
        auto_finalization_retry=False,
        hf_token="",
    ):
        result = self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            mask=mask,
            frame_count=16,
            audio=audio,
            audio_file_path=audio_file_path,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.0,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            n_ubatch=None,
            gpu_layers=None,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
            n_threads=None,
            n_threads_batch=None,
            flash_attn=True,
            offload_kqv=True,
            ctx_checkpoints=0,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="ThinkingLLM_QwenVL_GGUF",
            stream_to_terminal=stream_tokens_to_terminal,
            enable_thinking=enable_thinking,
            auto_finalization_retry=auto_finalization_retry,
            hf_token=hf_token,
        )
        hf_token = ""
        return result


class ThinkingLLM_Gemma4_Audio_GGUF(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        model_keys = _gguf_audio_model_keys() or ["(no Gemma 4 audio GGUF models found)"]
        default_model = next((key for key in model_keys if "gemma-4-12b" in key.lower()), model_keys[0])

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": GGUF_TOOLTIPS["audio_model_name"]}),
                "custom_prompt": ("STRING", {"default": GEMMA4_AUDIO_DEFAULT_PROMPT, "multiline": True, "tooltip": "Audio instruction sent with the AUDIO input. Gemma 4 audio works best with short 16 kHz mono WAV-style input; Comfy AUDIO is converted to WAV before inference."}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 32768, "tooltip": GGUF_TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": GGUF_TOOLTIPS["seed"]}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["stream_tokens_to_terminal"]}),
                "enable_thinking": ("BOOLEAN", {"default": False, "tooltip": "Gemma 4 can reason, but audio transcription and short analysis are usually clearer with thinking disabled."}),
                "auto_finalization_retry": ("BOOLEAN", {"default": False, "tooltip": "If enabled, runs an extra LLM completion when the first output is empty or reasoning-only. Disabled by default so one node execution performs one generation pass."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["hf_token"]}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_file_path": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["audio_file_path"]}),
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
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        stream_tokens_to_terminal=False,
        enable_thinking=False,
        auto_finalization_retry=False,
        hf_token="",
        audio=None,
        audio_file_path="",
        unique_id=None,
        extra_pnginfo=None,
    ):
        result = self.run(
            model_name=model_name,
            preset_prompt="🚫 No preset (image-only)",
            custom_prompt=custom_prompt,
            image=None,
            video=None,
            frame_count=1,
            audio=audio,
            audio_file_path=audio_file_path,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            n_ubatch=None,
            gpu_layers=None,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
            n_threads=None,
            n_threads_batch=None,
            flash_attn=True,
            offload_kqv=True,
            ctx_checkpoints=0,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="ThinkingLLM_Gemma4_Audio_GGUF",
            stream_to_terminal=stream_tokens_to_terminal,
            enable_thinking=enable_thinking,
            auto_finalization_retry=auto_finalization_retry,
            hf_token=hf_token,
        )
        hf_token = ""
        return result


class ThinkingLLM_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
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
                "model_name": (model_keys, {"default": default_model, "tooltip": GGUF_TOOLTIPS["model_name"]}),
                "device": (device_options, {"default": "auto", "tooltip": GGUF_TOOLTIPS["device"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": "Select 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional user input that gets combined with the preset template. Leave empty to use only the template."}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 32768, "tooltip": GGUF_TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "tooltip": "Sampling randomness. Lower values are more deterministic; higher values are more varied."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": "Nucleus sampling cutoff. Lower values restrict token choice; 0.9 is a balanced default."}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "tooltip": "Values above 1.0 reduce repeated phrases; 1.0 leaves repetition unmodified."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": GGUF_TOOLTIPS["frame_count"]}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512, "tooltip": GGUF_TOOLTIPS["ctx"]}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64, "tooltip": GGUF_TOOLTIPS["n_batch"]}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "tooltip": GGUF_TOOLTIPS["gpu_layers"]}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256, "tooltip": GGUF_TOOLTIPS["image_max_tokens"]}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 32768, "tooltip": GGUF_TOOLTIPS["top_k"]}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288, "tooltip": GGUF_TOOLTIPS["pool_size"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": GGUF_TOOLTIPS["seed"]}),
                "legacy_seed_mode": ([False, "fixed", "randomize", "increment", "decrement"], {"default": "fixed", "tooltip": "Legacy workflow compatibility only. This widget is ignored by current ThinkingLLM logic."}),
                "legacy_unload_after_run": ("BOOLEAN", {"default": False, "tooltip": "Legacy workflow compatibility only. Model unloading is controlled by keep_model_loaded."}),
                "n_ubatch": ("INT", {"default": 512, "min": 0, "max": 32768, "step": 32, "tooltip": GGUF_TOOLTIPS["n_ubatch"]}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": GGUF_TOOLTIPS["n_threads"]}),
                "n_threads_batch": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": GGUF_TOOLTIPS["n_threads_batch"]}),
                "flash_attn": ("BOOLEAN", {"default": True, "tooltip": GGUF_TOOLTIPS["flash_attn"]}),
                "offload_kqv": ("BOOLEAN", {"default": True, "tooltip": GGUF_TOOLTIPS["offload_kqv"]}),
                "ctx_checkpoints": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": GGUF_TOOLTIPS["ctx_checkpoints"]}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": GGUF_TOOLTIPS["stream_tokens_to_terminal"]}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen GGUF models this is advisory."}),
                "auto_finalization_retry": ("BOOLEAN", {"default": False, "tooltip": "If enabled, runs an extra LLM completion when the first output is empty or reasoning-only. Disabled by default so one node execution performs one generation pass."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["hf_token"]}),
                            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "mask": ("MASK",),
                "audio": ("AUDIO",),
                "audio_file_path": ("STRING", {"default": "", "multiline": False, "tooltip": GGUF_TOOLTIPS["audio_file_path"]}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE", "MASK_PREVIEW")
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
        legacy_seed_mode,
        legacy_unload_after_run,
        n_ubatch,
        n_threads,
        n_threads_batch,
        flash_attn,
        offload_kqv,
        ctx_checkpoints,
        image=None,
        video=None,
        mask=None,
        audio=None,
        audio_file_path="",
        unique_id=None,
        extra_pnginfo=None,
        stream_tokens_to_terminal=False,
        enable_thinking=True,
        auto_finalization_retry=False,
        hf_token="",
    ):
        _ = legacy_seed_mode
        _ = legacy_unload_after_run
        result = self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            mask=mask,
            frame_count=frame_count,
            audio=audio,
            audio_file_path=audio_file_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
            n_threads=n_threads if int(n_threads) > 0 else None,
            n_threads_batch=n_threads_batch if int(n_threads_batch) > 0 else None,
            flash_attn=flash_attn,
            offload_kqv=offload_kqv,
            ctx_checkpoints=ctx_checkpoints,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="ThinkingLLM_QwenVL_GGUF_Advanced",
            stream_to_terminal=stream_tokens_to_terminal,
            enable_thinking=enable_thinking,
            auto_finalization_retry=auto_finalization_retry,
            hf_token=hf_token,
        )
        hf_token = ""
        mask_preview, _ = apply_mask_highlight(image, mask)
        return (*result, mask_preview)


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_QwenVL_GGUF": ThinkingLLM_QwenVL_GGUF,
    "ThinkingLLM_QwenVL_GGUF_Advanced": ThinkingLLM_QwenVL_GGUF_Advanced,
    "ThinkingLLM_Gemma4_Audio_GGUF": ThinkingLLM_Gemma4_Audio_GGUF,
    "AILab_QwenVL_GGUF": ThinkingLLM_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": ThinkingLLM_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_QwenVL_GGUF": "ThinkingLLM (GGUF)",
    "ThinkingLLM_QwenVL_GGUF_Advanced": "ThinkingLLM Advanced (GGUF)",
    "ThinkingLLM_Gemma4_Audio_GGUF": "ThinkingLLM Gemma 4 Audio (GGUF)",
    "AILab_QwenVL_GGUF": "ThinkingLLM (GGUF)",
    "AILab_QwenVL_GGUF_Advanced": "ThinkingLLM Advanced (GGUF)",
}
