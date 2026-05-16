# ComfyUI-QwenVL GGUF prompt enhancer
#
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
import ctypes
import gc
import hashlib
import importlib
import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from AILab_LlamaCppInstaller import ensure_llama_cpp_backend

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output, prompt_output_guard
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.model_management import throw_exception_if_processing_interrupted

# Import cache functions from main module
import sys
sys.path.append(str(Path(__file__).parent))
from AILab_StreamDisplay import TerminalStreamDisplay
from AILab_QwenVL import (
    PROMPT_CACHE,
    apply_qwen_soft_thinking_directive,
    build_node_input_signature,
    download_hf_file_to_path,
    ensure_cuda_vram_headroom,
    estimate_qwen_text_tokens,
    get_cache_key,
    get_alternative_cache_key,
    save_prompt_cache,
    get_node_saved_prompt,
    get_node_saved_prompt_with_seed,
    resolve_qwen_thinking_mode,
    set_node_saved_prompt,
    load_node_prompt_state,
    _make_node_state_key,
    _build_workflow_fingerprint,
)
from AILab_QwenVL_GGUF import read_gguf_architecture, register_active_gguf_loader, release_other_gguf_loaders


def _parse_repo_quant_sizes(repo_key: str) -> dict[str, str] | None:
    """Parse bracket info from repo key like 'Gemma-4-E4B-it-GGUF [Q4:5.4GB|Q8:8GB|VRAM:~7GB]'.
    Returns a dict mapping quant names to size strings, or None if no bracket info."""
    import re
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
    from pathlib import Path
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

_EMPTY_THINK_RE = re.compile(r"<think[^>]*>\s*</think>", flags=re.IGNORECASE | re.DOTALL)
_STREAM_HEARTBEAT_INTERVAL_SECONDS = 3.0
_PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS = 3
_STREAM_POLL_INTERVAL_SECONDS = 0.25


def _looks_like_prompt_planning(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*(okay[,.:]?|first[,.:]?|next[,.:]?|then[,.:]?|wait[,.:]?|final\s+plan|final\s+check)\b",
            text,
        )
        or re.search(r"(?i)\b(i\s+(should|need|must|will|am\s+going\s+to|have\s+to))\b", text)
    )


def _prompt_output_is_usable(cleaned_text: str) -> bool:
    return bool(cleaned_text and not _looks_like_prompt_planning(cleaned_text))


def _maybe_emit_prompt_stream_heartbeat(stage_label: str, started_at: float, last_status_at: float, full_text: str) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    cleaned_preview = clean_model_output(full_text, OutputCleanConfig(mode="prompt"))
    if _prompt_output_is_usable(cleaned_preview):
        return now
    elapsed = max(0.0, now - started_at)
    print(f"[QwenVL GGUF] {stage_label}: reasoning hidden; still generating... ({len(full_text)} chars, {elapsed:.1f}s)")
    return now


def _maybe_emit_prompt_waiting_heartbeat(stage_label: str, started_at: float, last_status_at: float) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    print("[QwenVL GGUF] waiting for first streamed chunk...")
    return now


def _compact_progress_line_width(limit: int = 60) -> int:
    terminal_half_width = max(24, shutil.get_terminal_size(fallback=(120, 20)).columns // 2)
    return max(24, min(int(limit), terminal_half_width))


def _normalize_compact_progress_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").replace("\r", "").replace("\n", " ")).strip()


def _split_compact_progress_lines(text: str, line_width: int) -> tuple[list[str], str]:
    lines: list[str] = []
    remaining = text.lstrip()
    while len(remaining) > line_width:
        split_at = remaining.rfind(" ", 0, line_width + 1)
        if split_at <= 0:
            split_at = line_width
        line = remaining[:split_at].strip()
        if line:
            lines.append(line)
        remaining = remaining[split_at:].lstrip()
    return lines, remaining


def _maybe_emit_prompt_compact_progress(stage_label: str, started_at: float, last_status_at: float, full_text: str, progress_state: dict, *, force: bool = False, final: bool = False) -> float:
    now = time.monotonic()
    if not final and not force and (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    rendered_text = _normalize_compact_progress_text(clean_model_output(full_text, OutputCleanConfig(mode="prompt")) or full_text)
    if not rendered_text:
        print("[QwenVL GGUF] generating...")
        return now

    line_width = int(progress_state.get("line_width") or _compact_progress_line_width())
    progress_state["line_width"] = line_width
    previous_snapshot = progress_state.get("rendered_snapshot", "")
    pending_text = progress_state.get("pending_text", "")
    if previous_snapshot and rendered_text.startswith(previous_snapshot):
        delta = rendered_text[len(previous_snapshot):]
        pending_text = f"{pending_text}{delta}"
    else:
        pending_text = rendered_text
    progress_state["rendered_snapshot"] = rendered_text

    lines, pending_text = _split_compact_progress_lines(pending_text, line_width)
    emitted = False
    for line in lines:
        print(line)
        emitted = True
    if final and pending_text:
        print(pending_text)
        pending_text = ""
        emitted = True
    progress_state["pending_text"] = pending_text

    if emitted or final:
        return now
    return last_status_at


def _install_llama_abort_callback(llm, should_abort):
    try:
        llama_cpp_lib = importlib.import_module("llama_cpp.llama_cpp")
    except Exception:
        return None

    ctx = getattr(llm, "ctx", None)
    if ctx is None:
        ctx_holder = getattr(llm, "_ctx", None)
        ctx = getattr(ctx_holder, "ctx", None)
    callback_type = getattr(llama_cpp_lib, "ggml_abort_callback", None)
    set_abort_callback = getattr(llama_cpp_lib, "llama_set_abort_callback", None)
    if ctx is None or callback_type is None or set_abort_callback is None:
        return None

    def _active_callback(_):
        return bool(should_abort())

    def _inactive_callback(_):
        return False

    active_cb = callback_type(_active_callback)
    inactive_cb = callback_type(_inactive_callback)
    try:
        set_abort_callback(ctx, active_cb, ctypes.c_void_p())
    except Exception:
        return None

    def _clear_callback():
        try:
            set_abort_callback(ctx, inactive_cb, ctypes.c_void_p())
        except Exception:
            pass

    return active_cb, inactive_cb, _clear_callback


def _describe_prompt_enhancer_thinking(requested_thinking: bool, effective_thinking: bool, raw_text: str, *, retried: bool = False) -> str:
    raw = raw_text or ""
    raw_lower = raw.lower()
    if requested_thinking and not effective_thinking:
        raw_state = "budget_disabled"
    elif not requested_thinking:
        raw_state = "disabled"
    elif _EMPTY_THINK_RE.search(raw):
        raw_state = "think_empty"
    elif "<think" in raw_lower:
        raw_state = "think_present"
    else:
        raw_state = "think_hidden_or_absent"
    terminal_state = "off" if not effective_thinking else "hidden"
    pass_state = "retry" if retried else "single_pass"
    return (
        f"requested={bool(requested_thinking)} effective={bool(effective_thinking)} "
        f"raw={raw_state} terminal_reasoning={terminal_state} final_output=clean_prompt pass={pass_state}"
    )


# DEPRECATED: per-node state via set_node_saved_prompt / get_node_saved_prompt is used instead.
LAST_SAVED_PROMPT = None  # kept only to avoid ImportError in legacy importers

# Load per-node state at startup so keep_last_prompt works across restarts
load_node_prompt_state()

NODE_DIR = Path(__file__).parent
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
PROMPT_CONFIG_PATH = NODE_DIR / "AILab_System_Prompts.json"


def load_prompt_config():
    if not PROMPT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"[QwenVL] Missing AILab_System_Prompts.json at {PROMPT_CONFIG_PATH}")
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwen_text = data.get("qwen_text") or {}
        styles = qwen_text.get("styles")
        translation_prompt = qwen_text.get("translation_prompt")
        if not styles or not translation_prompt:
            raise ValueError("AILab_System_Prompts.json must include qwen_text.styles and qwen_text.translation_prompt")
        return {"styles": styles, "translation_prompt": translation_prompt}
    except Exception as exc:
        raise RuntimeError(f"[QwenVL] Failed to load AILab_System_Prompts.json: {exc}") from exc


PROMPT_CONFIG = load_prompt_config()
STYLES = PROMPT_CONFIG.get("styles", {})
CUSTOM_ONLY_STYLE = "✍️ Custom Only (no preset)"


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


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


class ThinkingLLM_QwenVL_GGUF_PromptEnhancer:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ENHANCED_OUTPUT", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def _load_backend(self):
        return ensure_llama_cpp_backend(require_vision_handlers=False)

    def __init__(self):
        self.llm = None
        self.current_signature = None
        self.gguf_models = self.load_gguf_models()
        self.styles = STYLES
        register_active_gguf_loader(self)

    @staticmethod
    def _scan_local_gguf_text_models(base_dir: Path, existing_filenames: set[str]) -> dict[str, dict]:
        """Scan the GGUF base directory for locally available text-only .gguf files."""
        local_models: dict[str, dict] = {}
        if not base_dir.exists() or not base_dir.is_dir():
            return local_models
        try:
            for gguf_file in base_dir.rglob("*.gguf"):
                if not gguf_file.is_file():
                    continue
                # Skip mmproj files — they are vision projectors, not text models
                if "mmproj" in gguf_file.name.lower():
                    continue
                if gguf_file.name in existing_filenames:
                    continue
                display = f"[local] {gguf_file.name}"
                local_models[display] = {
                    "filename": str(gguf_file),
                    "is_local": True,
                    "repo_id": None,
                    "alt_repo_ids": [],
                    "author": None,
                    "repo_dirname": gguf_file.parent.name,
                    "context_length": 32768,
                }
        except PermissionError:
            pass
        if local_models:
            print(f"[QwenVL] Discovered {len(local_models)} local GGUF text model(s) on disk")
        return local_models

    @staticmethod
    def load_gguf_models():
        fallback = {
            "base_dir": "LLM/GGUF",
            "models": {},
        }
        data = {}
        if GGUF_CONFIG_PATH.exists():
            try:
                with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
            except Exception as exc:
                print(f"[QwenVL] gguf_models.json load failed: {exc}")

        base_dir = data.get("base_dir") or fallback["base_dir"]

        models: dict[str, dict] = {}

        # Legacy/custom direct entries (optional)
        legacy_models = data.get("models") or {}
        if isinstance(legacy_models, dict):
            for name, entry in legacy_models.items():
                if isinstance(entry, dict):
                    models[name] = entry

        # Text-only catalog (use Qwen_model; do not use qwenVL_model here)
        qwen_repos = data.get("Qwen_model") or {}
        if isinstance(qwen_repos, dict):
            seen_display_names: set[str] = set()
            for repo_key, repo in qwen_repos.items():
                if not isinstance(repo, dict):
                    continue
                author = repo.get("author") or repo.get("publisher")
                repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
                _defaults_raw = repo.get("defaults")
                defaults: dict = _defaults_raw if isinstance(_defaults_raw, dict) else {}
                repo_id = repo.get("repo_id")
                alt_repo_ids = repo.get("alt_repo_ids") or []
                model_files = repo.get("model_files") or []
                quant_sizes = _parse_repo_quant_sizes(repo_key)
                for model_file in model_files:
                    # Prefer short names in UI: just the filename.
                    display = Path(model_file).name
                    if quant_sizes:
                        q = _quant_from_filename(model_file)
                        if q and q in quant_sizes:
                            display = f"{display} [~{quant_sizes[q]}]"
                    if display in seen_display_names:
                        display = f"{display} ({repo_key})"
                    seen_display_names.add(display)
                    entry = dict(defaults)
                    entry.update(
                        {
                            "author": author,
                            "repo_dirname": repo_name,
                            "repo_id": repo_id,
                            "alt_repo_ids": alt_repo_ids,
                            "filename": model_file,
                        }
                    )
                    models[display] = entry

        # Scan filesystem for locally available models not in JSON config
        # Collect all directories to scan: the configured base_dir + any extra LLM paths from ComfyUI
        existing_filenames = {Path(e.get("filename", "")).name for e in models.values() if e.get("filename")}
        scan_dirs: list[Path] = [_resolve_base_dir(base_dir)]
        try:
            if "LLM" in folder_paths.folder_names_and_paths:
                for llm_path in folder_paths.get_folder_paths("LLM"):
                    gguf_dir = Path(llm_path) / "GGUF"
                    if gguf_dir not in scan_dirs:
                        scan_dirs.append(gguf_dir)
                    llm_p = Path(llm_path)
                    if llm_p not in scan_dirs:
                        scan_dirs.append(llm_p)
        except Exception:
            pass
        for scan_dir in scan_dirs:
            local_models = ThinkingLLM_QwenVL_GGUF_PromptEnhancer._scan_local_gguf_text_models(scan_dir, existing_filenames)
            models.update(local_models)
            existing_filenames.update(Path(e.get("filename", "")).name for e in local_models.values() if e.get("filename"))

        return {"base_dir": base_dir, "models": models}

    @classmethod
    def INPUT_TYPES(cls):
        styles = [CUSTOM_ONLY_STYLE] + list(STYLES.keys())
        preferred_style = "📝 Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "📝 Enhance")
        temp = cls.load_gguf_models()
        model_keys = sorted(list((temp.get("models") or {}).keys())) or ["(no GGUF models found)"]
        default_model = model_keys[0]
        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": "GGUF model from config or auto-detected from models/LLM/GGUF directory. [local] prefix = found on disk."}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to enhance. Leave blank to just emit the preset instruction."}),
                "preset_system_prompt": (styles, {"default": default_style}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 32, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "english_output": ("BOOLEAN", {"default": False, "tooltip": "Force final output in English using translation prompt."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Select device; auto prefers GPU when available."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory for faster repeated inference (uses more VRAM)."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Show readable generation progress and thinking-status summaries in the ComfyUI terminal. When enabled, fixed-seed and prompt-cache reuse are bypassed so a fresh streamed run can occur."}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. Prompt enhancers still return a cleaned final prompt, so terminal reasoning may be hidden or empty."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def clear(self):
        print(f"[QwenVL PromptEnhancer DEBUG] Starting VRAM cleanup...")
        
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
                print(f"[QwenVL PromptEnhancer DEBUG] Error closing LLM: {e}")
            finally:
                self.llm = None
        
        # Clear signature
        self.current_signature = None
        
        # Aggressive garbage collection
        gc.collect()
        
        # Force CUDA cache cleanup multiple times
        if torch.cuda.is_available():
            print(f"[QwenVL PromptEnhancer DEBUG] Clearing CUDA cache...")
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
            # Additional cleanup
            torch.cuda.empty_cache()
        
        print(f"[QwenVL PromptEnhancer DEBUG] VRAM cleanup completed")

    def _resolve_model_path(self, model_name):
        models = self.gguf_models.get("models") or {}
        entry = models.get(model_name) or {}

        # Back-compat: allow workflows to pass a filename instead of a catalog key.
        if not entry:
            wanted = _model_name_to_filename_candidates(model_name)
            for candidate in models.values():
                filename = candidate.get("filename")
                if filename and Path(filename).name in wanted:
                    entry = candidate
                    break

        # Local models store absolute paths — use directly
        filename = entry.get("filename")
        if filename and Path(filename).is_absolute():
            return Path(filename)

        base_dir = _resolve_base_dir(self.gguf_models.get("base_dir") or "LLM/GGUF")

        path = entry.get("path")
        if path:
            return Path(path).expanduser()

        if filename:
            author = _safe_dirname(str(entry.get("author") or entry.get("publisher") or ""))
            repo_dir = _safe_dirname(str(entry.get("repo_dirname") or model_name))
            if author and author != "unknown":
                target = base_dir / author / repo_dir / Path(filename).name
            else:
                target = base_dir / repo_dir / Path(filename).name
            if target.exists():
                return target
            existing = _find_existing_local_file(base_dir, filename)
            if existing is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.hardlink_to(existing)
                    return target
                except Exception:
                    return existing
            return target

        return base_dir / model_name

    def _maybe_download_model(self, model_name, resolved, unique_id=None):
        if resolved.exists():
            return
        # Local models should already exist on disk — don't attempt download
        models = self.gguf_models.get("models") or {}
        entry = models.get(model_name) or {}
        if entry.get("is_local"):
            raise FileNotFoundError(f"[QwenVL] Local GGUF model not found: {resolved}")
        if not entry:
            wanted = _model_name_to_filename_candidates(model_name)
            for candidate in models.values():
                filename = candidate.get("filename")
                if filename and Path(filename).name in wanted:
                    entry = candidate
                    break

        repo_ids = [rid for rid in (entry.get("alt_repo_ids") or []) + [entry.get("repo_id")] if rid]
        filename = entry.get("filename") or resolved.name
        if not repo_ids or not filename:
            raise FileNotFoundError(f"[QwenVL] GGUF missing and no repo_id/filename to download: {resolved}")
        target_dir = resolved.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        attempted = []
        for repo_id in repo_ids:
            attempted.append(repo_id)
            print(f"[QwenVL] Downloading GGUF {filename} from {repo_id}")
            try:
                download_hf_file_to_path(
                    [repo_id],
                    filename,
                    resolved,
                    node_id=unique_id,
                    progress_label=f"QwenVL PromptEnhancer GGUF Download: {Path(filename).name}",
                )
            except Exception as exc:
                print(f"[QwenVL] Download failed from {repo_id}: {exc}")
            if resolved.exists():
                break
        if not resolved.exists():
            raise FileNotFoundError(f"[QwenVL] GGUF model not found after download: {resolved} (tried: {', '.join(attempted)})")

    def _load_model(self, model_name, device, enable_thinking=True, unique_id=None):
        Llama = self._load_backend()
        resolved = self._resolve_model_path(model_name)
        self._maybe_download_model(model_name, resolved, unique_id=unique_id)
        model_cfg = self.gguf_models["models"].get(model_name, {})
        context_length = model_cfg.get("context_length", 32768)
        signature = (resolved, context_length, device, bool(enable_thinking))
        if self.llm is not None and self.current_signature == signature:
            ensure_cuda_vram_headroom("QwenVL PromptEnhancer GGUF", min_free_gb=1.0, min_free_ratio=0.08)
            return

        release_other_gguf_loaders(self, resolved.name)
        
        # Force aggressive cleanup before loading new model (especially for same model conflicts)
        print(f"[QwenVL PromptEnhancer DEBUG] Forcing cleanup before model loading...")
        self.clear()
        
        # Additional wait for CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(0.1)  # Brief pause for cleanup to complete
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not resolved.exists():
            raise FileNotFoundError(f"[QwenVL] GGUF model not found: {resolved}")
        print(f"[QwenVL] Loading GGUF model from {resolved}")
        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device
        auto_gpu_layers = -1 if device_choice == "cuda" else 0
        threads = None
        if device_choice == "cpu":
            threads = max(os.cpu_count() or 1, 1)
        kwargs = {
            "model_path": str(resolved),
            "n_ctx": context_length,
            "n_gpu_layers": auto_gpu_layers,
            "n_threads": None if threads == 0 else threads,
            "n_batch": 1024,
            "verbose": False,
            "chat_format": "qwen",
        }
        self.current_context_length = context_length

        # Detect architecture from GGUF metadata instead of relying on model name
        arch = read_gguf_architecture(resolved)
        self.gguf_arch = arch
        is_qwen = (arch and any(a in arch for a in ("qwen35", "qwen35moe", "qwen3", "qwen"))) or "qwen3.5-" in model_name.lower() or "qwen3-" in model_name.lower()
        self.supports_qwen_soft_think = arch == "qwen3" if arch else "qwen3-" in model_name.lower()
        if is_qwen:
            thinking_state = bool(enable_thinking)
            kwargs["chat_template_kwargs"] = {"enable_thinking": thinking_state}
            state_label = "enabled" if thinking_state else "disabled"
            print(f"[QwenVL] Qwen architecture detected (arch={arch}): Thinking {state_label} via chat template.")
        else:
            thinking_state = bool(enable_thinking)
            if not thinking_state:
                # Prefill steering: seed the assistant response with a direct-answer phrase
                # to push models that ignore enable_thinking toward direct output.
                kwargs["chat_template_kwargs"] = kwargs.get("chat_template_kwargs", {})
                kwargs["chat_template_kwargs"]["prefill"] = (
                    "I'll answer directly without any analysis or thinking:\n\n"
                )
                print(f"[QwenVL] Non-Qwen architecture (arch={arch}): Thinking OFF — prefill steering applied.")
            else:
                print(f"[QwenVL] Non-Qwen architecture (arch={arch}): Thinking ON (advisory — backend may ignore).")

        self.llm = Llama(**kwargs)
        self.current_signature = signature

    def _invoke_llama(
        self,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
        stream_to_terminal=False,
        initial_stage_label="INITIAL GENERATION",
        enable_thinking=True,
    ):
        """Returns (cleaned_prompt, raw_trace) tuple."""
        stream_display = TerminalStreamDisplay("QwenVL GGUF", suppress_planning=True, compact=False) if stream_to_terminal else None
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        if supports_soft_think:
            directive = "/think" if enable_thinking else "/no_think"
            print(f"[QwenVL] Qwen3 GGUF prompt enhancer: Thinking {'enabled' if enable_thinking else 'disabled'} via chat template and {directive}.")

        def _call(
            system: str,
            user: str,
            temp: float,
            seed_val: int,
            stage_label: str,
            *,
            attempt_enable_thinking: bool | None = None,
        ) -> str:
            current_enable_thinking = enable_thinking if attempt_enable_thinking is None else bool(attempt_enable_thinking)
            effective_user_prompt = apply_qwen_soft_thinking_directive(
                user,
                current_enable_thinking,
                supports_soft_switch=supports_soft_think,
            )
            ensure_cuda_vram_headroom("QwenVL PromptEnhancer GGUF", min_free_gb=1.0, min_free_ratio=0.08)
            if self.llm is not None and hasattr(self.llm, "reset"):
                try:
                    self.llm.reset()
                except Exception as exc:
                    print(f"[QwenVL PromptEnhancer DEBUG] llama context reset skipped: {exc}")
            llm = self.llm
            if llm is None:
                raise RuntimeError("[QwenVL] GGUF model is not loaded")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": effective_user_prompt},
            ]
            full_text = ""
            stage_started_at = time.monotonic()
            last_status_at = stage_started_at
            compact_progress_state = {"rendered_snapshot": "", "pending_text": ""}
            if stream_to_terminal:
                if stream_display is None:
                    raise RuntimeError("[QwenVL] Stream display was not initialized")
                print(f"[QwenVL PromptEnhancer GGUF] STREAMING {stage_label}")
                stream_display.start_stage(stage_label)
            abort_requested = threading.Event()
            abort_callback_refs = _install_llama_abort_callback(llm, abort_requested.is_set)
            result_queue: queue.Queue[tuple[str, dict | BaseException | None]] = queue.Queue()

            def _stream_worker():
                try:
                    response = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temp,
                        top_p=top_p,
                        repeat_penalty=repetition_penalty,
                        seed=seed_val,
                        stop=["<|im_end|>", "<|im_start|>"],
                        stream=True,
                    )
                    for chunk in response:
                        result_queue.put(("chunk", chunk))
                    result_queue.put(("done", None))
                except BaseException as exc:
                    result_queue.put(("error", exc))

            worker = threading.Thread(target=_stream_worker, name=f"QwenGGUFPrompt-{stage_label}", daemon=True)
            worker.start()
            try:
                while True:
                    try:
                        kind, payload = result_queue.get(timeout=_STREAM_POLL_INTERVAL_SECONDS)
                    except queue.Empty:
                        try:
                            throw_exception_if_processing_interrupted()
                        except Exception:
                            abort_requested.set()
                            raise
                        if full_text:
                            if stream_display is not None:
                                last_status_at = _maybe_emit_prompt_stream_heartbeat(stage_label, stage_started_at, last_status_at, full_text)
                            else:
                                last_status_at = _maybe_emit_prompt_compact_progress(stage_label, stage_started_at, last_status_at, full_text, compact_progress_state)
                        else:
                            last_status_at = _maybe_emit_prompt_waiting_heartbeat(stage_label, stage_started_at, last_status_at)
                        continue

                    if kind == "chunk":
                        throw_exception_if_processing_interrupted()
                        chunk = payload if isinstance(payload, dict) else {}
                        token = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if token:
                            had_text_before = bool(full_text)
                            full_text += token
                            if stream_display is not None:
                                stream_display.push(token)
                                last_status_at = _maybe_emit_prompt_stream_heartbeat(stage_label, stage_started_at, last_status_at, full_text)
                            else:
                                last_status_at = _maybe_emit_prompt_compact_progress(
                                    stage_label,
                                    stage_started_at,
                                    last_status_at,
                                    full_text,
                                    compact_progress_state,
                                    force=not had_text_before,
                                )
                        continue

                    if kind == "done":
                        break

                    if kind == "error":
                        if isinstance(payload, BaseException):
                            raise payload
                        raise RuntimeError(f"[QwenVL] llama_cpp streaming failed: {payload!r}")
            finally:
                abort_requested.set()
                worker.join(timeout=1.0)
                if abort_callback_refs is not None:
                    _, _, clear_abort_callback = abort_callback_refs
                    clear_abort_callback()
                if stream_display is not None:
                    stream_display.end_stage()
                elif full_text:
                    _maybe_emit_prompt_compact_progress(
                        stage_label,
                        stage_started_at,
                        last_status_at,
                        full_text,
                        compact_progress_state,
                        final=True,
                    )
            if not full_text.strip():
                raise RuntimeError("[QwenVL] llama_cpp streaming returned empty response")
            return full_text.strip()

        raw = _call(system_prompt, user_prompt, float(temperature), int(seed), initial_stage_label)
        cleaned = clean_model_output(raw, OutputCleanConfig(mode="prompt"))
        raw_trace_parts = [f"[{initial_stage_label}]\n{raw}"]
        best_cleaned = cleaned.strip()
        if _prompt_output_is_usable(best_cleaned):
            return best_cleaned, "\n\n".join(raw_trace_parts)

        current_raw = raw
        for attempt_number in range(2, _PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS + 1):
            if stream_to_terminal:
                print(
                    "[QwenVL PromptEnhancer GGUF] "
                    f"Finalization attempt {attempt_number}/{_PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS} "
                    "— prior output was empty or reasoning-only"
                )
            retry_system = (
                "You are a professional photography prompt writer.\n"
                "Output ONLY ONE final photography prompt paragraph.\n"
                "No analysis, no planning steps, no first-person, and no <think>.\n"
                "No bullet points, no headings, no JSON, no markdown, no quotes."
            )
            retry_user = (
                "Rewrite the following into the final prompt paragraph:\n\n"
                f"{current_raw}\n"
            )
            force_non_thinking = attempt_number == _PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS
            stage_label = f"FINALIZATION ATTEMPT {attempt_number}/{_PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS}"
            raw_retry = _call(
                retry_system,
                retry_user,
                0.4,
                int(seed) + 999 + attempt_number,
                stage_label,
                attempt_enable_thinking=False if force_non_thinking else enable_thinking,
            )
            raw_trace_parts.append(f"[{stage_label}]\n{raw_retry}")
            cleaned_retry = clean_model_output(raw_retry, OutputCleanConfig(mode="prompt")).strip()
            if cleaned_retry and len(cleaned_retry) >= len(best_cleaned):
                best_cleaned = cleaned_retry
            if _prompt_output_is_usable(cleaned_retry):
                return cleaned_retry, "\n\n".join(raw_trace_parts)
            current_raw = raw_retry

        if stream_to_terminal:
            print(
                "[QwenVL PromptEnhancer GGUF] "
                f"Finalization limit reached ({_PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS}/{_PROMPT_ENHANCER_MAX_FINALIZATION_ATTEMPTS}) "
                "— returning the best cleaned prompt available"
            )
        return best_cleaned or "", "\n\n".join(raw_trace_parts)

    def process(
        self,
        model_name,
        prompt_text,
        preset_system_prompt,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        english_output,
        device,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        stream_tokens_to_terminal=False,
        unique_id=None,
        extra_pnginfo=None,
        enable_thinking=True,
    ):
        node_class = "ThinkingLLM_QwenVL_GGUF_PromptEnhancer"
        input_signature = build_node_input_signature(
            model_name=model_name,
            prompt_text=prompt_text,
            preset_system_prompt=preset_system_prompt,
            custom_system_prompt=custom_system_prompt,
            english_output=bool(english_output),
            device=device,
            enable_thinking=bool(enable_thinking),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Auto-retrieve saved prompt when seed is fixed (no keep_last_prompt needed)
        saved_prompt = get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        if saved_prompt and not stream_tokens_to_terminal:
            print(f"[QwenVL PromptEnhancer GGUF] Fixed seed {seed} matched — using per-node prompt: {saved_prompt[:50]}...")
            return (saved_prompt, "")
        if saved_prompt and stream_tokens_to_terminal:
            print("[QwenVL PromptEnhancer GGUF] Streaming requested — bypassing fixed-seed prompt reuse for a fresh streamed run")
        if keep_last_prompt:
            print(f"[QwenVL PromptEnhancer GGUF] Keep last prompt enabled but no saved prompt found — returning empty")
            return ("", "")

        # Always generate unless keep_last_prompt was requested
        print(f"[QwenVL PromptEnhancer GGUF] Generating new prompt")

        # Generate cache key with all inputs including seed
        cache_prompt = "\n\n".join(
            part for part in (
                prompt_text.strip(),
                custom_system_prompt.strip(),
                f"english_output={bool(english_output)}",
            ) if part
        )
        cache_key = get_cache_key(model_name, preset_system_prompt, cache_prompt, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)

        # Check cache first (only for random mode)
        if cache_key in PROMPT_CACHE and not stream_tokens_to_terminal:
            cached_text = PROMPT_CACHE[cache_key].get("text", "")
            if cached_text:
                print(f"[QwenVL PromptEnhancer GGUF] Using cached prompt for seed {seed}: {cache_key[:8]}...")
                return (cached_text.strip(), "")
        if stream_tokens_to_terminal and cache_key in PROMPT_CACHE:
            cached_text = PROMPT_CACHE[cache_key].get("text", "")
            if cached_text:
                print("[QwenVL PromptEnhancer GGUF] Streaming requested — bypassing prompt cache for a fresh streamed run")

        is_custom_only = preset_system_prompt == CUSTOM_ONLY_STYLE
        style_entry = {} if is_custom_only else self.styles.get(preset_system_prompt, {})
        style_system_prompt = (style_entry.get("system_prompt") or "").strip()
        custom_system_prompt = custom_system_prompt.strip()
        system_prompt = "\n\n".join(part for part in (custom_system_prompt, style_system_prompt) if part).strip()
        if not system_prompt:
            if is_custom_only:
                raise ValueError("custom_system_prompt is required when using Custom Only (no preset).")
            raise ValueError("system_prompt is empty; check AILab_System_Prompts.json or preset selection.")
        system_prompt = f"{system_prompt}\n\n{prompt_output_guard()}"
        merged_prompt = prompt_text.strip() or "Describe a scene vividly."
        model_cfg = self.gguf_models["models"].get(model_name, {})
        context_window = model_cfg.get("context_length", 32768)
        estimated_prompt_tokens = estimate_qwen_text_tokens(system_prompt, merged_prompt)
        effective_thinking = resolve_qwen_thinking_mode(
            enable_thinking,
            max_tokens,
            label="QwenVL PromptEnhancer GGUF",
            prompt_tokens=estimated_prompt_tokens,
            context_window=context_window,
        )
        if stream_tokens_to_terminal:
            print("[QwenVL GGUF] Prompt enhancer terminal stream shows readable progress only; reasoning text may be hidden or stripped from the final prompt.")
        else:
            print("[QwenVL GGUF] Prompt enhancer compact terminal progress is active; enable stream_tokens_to_terminal for full readable chunk output.")
        self._load_model(model_name, device, enable_thinking=effective_thinking, unique_id=unique_id)
        enhanced, raw_trace = self._invoke_llama(
            system_prompt=system_prompt,
            user_prompt=merged_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stream_to_terminal=stream_tokens_to_terminal,
            initial_stage_label="INITIAL GENERATION",
            enable_thinking=effective_thinking,
        )
        if stream_tokens_to_terminal:
            print(
                f"[QwenVL GGUF] Thinking status: "
                f"{_describe_prompt_enhancer_thinking(bool(enable_thinking), effective_thinking, raw_trace, retried='[FINALIZATION ATTEMPT ' in raw_trace)}"
            )
        full_raw_trace = raw_trace
        if english_output:
            translated, trans_trace = self._invoke_llama(
                system_prompt=(
                    PROMPT_CONFIG.get("translation_prompt")
                    or "Return a single English paragraph (150-300 words). No prefixes, bullets, JSON, or </think>. "
                    "Cover subject, environment, lighting, camera settings, composition, color/texture, and style. Output only the prompt."
                ),
                user_prompt=enhanced,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.05,
                seed=seed + 1,
                stream_to_terminal=stream_tokens_to_terminal,
                initial_stage_label="TRANSLATION STAGE",
                enable_thinking=effective_thinking,
            )
            full_raw_trace = f"{raw_trace}\n\n{trans_trace}"
            final = clean_model_output(translated, OutputCleanConfig(mode="prompt")) or translated.strip()
        else:
            final = clean_model_output(enhanced, OutputCleanConfig(mode="prompt")) or enhanced.strip()

        # Cache the generated text
        PROMPT_CACHE[cache_key] = {
            "text": final,
            "timestamp": None,
            "model": model_name,
            "preset": preset_system_prompt,
            "seed": seed,
            "image_hash": None,
            "video_hash": None
        }
        save_prompt_cache()

        print(f"[QwenVL PromptEnhancer GGUF] Cached new prompt for seed {seed}: {cache_key[:8]}...")

        try:
            # Persist per-node for future keep_last_prompt=True
            set_node_saved_prompt(node_class, unique_id, extra_pnginfo, final, raw_trace=full_raw_trace, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
            print(f"[QwenVL PromptEnhancer GGUF] Saved per-node prompt: {final[:50]}...")

            return (final, full_raw_trace)
        finally:
            if not keep_model_loaded:
                self.clear()
                print(f"[QwenVL PromptEnhancer GGUF] keep_model_loaded=False — cleaning up model...")

    @staticmethod
    def _is_english(text):
        letters = len(re.findall(r"[A-Za-z]", text))
        tokens = len(re.findall(r"\S", text))
        return tokens > 0 and letters / tokens > 0.7

    @staticmethod
    def _strip_think(text):
        return clean_model_output(text, OutputCleanConfig(mode="prompt"))


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_QwenVL_GGUF_PromptEnhancer": ThinkingLLM_QwenVL_GGUF_PromptEnhancer,
    "AILab_QwenVL_GGUF_PromptEnhancer": ThinkingLLM_QwenVL_GGUF_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_QwenVL_GGUF_PromptEnhancer": "ThinkingLLM Prompt Enhancer (GGUF)",
    "AILab_QwenVL_GGUF_PromptEnhancer": "ThinkingLLM Prompt Enhancer (GGUF)",
}
