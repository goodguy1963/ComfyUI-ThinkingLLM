# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import hashlib
import importlib
import json
import platform
import re
import time
from enum import Enum
from pathlib import Path

import torch
import subprocess
import sys
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output, prompt_output_guard
from AILab_StreamDisplay import TerminalStreamDisplay
from comfy.model_management import throw_exception_if_processing_interrupted

from AILab_QwenVL import (
    ATTENTION_MODES,
    HF_ALL_MODELS,
    HF_TEXT_MODELS,
    HF_VL_MODELS,
    NODE_PROMPT_STATE,
    PROMPT_CACHE,
    _make_node_state_key,
    apply_qwen_soft_thinking_directive,
    build_node_input_signature,
    ensure_model,
    ensure_cuda_vram_headroom,
    get_cache_key,
    get_alternative_cache_key,
    get_node_saved_prompt,
    get_node_saved_prompt_with_seed,
    resolve_qwen_thinking_mode,
    resolve_qwen_context_window,
    save_prompt_cache,
    QwenVLBase,
    Quantization,
    set_node_saved_prompt,
    TOOLTIPS,
)

# DEPRECATED: per-node state via set_node_saved_prompt / get_node_saved_prompt is used instead.
LAST_SAVED_PROMPT = None  # kept only to avoid ImportError in legacy importers

NODE_DIR = Path(__file__).parent
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
CUSTOM_ONLY_STYLE = "✍️ Custom Only (no preset)"

DEFAULT_STYLES = {
    "📝 Enhance": "Write one production-ready prompt paragraph in the same language as the user. Expand the idea with concrete subject, action, environment, lighting, camera, composition, color, texture, mood, and style details. Output only the final prompt paragraph.",
    "📝 Refine": "Write one polished prompt paragraph in the same language as the user. Preserve the core intent, remove redundancy and contradiction, and add useful visual specificity for subject, scene, lighting, camera perspective, composition, palette, texture, and atmosphere. Output only the final prompt paragraph.",
    "📝 Creative Rewrite": "Write one fresh, imaginative prompt paragraph in the same language as the user. Preserve the core intent while adding cohesive cinematic atmosphere, gesture, micro-details, color relationships, light interaction, materials, depth, and composition. Output only the final prompt paragraph.",
    "📝 Detailed Visual": "Write one highly detailed visual prompt paragraph in the same language as the user. Include subject traits, pose, expression, wardrobe, materials, foreground, midground, background, lighting source and direction, palette, contrast, lens feel, depth of field, framing, and final aesthetic style. Output only the final prompt paragraph.",
    "📝 Artistic Style": "Write one artistic prompt paragraph in the same language as the user. Build a coherent visual direction with mood, palette, shape language, contrast, material feel, camera perspective, composition rhythm, and fitting style references. Output only the final prompt paragraph.",
    "📝 Technical Specs": "Write one clear technical photography or cinematography prompt paragraph in the same language as the user. Include camera distance, angle, lens feel, aperture or depth of field, focus target, lighting type and direction, color temperature, contrast, framing, background separation, texture rendering, and final image style. Output only the final prompt paragraph.",
}

_EMPTY_THINK_RE = re.compile(r"<think[^>]*>\s*</think>", flags=re.IGNORECASE | re.DOTALL)
_STREAM_HEARTBEAT_INTERVAL_SECONDS = 3.0


def _describe_prompt_enhancer_thinking(requested_thinking: bool, effective_thinking: bool, raw_text: str) -> str:
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
    return (
        f"requested={bool(requested_thinking)} effective={bool(effective_thinking)} "
        f"raw={raw_state} terminal_reasoning={terminal_state} final_output=clean_prompt"
    )


def _maybe_emit_hf_prompt_stream_heartbeat(stage_label: str, started_at: float, last_status_at: float, full_text: str) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    cleaned_preview = clean_model_output(full_text, OutputCleanConfig(mode="prompt"))
    if cleaned_preview and len(cleaned_preview.strip()) >= 32:
        return now
    elapsed = max(0.0, now - started_at)
    print(f"[QwenVL HF] {stage_label}: reasoning hidden; still generating... ({len(full_text)} chars, {elapsed:.1f}s)")
    return now


def _load_prompt_styles() -> dict[str, str]:
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwen_text = data.get("qwen_text") or {}
        styles = qwen_text.get("styles") or {}
        if isinstance(styles, dict) and styles:
            resolved = {
                name: entry.get("system_prompt", "")
                for name, entry in styles.items()
                if isinstance(entry, dict) and entry.get("system_prompt")
            }
            if resolved:
                return resolved
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] Prompt style load failed: {exc}")
    return DEFAULT_STYLES


PROMPT_STYLES = _load_prompt_styles()
PROMPT_STYLES = {CUSTOM_ONLY_STYLE: "", **PROMPT_STYLES}

PROMPT_ENHANCER_TOOLTIPS = {
    "prompt_text": "Prompt text to enhance. Leave blank to emit the selected preset instruction as the base prompt.",
    "enhancement_style": "Preset enhancement style. Use Custom Only when you want custom_system_prompt to fully control the instruction.",
    "custom_system_prompt": "Optional extra instruction. Required when using Custom Only; otherwise it is prepended to the selected style.",
    "max_tokens": "Maximum new tokens for the enhanced prompt. Increase only when the model truncates useful detail.",
    "temperature": "Sampling randomness. Lower is more stable; higher is more varied.",
    "top_p": "Nucleus sampling cutoff. Lower values restrict token choice; 0.9 is a balanced default.",
    "repetition_penalty": "Values above 1.0 reduce repeated phrases in the enhanced prompt.",
    "keep_model_loaded": "Keep the HF model in memory after generation so repeated prompt enhancement skips model loading.",
    "seed": "Sampling seed. Reusing it with identical inputs can reuse the saved prompt result.",
}


class ThinkingLLM_QwenVL_PromptEnhancer(QwenVLBase):
    STYLES = PROMPT_STYLES

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ENHANCED_OUTPUT", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def __init__(self):
        super().__init__()
        self.text_model = None
        self.text_processor = None
        self.text_tokenizer = None
        self.text_signature = None

    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_ALL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        styles = list(cls.STYLES.keys())
        preferred_style = "📝 Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "📝 Enhance")
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": PROMPT_ENHANCER_TOOLTIPS["prompt_text"]}),
                "enhancement_style": (styles, {"default": default_style, "tooltip": PROMPT_ENHANCER_TOOLTIPS["enhancement_style"]}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": PROMPT_ENHANCER_TOOLTIPS["custom_system_prompt"]}),
                "max_tokens": ("INT", {"default": 1024, "min": 32, "max": 16384, "tooltip": PROMPT_ENHANCER_TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "tooltip": PROMPT_ENHANCER_TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": PROMPT_ENHANCER_TOOLTIPS["top_p"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "tooltip": PROMPT_ENHANCER_TOOLTIPS["repetition_penalty"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": PROMPT_ENHANCER_TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": PROMPT_ENHANCER_TOOLTIPS["seed"]}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Show readable generation progress and thinking-status summaries in the ComfyUI terminal. When enabled, fixed-seed prompt reuse is bypassed so a fresh streamed run can occur."}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. Prompt enhancers still return a cleaned final prompt, so terminal reasoning may be hidden or empty."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": TOOLTIPS["hf_token"]}),
            }
            ,
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def process(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        prompt_text,
        enhancement_style,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        keep_last_prompt=False,
        stream_tokens_to_terminal=False,
        unique_id=None,
        extra_pnginfo=None,
        enable_thinking=True,
        hf_token="",
    ):
        node_class = "ThinkingLLM_QwenVL_PromptEnhancer"
        input_signature = build_node_input_signature(
            model_name=model_name,
            quantization=quantization,
            attention_mode=attention_mode,
            use_torch_compile=bool(use_torch_compile),
            device=device,
            prompt_text=prompt_text,
            enhancement_style=enhancement_style,
            custom_system_prompt=custom_system_prompt,
            enable_thinking=bool(enable_thinking),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Auto-retrieve saved prompt when seed is fixed (no keep_last_prompt needed)
        saved_prompt = get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        if saved_prompt and not stream_tokens_to_terminal:
            print(f"[QwenVL PromptEnhancer HF] Fixed seed {seed} matched — using per-node prompt: {saved_prompt[:50]}...")
            return (saved_prompt, "")
        if saved_prompt and stream_tokens_to_terminal:
            print("[QwenVL PromptEnhancer HF] Streaming requested — bypassing fixed-seed prompt reuse for a fresh streamed run")
        if keep_last_prompt:
            print(f"[QwenVL PromptEnhancer HF] Keep last prompt enabled but no saved prompt found — returning empty")
            return ("", "")

        # Always generate unless keep_last_prompt was requested
        print(f"[QwenVL PromptEnhancer HF] Generating new prompt")

        is_custom_only = enhancement_style == CUSTOM_ONLY_STYLE
        style_instruction = "" if is_custom_only else self.STYLES.get(
            enhancement_style,
            next(iter(self.STYLES.values()), ""),
        ).strip()
        custom_instruction = custom_system_prompt.strip()
        base_instruction = "\n\n".join(part for part in (custom_instruction, style_instruction) if part)
        if not base_instruction and is_custom_only:
            raise ValueError("custom_system_prompt is required when using Custom Only (no preset).")
        base_instruction = "\n\n".join(part for part in (base_instruction, prompt_output_guard()) if part)
        user_prompt = prompt_text.strip() or "Describe a scene vividly."
        merged_prompt = f"{user_prompt}\n\n{base_instruction}".strip()
        if model_name in HF_TEXT_MODELS:
            enhanced, enhanced_raw = self._invoke_text(
                model_name,
                quantization,
                device,
                merged_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                keep_model_loaded,
                seed,
                stream_to_terminal=stream_tokens_to_terminal,
                unique_id=unique_id,
                enable_thinking=enable_thinking,
                hf_token=hf_token,
            )
            raw_trace = f"[HF TEXT GENERATION]\n{enhanced_raw}"
        else:
            enhanced = self._invoke_qwen(
                model_name,
                quantization,
                attention_mode,
                use_torch_compile,
                device,
                merged_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                keep_model_loaded,
                seed,
                stream_to_terminal=stream_tokens_to_terminal,
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
                enable_thinking=enable_thinking,
                hf_token=hf_token,
            )
            hf_token = ""
            key = _make_node_state_key(node_class, unique_id, extra_pnginfo)
            entry = NODE_PROMPT_STATE.get(key, {})
            raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""

        final = enhanced.strip()

        # Persist per-node for future keep_last_prompt=True
        set_node_saved_prompt(node_class, unique_id, extra_pnginfo, final, raw_trace=raw_trace, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        print(f"[QwenVL PromptEnhancer HF] Saved per-node prompt: {final[:50]}...")

        return (final, raw_trace)

    def _invoke_qwen(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        stream_to_terminal=False,
        unique_id=None,
        extra_pnginfo=None,
        enable_thinking=True,
        hf_token: str | None = None,
    ):
        output = self.run(
            model_name=model_name,
            quantization=quantization,
            preset_prompt="🪄 Prompt Refine & Expand",
            custom_prompt=prompt,
            image=None,
            video=None,
            frame_count=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            attention_mode=attention_mode,
            use_torch_compile=use_torch_compile,
            device=device,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            node_class="ThinkingLLM_QwenVL_PromptEnhancer",
            stream_to_terminal=stream_to_terminal,
            enable_thinking=enable_thinking,
            hf_token=hf_token,
        )
        return output[0]

    def _load_text_model(self, model_name, quantization, device_choice, unique_id=None, hf_token: str | None = None):
        info = HF_TEXT_MODELS.get(model_name, {})
        repo_id = info.get("repo_id")
        if not repo_id:
            raise ValueError(f"[QwenVL] Missing repo_id for text model: {model_name}")

        prefers_processor = "gemma" in model_name.lower() or "gemma" in repo_id.lower()
        model_source = ensure_model(
            model_name,
            require_processor=prefers_processor,
            node_id=unique_id,
            progress_label=f"QwenVL PromptEnhancer HF Download: {model_name}",
            hf_token=hf_token,
        )

        if device_choice == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device = device_choice

        if quantization == Quantization.Q4:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == Quantization.Q8:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_cfg = None

        signature = (repo_id, quantization, device)
        if self.text_model is not None and self.text_signature == signature:
            ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)
            return

        self.text_model = None
        self.text_processor = None
        self.text_tokenizer = None
        self.text_signature = None

        load_kwargs = {}
        if quant_cfg:
            load_kwargs["quantization_config"] = quant_cfg
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

        print(f"[QwenVL] Loading text model {model_name} ({quantization})")
        tokenizer_kwargs = {"trust_remote_code": True}

        def _load_tokenizer(prefer_slow: bool = False):
            if prefers_processor:
                processor = AutoProcessor.from_pretrained(model_source, use_fast=not prefer_slow, **tokenizer_kwargs)
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError(f"[QwenVL] AutoProcessor for {repo_id} did not expose a tokenizer")
                self.text_processor = processor
                return tokenizer
            self.text_processor = None
            return AutoTokenizer.from_pretrained(model_source, use_fast=not prefer_slow, **tokenizer_kwargs)

        try:
            self.text_tokenizer = _load_tokenizer(prefer_slow=False)
        except ValueError as exc:
            message = str(exc)
            missing_backend = "sentencepiece or tiktoken" in message.lower()
            if missing_backend:
                print("[QwenVL] Tokenizer backend missing; installing sentencepiece and tiktoken into the active ComfyUI Python environment.")
                install_cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "sentencepiece",
                    "tiktoken",
                ]
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    stdout = (result.stdout or "").strip()
                    stderr = (result.stderr or "").strip()
                    raise RuntimeError(
                        "[QwenVL] Failed to install tokenizer backends automatically. "
                        f"Command: {' '.join(install_cmd)}\nstdout:\n{stdout or '<empty>'}\nstderr:\n{stderr or '<empty>'}"
                    ) from exc
                importlib.invalidate_caches()
                for module_name in ("sentencepiece", "tiktoken"):
                    try:
                        importlib.import_module(module_name)
                    except Exception:
                        pass
            try:
                self.text_tokenizer = _load_tokenizer(prefer_slow=True)
            except Exception as retry_exc:
                raise RuntimeError(
                    f"[QwenVL] Failed to load tokenizer for {repo_id}. "
                    f"Initial error: {exc}. Retry error: {retry_exc}"
                ) from retry_exc
        self.text_model = AutoModelForCausalLM.from_pretrained(model_source, trust_remote_code=True, **load_kwargs).eval()
        self.text_model.to(device)
        ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)
        # Detect architecture from loaded model config
        hf_model_type = getattr(self.text_model.config, "model_type", None)
        self.hf_model_type = hf_model_type
        self.is_qwen35 = hf_model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_vl") if hf_model_type else "qwen3.5-" in model_name.lower()
        self.supports_qwen_soft_think = hf_model_type == "qwen3" if hf_model_type else "qwen3-" in model_name.lower()
        if self.is_qwen35:
            print(f"[QwenVL] Qwen3.5 detected (model_type={hf_model_type}): Will disable thinking in chat template.")
        self.text_signature = signature

    def _invoke_text(
        self,
        model_name,
        quantization,
        device,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        stream_to_terminal=False,
        unique_id=None,
        enable_thinking=True,
        hf_token: str | None = None,
    ):
        """Returns (cleaned_prompt, raw_generated_text) tuple."""
        self._load_text_model(model_name, quantization, device, unique_id=unique_id, hf_token=hf_token)
        hf_token = None
        ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)

        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device

        is_qwen35 = getattr(self, "is_qwen35", False)
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        context_window = resolve_qwen_context_window(getattr(self.text_model, "config", None))

        def _build_inputs(thinking_enabled: bool):
            effective_prompt = apply_qwen_soft_thinking_directive(
                prompt,
                thinking_enabled,
                supports_soft_switch=supports_soft_think,
            )
            messages = [{"role": "user", "content": effective_prompt}]
            local_template_kwargs = dict(template_kwargs)
            if is_qwen35 or supports_soft_think:
                local_template_kwargs["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
            try:
                formatted_prompt = self.text_tokenizer.apply_chat_template(messages, **local_template_kwargs)
            except Exception:
                formatted_prompt = effective_prompt
            inputs = self.text_tokenizer(formatted_prompt, return_tensors="pt").to(device_choice)
            return formatted_prompt, inputs

        requested_thinking = bool(enable_thinking)
        _, inputs = _build_inputs(requested_thinking)
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        prompt_tokens = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
        effective_thinking = resolve_qwen_thinking_mode(
            enable_thinking,
            max_tokens,
            label="QwenVL PromptEnhancer HF",
            prompt_tokens=prompt_tokens,
            context_window=context_window,
        )
        if effective_thinking != requested_thinking:
            _, inputs = _build_inputs(effective_thinking)

        if stream_to_terminal:
            print("[QwenVL HF] Prompt enhancer terminal stream shows readable progress only; reasoning text may be hidden or stripped from the final prompt.")

        if is_qwen35 or supports_soft_think:
            if supports_soft_think:
                directive = "/think" if effective_thinking else "/no_think"
                print(f"[QwenVL] Qwen3 HF prompt enhancer: Thinking {'enabled' if effective_thinking else 'disabled'} via chat template and {directive}.")
        else:
            print(f"[QwenVL] Non-Qwen model in HF text path: thinking toggle is advisory (model may ignore).")
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": self.text_tokenizer.eos_token_id,
            "pad_token_id": self.text_tokenizer.eos_token_id,
        }

        # Optional: Apply seed for generation reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if stream_to_terminal:
            try:
                from transformers import TextIteratorStreamer
                from threading import Thread
                stage_label = "INITIAL GENERATION"
                stream_display = TerminalStreamDisplay("QwenVL PromptEnhancer HF", suppress_planning=True, compact=False)
                streamer = TextIteratorStreamer(self.text_tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs["streamer"] = streamer
                # Launch generation in a thread so we can iterate the streamer
                import threading
                thread = threading.Thread(target=self.text_model.generate, kwargs={**inputs, **gen_kwargs})
                print(f"[QwenVL PromptEnhancer HF] STREAMING {stage_label}")
                stream_display.start_stage(stage_label)
                stage_started_at = time.monotonic()
                last_status_at = stage_started_at
                thread.start()
                full_streamed = ""
                for token_str in streamer:
                    if token_str:
                        throw_exception_if_processing_interrupted()
                        full_streamed += token_str
                        stream_display.push(token_str)
                        last_status_at = _maybe_emit_hf_prompt_stream_heartbeat(stage_label, stage_started_at, last_status_at, full_streamed)
                thread.join()
                stream_display.end_stage()
                if not full_streamed.strip():
                    raise RuntimeError("[QwenVL] HF streaming returned empty response")
                raw_text = full_streamed.strip()
                result = clean_model_output(raw_text, OutputCleanConfig(mode="prompt")) or raw_text
                print(f"[QwenVL HF] Thinking status: {_describe_prompt_enhancer_thinking(requested_thinking, effective_thinking, raw_text)}")
                if not keep_model_loaded:
                    self.text_model = None
                    self.text_processor = None
                    self.text_tokenizer = None
                    self.text_signature = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return result, raw_text
            except ImportError:
                print("[QwenVL PromptEnhancer HF] TextIteratorStreamer not available — falling back to non-streaming")
                stream_to_terminal = False

        if not stream_to_terminal:
            outputs = self.text_model.generate(**inputs, **gen_kwargs)
            throw_exception_if_processing_interrupted()

        # Non-streaming path: Strip out the input tokens to get just the generated response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_text = self.text_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        result = clean_model_output(raw_text, OutputCleanConfig(mode="prompt")) or raw_text

        if stream_to_terminal:
            print(f"[QwenVL HF] Thinking status: {_describe_prompt_enhancer_thinking(requested_thinking, effective_thinking, raw_text)}")

        if not keep_model_loaded:
            self.text_model = None
            self.text_processor = None
            self.text_tokenizer = None
            self.text_signature = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result, raw_text

NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_QwenVL_PromptEnhancer": ThinkingLLM_QwenVL_PromptEnhancer,
    "AILab_QwenVL_PromptEnhancer": ThinkingLLM_QwenVL_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_QwenVL_PromptEnhancer": "ThinkingLLM Prompt Enhancer",
    "AILab_QwenVL_PromptEnhancer": "ThinkingLLM Prompt Enhancer",
}
