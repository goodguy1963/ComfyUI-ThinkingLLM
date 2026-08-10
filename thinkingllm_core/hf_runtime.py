"""Hugging Face multimodal runtime independent from ComfyUI node declarations."""

import gc
import time

import numpy as np
import torch
from PIL import Image
try:
    from transformers import AutoProcessor, AutoTokenizer
except ImportError:
    AutoProcessor = None
    AutoTokenizer = None
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

import folder_paths
from AILab_StreamDisplay import TerminalStreamDisplay
from comfy.model_management import throw_exception_if_processing_interrupted

from thinkingllm_core.hf_models import (
    COMMERCIAL_RELEASE,
    HF_ALL_MODELS,
    NO_PRESET_PROMPT,
    SYSTEM_PROMPTS,
    Quantization,
    ensure_cuda_vram_headroom,
    ensure_model,
    get_device_info,
    is_fp8_model,
    normalize_device_choice,
    quantization_config,
    read_hf_model_type,
    resolve_attention_mode,
)
from thinkingllm_core.media import (
    MASK_FOCUS_INSTRUCTION,
    apply_mask_highlight,
    get_image_hash,
    get_video_hash,
)
from thinkingllm_core.model_access import resolve_model_catalog_name
from thinkingllm_core.prompt_contracts import (
    apply_qwen_soft_thinking_directive,
    apply_video_duration_context,
    ensure_video_prompt_duration,
    resolve_qwen_context_window,
    resolve_qwen_thinking_mode,
    resolve_video_duration,
    validate_minimax_reference_request,
    validate_minimax_source_duration,
)
from thinkingllm_core.prompt_state import (
    PROMPT_CACHE,
    build_node_input_signature,
    get_cache_key,
    get_node_saved_prompt_with_seed,
    log_llm_input,
    save_prompt_cache,
    set_node_saved_prompt,
)

_STREAM_HEARTBEAT_INTERVAL_SECONDS = 3.0


def _maybe_emit_hf_stream_heartbeat(stage_label: str, started_at: float, last_status_at: float, full_text: str) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    elapsed = max(0.0, now - started_at)
    print(f"[QwenVL HF] {stage_label}: still generating... ({len(full_text)} chars, {elapsed:.1f}s)")
    return now


class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL] Node on {self.device_info['device_type']}")

    def clear(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
        unique_id=None,
        hf_token: str | None = None,
    ):
        model_name = resolve_model_catalog_name(HF_ALL_MODELS, model_name)
        model_info = HF_ALL_MODELS.get(model_name, {})
        comfy_filename = model_info.get("comfy_text_encoder")
        if comfy_filename:
            model_path = folder_paths.get_full_path_or_raise("text_encoders", comfy_filename)
            signature = (model_name, "comfy", model_path)
            if keep_model_loaded and self.model is not None and self.current_signature == signature:
                return
            self.clear()
            from comfy import sd as comfy_sd
            self.model = comfy_sd.load_clip(
                ckpt_paths=[model_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            self.current_signature = signature
            print(f"[QwenVL] Loaded shared ComfyUI text encoder: {comfy_filename}")
            return

        quant = Quantization.from_value(quant_value)  # Skip enforce_memory for now

        # Safety check: ensure quant is not None
        if quant is None:
            print(f"[QwenVL] Invalid quantization value: {quant_value}, falling back to FP16")
            quant = Quantization.FP16

        # Check if BitsAndBytes quantization is being used
        is_bnb_quantization = quant in [Quantization.Q4, Quantization.Q8]

        # Check if this is a pre-quantized FP8 model
        is_prequantized_fp8 = is_fp8_model(model_name) or HF_ALL_MODELS.get(model_name, {}).get("quantized", False)

        # Determine if we need to force SDPA (for FP8 or BitsAndBytes models)
        force_sdpa = is_prequantized_fp8 or is_bnb_quantization

        # Resolve attention mode with force_sdpa flag
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)

        # Additional info messages for forced SDPA
        if force_sdpa and attention_mode in ["auto", "sage", "flash_attention_2"]:
            if is_prequantized_fp8:
                print("[QwenVL] FP8 model detected - forcing SDPA attention")
            elif is_bnb_quantization:
                print("[QwenVL] BitsAndBytes quantization detected - forcing SDPA attention")

        print(f"[QwenVL] Attention backend selected: {attn_impl}")

        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_requested)
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            ensure_cuda_vram_headroom("QwenVL", min_free_gb=1.0, min_free_ratio=0.08)
            return
        self.clear()
        model_path = ensure_model(
            model_name,
            require_processor=True,
            node_id=unique_id,
            progress_label=f"QwenVL HF Download: {model_name}",
            hf_token=hf_token,
        )
        quant_config, dtype = quantization_config(model_name, quant)

        # Handle attention mode for loading
        # SageAttention requires loading with SDPA first, then patching
        actual_attn_impl = attn_impl
        if attn_impl == "sage":
            actual_attn_impl = "sdpa"

        # MEMORY DEBUGGING: Check memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[QwenVL] 📊 Memory BEFORE load - Allocated: {allocated_before:.1f}GB, Reserved: {reserved_before:.1f}GB, Total: {total_memory:.1f}GB")

            # Check what's using memory
            if allocated_before > 2.0:
                print(f"[QwenVL] ⚠️  WARNING: {allocated_before:.1f}GB already allocated before loading!")

        # DEBUG: Print quantization config details
        print(f"[QwenVL] 🔍 DEBUG - Model: {model_name}, Quant: {quant}, Quant config: {quant_config}")
        print(f"[QwenVL] 🔍 DEBUG - Device: {device}, Dtype: {dtype}")
        print(f"[QwenVL] 🔍 DEBUG - Attention impl: {actual_attn_impl}")

        # Quantization is back - enforce_memory was the problem

        load_kwargs = {
            "device_map": "auto" if device != "cpu" and torch.cuda.is_available() else "cpu",
            "dtype": dtype or torch.float16,
            "attn_implementation": actual_attn_impl,
            "local_files_only": COMMERCIAL_RELEASE,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if device != "cpu" and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = max(1, int(gpu_mem / (1024**3)))
            load_kwargs["max_memory"] = {
                0: f"{gpu_mem_gb}GB",
                "cpu": "16GB",
            }

        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()

        # Apply SageAttention patching if needed
        if attn_impl == "sage":
            try:
                from sageattention_patch import set_sage_attention
                set_sage_attention(self.model)
                print("[QwenVL] SageAttention patching applied successfully")
            except Exception as e:
                print(f"[QwenVL] SageAttention patching failed: {e}")
                print("[QwenVL] Falling back to SDPA attention")
                # Model is already loaded with SDPA, so we can continue

        # MEMORY CLEANUP: Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[QwenVL] GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=COMMERCIAL_RELEASE,
            trust_remote_code=not COMMERCIAL_RELEASE,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=COMMERCIAL_RELEASE,
            trust_remote_code=not COMMERCIAL_RELEASE,
        )
        # Detect architecture from config.json instead of relying on model name
        hf_model_type = read_hf_model_type(model_path)
        self.hf_model_type = hf_model_type
        self.is_qwen35 = hf_model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_vl") if hf_model_type else "qwen3.5-" in model_name.lower()
        self.supports_qwen_soft_think = hf_model_type == "qwen3" if hf_model_type else "qwen3-" in model_name.lower()
        if self.is_qwen35:
            print(f"[QwenVL] Qwen3.5 detected (model_type={hf_model_type}): Will disable thinking in chat template.")
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.shape[-1] == 1:
            return Image.fromarray(array[..., 0], mode="L")
        if array.shape[-1] == 4:
            return Image.fromarray(array, mode="RGBA")
        return Image.fromarray(array[..., :3], mode="RGB")

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        model_name="",
        stream_to_terminal=False,
        enable_thinking=True,
        seed=0,
        preset_name="",
    ):
        model_info = HF_ALL_MODELS.get(model_name, {})
        native_family = model_info.get("native_family")
        if native_family:
            if num_beams != 1:
                raise ValueError("[QwenVL] ComfyUI text encoders support num_beams=1 only")
            if native_family == "gemma3" and video is not None:
                raise ValueError("[QwenVL] Gemma 3 shared text encoders support text and image input, not video")

            tokenize_kwargs = {"skip_template": False, "thinking": bool(enable_thinking)}
            if native_family == "gemma3":
                tokenize_kwargs["image"] = image[:1] if image is not None else None
            else:
                images = []
                if image is not None:
                    if image.dim() == 4 and image.shape[0] > 1:
                        print(f"[QwenVL] IMAGE input contains {image.shape[0]} items; using the first item only. Use the video input for multi-frame analysis.")
                    images.append(image[:1] if image.dim() == 4 else image)
                if video is not None:
                    frame_indexes = list(range(video.shape[0]))
                    if len(frame_indexes) > frame_count:
                        frame_indexes = np.linspace(0, len(frame_indexes) - 1, frame_count, dtype=int).tolist()
                    images.extend(video[index:index + 1] for index in frame_indexes)
                tokenize_kwargs["images"] = images

            throw_exception_if_processing_interrupted()
            log_llm_input(
                "QwenVL ComfyUI Native",
                "INITIAL GENERATION",
                preset_name,
                prompt_text,
                media={
                    "image": 1 if image is not None else 0,
                    "video_frames": max(0, len(tokenize_kwargs.get("images", [])) - (1 if image is not None else 0)),
                },
            )
            tokens = self.model.tokenize(prompt_text, **tokenize_kwargs)
            output_tokens = self.model.generate(
                tokens,
                do_sample=True,
                max_length=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=top_p,
                min_p=0.0,
                repetition_penalty=repetition_penalty,
                seed=seed,
                presence_penalty=0.0,
            )
            throw_exception_if_processing_interrupted()
            raw_text = self.model.decode(output_tokens).strip()
            if not raw_text:
                raise RuntimeError(
                    "[QwenVL] The model ended generation without producing text. Change the seed or add a more "
                    "specific custom prompt; the empty result was not cached."
                )
            if stream_to_terminal:
                print("[QwenVL] ComfyUI native generation completed (live token streaming is unavailable):")
                print(raw_text)
            return raw_text, raw_text

        # Memory optimization: clear cache before generation
        ensure_cuda_vram_headroom("QwenVL", min_free_gb=1.0, min_free_ratio=0.08)
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        is_qwen35 = getattr(self, "is_qwen35", False)
        context_window = resolve_qwen_context_window(getattr(self.model, "config", None))

        def _build_processed(thinking_enabled: bool):
            effective_prompt_text = apply_qwen_soft_thinking_directive(
                prompt_text,
                thinking_enabled,
                supports_soft_switch=supports_soft_think,
            )
            conversation = [{"role": "user", "content": []}]
            if image is not None:
                if image.dim() == 4 and image.shape[0] > 1:
                    print(f"[QwenVL] IMAGE input contains {image.shape[0]} items; using the first item only. Use the video input for multi-frame analysis.")
                conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
            if video is not None:
                frames = [self.tensor_to_pil(frame) for frame in video]
                if len(frames) > frame_count:
                    idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                    frames = [frames[i] for i in idx]
                if frames:
                    conversation[0]["content"].append({"type": "video", "video": frames})
            conversation[0]["content"].append({"type": "text", "text": effective_prompt_text})

            chat_kwargs = {}
            if is_qwen35 or supports_soft_think:
                chat_kwargs["enable_thinking"] = thinking_enabled

            chat = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs
            )

            images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
            video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
            videos = [video_frames] if video_frames else None
            processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
            return effective_prompt_text, chat, processed, {
                "image": len(images),
                "video_frames": len(video_frames),
            }

        requested_thinking = bool(enable_thinking)
        effective_prompt_text, chat, processed, media = _build_processed(requested_thinking)
        input_ids = processed.get("input_ids")
        prompt_tokens = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
        effective_thinking = resolve_qwen_thinking_mode(
            enable_thinking,
            max_tokens,
            label="QwenVL HF",
            prompt_tokens=prompt_tokens,
            context_window=context_window,
            quiet=stream_to_terminal,
        )
        if effective_thinking != requested_thinking:
            effective_prompt_text, chat, processed, media = _build_processed(effective_thinking)
        if supports_soft_think:
            directive = "/think" if effective_thinking else "/no_think"
            print(f"[QwenVL] Qwen3 detected: Thinking {'enabled' if effective_thinking else 'disabled'} via chat template and {directive}.")

        # Move to device more efficiently
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        # Memory-efficient generation parameters
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
            # --- Qwen3.5 Heretic Logic: Top K ---
            if is_qwen35:
                kwargs["top_k"] = 20
                print("[QwenVL] Qwen3.5 detected: Forcing top_k=20 for recommended tuning.")
        else:
            kwargs["do_sample"] = False

        log_llm_input(
            "QwenVL HF",
            "INITIAL GENERATION",
            preset_name,
            effective_prompt_text,
            formatted_text=chat,
            media=media,
        )

        # Optional: staged readable terminal streaming
        if stream_to_terminal:
            try:
                from AILab_StreamDisplay import TerminalStreamDisplay
                from transformers import TextIteratorStreamer
                from threading import Thread
                stage_label = "STREAMING"
                stream_display = TerminalStreamDisplay("QwenVL HF", suppress_planning=True, compact=True)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                kwargs["streamer"] = streamer
                thread = Thread(target=self.model.generate, kwargs={**model_inputs, **kwargs})
                stream_display.start_stage(stage_label)
                thread.start()
                full_streamed = ""
                for token_str in streamer:
                    if token_str:
                        throw_exception_if_processing_interrupted()
                        full_streamed += token_str
                        stream_display.push_compact(token_str)
                thread.join()
                stream_display.end_compact()
                stream_display.end_stage()
                if not full_streamed.strip():
                    raise RuntimeError("[QwenVL] HF streaming returned empty response")
                cleaned_text = full_streamed.strip()
                raw_text = full_streamed.strip()
                return cleaned_text, raw_text
            except ImportError:
                print("[QwenVL] TextIteratorStreamer not available — falling back to non-streaming")
                stream_to_terminal = False

        # Generate with interrupt support via background thread
        import threading, queue as qmod
        abort_event = threading.Event()
        output_queue: qmod.Queue = qmod.Queue()
        worker_error: Exception | None = None

        def _generate_worker():
            nonlocal worker_error
            try:
                result = self.model.generate(**model_inputs, **kwargs)
                output_queue.put(("ok", result))
            except Exception as exc:
                worker_error = exc
                output_queue.put(("error", None))

        worker = threading.Thread(target=_generate_worker, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    kind, result = output_queue.get(timeout=0.25)
                except qmod.Empty:
                    throw_exception_if_processing_interrupted()
                    continue
                if kind == "error":
                    break
                outputs = result
                break
        finally:
            abort_event.set()
            worker.join(timeout=5.0)
        if worker_error:
            raise worker_error
        throw_exception_if_processing_interrupted()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        raw_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
        cleaned_text = raw_text  # generate() returns raw; caller cleans if needed
        return cleaned_text, raw_text

    def run(self, model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt=False, unique_id=None, extra_pnginfo=None, node_class="QwenVL", stream_to_terminal=False, enable_thinking=True, hf_token: str | None = None, mask=None, llm_input_preset_name: str | None = None, duration_seconds=5.0):
        model_name = resolve_model_catalog_name(HF_ALL_MODELS, model_name)
        validate_minimax_reference_request(preset_prompt, custom_prompt)
        duration_preset_name = llm_input_preset_name or preset_prompt
        resolved_duration = resolve_video_duration(duration_preset_name, duration_seconds)
        validate_minimax_source_duration(duration_preset_name, duration_seconds, custom_prompt)
        effective_duration_seconds = resolved_duration["effective_seconds"] if resolved_duration else None
        duration_signature = ({
            "effective_duration_seconds": effective_duration_seconds,
            "video_prompt_contract_version": resolved_duration.get("contract_version"),
        } if resolved_duration else {})
        torch.manual_seed(seed)
        image, mask_hash = apply_mask_highlight(image, mask)
        image_hash = get_image_hash(image)
        if mask_hash:
            image_hash = f"{image_hash}:{mask_hash}"
        video_hash = get_video_hash(video)
        input_signature = build_node_input_signature(
            model_name=model_name,
            quantization=quantization,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image_hash=image_hash,
            video_hash=video_hash,
            frame_count=frame_count,
            attention_mode=attention_mode,
            use_torch_compile=bool(use_torch_compile),
            device=device,
            enable_thinking=bool(enable_thinking),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **duration_signature,
        )

        # Auto-retrieve saved prompt when seed is fixed
        saved = get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        if saved and not stream_to_terminal:
            print(f"[QwenVL] Preset {llm_input_preset_name or preset_prompt}: fixed seed {seed} matched; no LLM call was made.")
            return (saved,)
        if saved and stream_to_terminal:
            pass
        if keep_last_prompt:
            print(f"[QwenVL] Keep last prompt enabled — looking up per-node state")
            if saved:
                print(f"[QwenVL] Preset {llm_input_preset_name or preset_prompt}: using saved prompt; no LLM call was made.")
                return (saved,)
            else:
                print(f"[QwenVL] No per-node prompt found, returning empty")
                return ("",)

        # Always generate unless keep_last_prompt was requested
        if not stream_to_terminal:
            print(f"[QwenVL] Generating new prompt")

        prompt_template = "" if preset_prompt == NO_PRESET_PROMPT else SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)

        # Generate cache key with all inputs including seed
        cache_key = get_cache_key(model_name, preset_prompt, custom_prompt, image_hash, video_hash, seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, enable_thinking=enable_thinking, effective_duration_seconds=effective_duration_seconds, video_prompt_contract_version=resolved_duration.get("contract_version") if resolved_duration else None)

        # Check cache first (only for random mode)
        if cache_key in PROMPT_CACHE:
            cached_text = PROMPT_CACHE[cache_key].get("text", "")
            if cached_text:
                print(f"[QwenVL] Preset {llm_input_preset_name or preset_prompt}: cache hit for seed {seed}; no LLM call was made.")
                return (cached_text,)

        if custom_prompt and custom_prompt.strip():
            # Combine user input with template - custom prompt first for priority
            prompt = f"{custom_prompt.strip()}\n\n{prompt_template}" if prompt_template else custom_prompt.strip()
        else:
            prompt = prompt_template
        if mask_hash:
            prompt = f"{prompt}\n\n{MASK_FOCUS_INSTRUCTION}".strip()
        prompt = apply_video_duration_context(prompt, duration_preset_name, duration_seconds)

        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
            unique_id=unique_id,
            hf_token=hf_token,
        )
        hf_token = None
        try:
            text, raw_text = self.generate(
                prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
                model_name=model_name,
                stream_to_terminal=stream_to_terminal,
                enable_thinking=enable_thinking,
                seed=seed,
                preset_name=llm_input_preset_name or preset_prompt,
            )
            text = ensure_video_prompt_duration(
                text,
                duration_preset_name,
                duration_seconds,
                source_prompt_text=custom_prompt,
            )

            # Cache the generated text
            PROMPT_CACHE[cache_key] = {
                "text": text,
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "model": model_name,
                "preset": preset_prompt,
                "seed": seed,
                "image_hash": image_hash,
                "mask_hash": mask_hash,
                "video_hash": video_hash,
                **duration_signature,
            }
            save_prompt_cache()  # Save cache to file

            print(f"[QwenVL] Cached new prompt for seed {seed}: {cache_key[:8]}...")

            # Save the generated prompt for future per-node keep-last-prompt
            set_node_saved_prompt(node_class, unique_id, extra_pnginfo, text, raw_trace=raw_text, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
            print(f"[QwenVL] Saved per-node prompt: {text[:50]}...")

            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()
