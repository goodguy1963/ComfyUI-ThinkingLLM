# ComfyUI-ThinkingLLM

**ThinkingLLM** is a local-LLM custom-node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps Qwen3.5, Qwen3, Qwen-VL, Gemma 4, and Whisper ASR behind a cleaner node interface with thinking-mode control, audio/vision workflows, live token streaming, and a raw trace output for debugging.

📺 **[Watch the demo](https://youtu.be/K1JsvnzujOw)** for a better viewing experience.

> GPL-3.0 fork of Deaquay/ComfyUI-Qwen3.5-Uncensored, itself derived from huchukato/ComfyUI-QwenVL-Mod and 1038lab/ComfyUI-QwenVL. See [LICENSE](LICENSE).

![ThinkingLLM Demo](docs/DemoThinkingLLM%20(1).gif)
![Node screenshot](docs/Screenshot%20of%20Notes.png)

> 🎥 Prefer a narrated version? **[Watch the demo on YouTube](https://youtu.be/K1JsvnzujOw)**

## What ThinkingLLM adds

### Model guidance box

ThinkingLLM nodes add a read-only `recommended_settings` info box.

- It explains the selected model family and recommended generation settings.
- It does not change your saved widget values.
- For GGUF audio-capable nodes, it tells you whether the selected model actually supports audio.
- In Advanced GGUF nodes, if you connect or expose audio for a model without known audio support, the box warns you instead of silently implying that audio will work.

### The Thinking Toggle

Every ThinkingLLM node exposes an `enable_thinking` toggle.

- When it is on, the model may reason internally before answering. That can make a run take longer.
- For easy prompts, the model can still decide that explicit reasoning is unnecessary and answer directly. That is not a bug. It is a feature.
- When it is off, the node sends a `/no_think` style directive to discourage visible reasoning and push a more direct one-shot answer.

### Live token streaming

Enable `stream_tokens_to_terminal` to print tokens into the ComfyUI terminal as they arrive.

This is useful when:

- a prompt seems stuck and you want to see whether the model is actually working
- you want to debug a bad prompt while it is going wrong
- you want to understand whether the model is thinking or simply producing a long answer

### Raw trace output

Each node returns a second string output named `RAW_TRACE`.

- The main output is the cleaned answer for downstream workflow use.
- `RAW_TRACE` preserves the raw generation stream, including visible thinking blocks when the model emits them.

That makes it possible to inspect what the model thought about the prompt while still keeping the primary output clean.

### Audio support

Audio is available through dedicated nodes so the model list stays honest:

- `ThinkingLLM Gemma 4 Audio (GGUF)` for Gemma 4 audio-capable models.
- `ThinkingLLM Whisper ASR` for reliable speech-to-text transcription through `faster-whisper`.

The Advanced GGUF node also exposes optional `audio`, but this is for power users. Use the info box before assuming a model can hear audio. Normal Qwen/Qwen-VL models are not audio models.

### Mask-focused image analysis and object removal

The four ThinkingLLM vision nodes accept an optional ComfyUI `MASK` input:

- `ThinkingLLM`
- `ThinkingLLM (Advanced)`
- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`

White mask pixels select the area to analyze. Black pixels preserve surrounding context, and soft mask values create a gradual transition. ThinkingLLM keeps the selected area at normal brightness while dimming everything outside it, helping the vision model focus on the mask without losing the scene's geometry, lighting, or background context.

#### Object-removal workflow

1. Connect the source image to ThinkingLLM's `image` input.
2. Connect the corresponding mask to its `mask` input.
3. Select `👁️ Remove (Mask Area)` from `preset_prompt`.
4. Optionally describe the desired replacement in `custom_prompt`, such as `continue the brick wall` or `fill with matching wooden floor`.
5. Connect `RESPONSE` to the positive prompt input of your diffusion inpainting workflow.
6. Send the original image and mask separately to the diffusion model's inpainting inputs.

ThinkingLLM does not edit the image itself. It examines the masked area and surrounding scene, then generates a positive description of what should appear inside the mask. The downstream diffusion model performs the actual replacement.

#### Previewing the LLM mask view

`ThinkingLLM (Advanced)` and `ThinkingLLM (GGUF Advanced)` return a third output named `MASK_PREVIEW`. Connect it to a ComfyUI **Preview Image** node to inspect the exact highlighted image used for mask-focused analysis. The selected area remains at normal brightness and the surrounding scene is dimmed to 20%. When no mask is connected, the preview returns the original connected image.

Without a custom instruction, the preset infers the most natural background continuation. It describes matching surfaces, perspective, materials, texture, color, lighting, shadows, reflections, and image style while avoiding phrases such as "remove the object" or "erase this area."

Example output:

> Continuous warm beige plaster wall with subtle uneven texture, matching the existing perspective, soft window illumination, natural tonal variation, and the floor-edge shadows.

#### Notes and limitations

- Masks apply to still images only, not video input.
- Mask dimensions are resized automatically to match the image.
- An empty mask or a mask without an image produces a clear error.
- Image batches continue to use the first image and first mask.
- Precise masks with a small amount of surrounding context generally produce the best reconstruction prompt.

## Nodes

### Transformers / HF nodes

This is the recommended default path.

- `ThinkingLLM`
- `ThinkingLLM (Advanced)`
- `ThinkingLLM Prompt Enhancer`

The HF path is the most straightforward option for Windows and Linux when you want the fewest install complications.

HF tokenizer note: some Gemma and Qwen variants need `sentencepiece` or `tiktoken` available in the ComfyUI Python environment. ThinkingLLM now declares those dependencies and will attempt to install them automatically if a tokenizer backend is missing at first load.

#### Shared ComfyUI text encoders

Compatible single-file checkpoints already in `ComfyUI/models/text_encoders` appear in the model selector with a `[ComfyUI]` prefix. ThinkingLLM loads them through ComfyUI, so it does not download or duplicate a Transformers snapshot.

- Qwen3-VL 4B and 8B `.safetensors`, including `qwen3vl_8b_fp8_scaled.safetensors`
- Gemma 3 12B `.safetensors`, including `gemma-3-12b-it-heretic-v2_int8.safetensors`

The MiniMax H3/Qwen3-VL 32B conditioning checkpoint is intentionally excluded because it is not a complete text-generation model. The valid SafeTensors extension is `.safetensors` (plural). ComfyUI-native generation supports one-beam sampling; terminal output is printed after generation rather than streamed token by token.

### GGUF / llama.cpp nodes

- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`
- `ThinkingLLM Gemma 4 Audio (GGUF)`
- `ThinkingLLM Prompt Enhancer (GGUF)`

The GGUF path requires a multimodal-capable `llama-cpp-python` build. The normal PyPI package may not include the chat handlers needed for Qwen vision, Gemma 4, or MTMD audio. On Linux, ThinkingLLM auto-checks this at first GGUF use and attempts an automatic install of a matching JamePeng backend. Use the setup notes in [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md) if you need to override the wheel/package source.

## Workflow tips

- **Pre-process input images** — use a resize or scale node before ThinkingLLM so large images don't blow up the context window.
- **Display the output** — connect RESPONSE or ENHANCED_OUTPUT to a **Show Text** or **Show Anything** node to see the generated text in the UI.
- **Inspect reasoning** — connect the RAW_TRACE output to a second Show Text node to see what the model was thinking.
- **Use the dedicated audio nodes first** — Gemma 4 Audio is for audio understanding; Whisper ASR is for transcription. The Advanced GGUF audio input is intentionally more flexible, but the info box will warn you when the selected model has no curated audio support.

### MiniMax H3 prompt preset

Select `🎬 MiniMax H3 Text-to-Video` for text and keyframe prompting or `🖼️ MiniMax H3 Reference-to-Video` for full-reference generation. Vision nodes support text-to-video, first-frame, first-and-last-frame, last-frame, and full-reference prompting; text-only Prompt Enhancer nodes emit the T2VA format. The presets keep timed shots, camera motion, dialogue, soundscape, music, keyframe alignment, and reference-retention fields in the structure expected by H3.

See the official [MiniMax H3 ComfyUI workflows](https://docs.comfy.org/tutorials/video/minimax/minimax-h3), [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), and [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

## Supported models

ThinkingLLM supports these model families out of the box, with pre-configured entries in `hf_models.json` and `gguf_models.json`.

Commercial-release model dropdowns group entries by rights status and accept only reviewed, locked models. Community dropdowns keep their normal model names and download behavior; gated Hugging Face repositories still require the model owner's access terms and a read token.

### Qwen — Vision-language (HF)
- Qwen3.5 4B / 9B
- Qwen3-VL 4B / 8B (Instruct, abliterated, unredacted)
- Qwen3-VL 32B Heretic
- Qwen2.5-VL 3B / 7B

### Qwen — Vision-language (GGUF)
- Qwen3.5 4B / 9B / 27B (Q4 K M through BF16)
- Qwen3-VL 4B / 8B / 32B (Instruct, abliterated, thinking)
- Qwen2.5-VL 3B / 7B

### Qwen — Text-only (HF)
- Qwen3 0.6B / 4B / 8B
- Qwen3.5 4B / 9B (base + heretic + uncensored)

### Qwen — Text-only (GGUF)
- Qwen3 4B / 8B (abliterated, Josiefied, base)
- Qwen3.5 4B / 9B / 27B (base + uncensored HauhauCS)

### Gemma 4 — Vision-language (HF)
- Gemma-4-E2B-it / Uncensored
- Gemma-4-E4B-it / Uncensored
- Gemma-4-12B
- Gemma-4-26B-A4B-it / Heretic
- Gemma-4-31B-it

### Gemma 4 — Vision-language (GGUF)
- Gemma-4-E2B / E4B (Q4 K M through BF16, uncensored)
- Gemma-4-12B-it (Q4/Q5/Q6/Q8/BF16/UD-Q4)
- Gemma-4-26B-A4B-it (Q4 K M through BF16)
- Gemma-4-31B-it (Q4 K M through BF16)

### Gemma 4 — Audio (GGUF)
- Gemma-4-E2B / E4B when present in the GGUF multimodal catalog
- Gemma-4-12B-it via `unsloth/gemma-4-12b-it-GGUF`

Use `ThinkingLLM Gemma 4 Audio (GGUF)` for the clean audio-only interface. Gemma 4 audio currently depends on a recent multimodal `llama-cpp-python`/`Gemma4ChatHandler` build, so check `RAW_TRACE` if audio behaves unexpectedly.

### Whisper — ASR
- tiny
- base
- small
- medium
- large-v3
- distil-large-v3

Use `ThinkingLLM Whisper ASR` for transcription. It accepts a connected ComfyUI `AUDIO` input or an `audio_file_path` pointing to M4A, MP3, WAV, FLAC, or any FFmpeg-readable audio file. The node uses `faster-whisper`; first use may download the selected Whisper model into the Hugging Face cache. The default is `small` on CPU with `int8`, which is slower than CUDA but reliable on Windows; switch to CUDA once the local CTranslate2 CUDA runtime is confirmed working.

### Gemma 4 — Text-only (HF)
- Gemma-4-E2B-it / Uncensored
- Gemma-4-E4B-it / Uncensored
- Gemma-4-12B
- Gemma-4-26B-A4B-it / Heretic
- Gemma-4-31B-it

Local GGUF models placed in `models/LLM/GGUF/` with a matching mmproj file are auto-discovered. Pre-configured entries live in `hf_models.json` and `gguf_models.json`.

## Quick install

### ComfyUI Manager (recommended)
1. Open **ComfyUI Manager** → **Install via Registry**
2. Search for `ThinkingLLM`
3. Click **Install**

### Manual install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/goodguy1963/ComfyUI-ThinkingLLM.git
cd ComfyUI-ThinkingLLM
pip install -r requirements.txt
```

For GGUF vision support, follow [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md).

## Model locations

Models are discovered automatically.

- HF models: `ComfyUI/models/LLM/Qwen-VL/`
- GGUF models: `ComfyUI/models/LLM/GGUF/`

For local GGUF models, keep the matching mmproj file beside the model file.

## Platform notes

- HF is the recommended default path and is the simplest cross-platform option.
- GGUF on Windows needs a matching `win_amd64` vision-capable wheel.
- GGUF on Linux now auto-installs a matching backend on first use when possible. Override with `THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL` or `THINKINGLLM_LLAMA_CPP_LINUX_SPEC` when your server needs a different build.
- GGUF audio needs an audio-capable multimodal backend. `Gemma4ChatHandler` is used for Gemma 4.
- Whisper ASR needs `faster-whisper` and FFmpeg. The node returns an install hint if `faster-whisper` is missing.
- Flash Attention support is best on Linux. The nodes fall back when it is unavailable.
- The thinking toggle works best with Qwen3.5 and Qwen3 style models. Other architectures may ignore the steering.

## Production deployment

This repository provides ComfyUI nodes, not a complete production runtime. If you plan to expose a workflow through an API on RunPod Serverless, review the dependency, validation, testing, and rollback requirements in the [ComfyUI on RunPod production-readiness checklist](https://comfyrail.dev/production-readiness).

## FAQ

**Why does the node sometimes answer quickly and sometimes take longer?**

When thinking is enabled, the model may decide to reason before answering. For easy prompts it may skip explicit reasoning. That is expected behavior.

**How do I see what the model was thinking?**

Use the second `RAW_TRACE` output, or enable live terminal token streaming.

**How do I debug prompts that seem bad or stuck?**

Turn on `stream_tokens_to_terminal`. It is there specifically so you can see the generation live and catch bad prompt behavior sooner.

## Credits and fork lineage

This fork preserves the GPL-3.0 lineage of its predecessors.

- Deaquay/ComfyUI-Qwen3.5-Uncensored
- huchukato/ComfyUI-QwenVL-Mod
- 1038lab/ComfyUI-QwenVL
- Qwen Team (Alibaba Cloud)
- JamePeng/llama-cpp-python
- comfyanonymous/ComfyUI

Maintainer: goodguy1963  
Planned repo: [goodguy1963/ComfyUI-ThinkingLLM](https://github.com/goodguy1963/ComfyUI-ThinkingLLM)

## License

GPL-3.0. See [LICENSE](LICENSE).
