# ComfyUI-ThinkingLLM

**ThinkingLLM** is a local-LLM custom-node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps Qwen3.5, Qwen3, Qwen-VL, Gemma 4, and Whisper ASR behind a cleaner node interface with thinking-mode control, audio/vision workflows, live token streaming, a raw trace output for debugging — and a secure OpenAI-compatible API node for remote providers (OpenRouter, OrcaRouter, …).

📺 **[Watch the demo](https://youtu.be/K1JsvnzujOw)** for a better viewing experience.

> GPL-3.0 fork of Deaquay/ComfyUI-Qwen3.5-Uncensored, itself derived from huchukato/ComfyUI-QwenVL-Mod and 1038lab/ComfyUI-QwenVL. See [LICENSE](LICENSE).

![ThinkingLLM Demo](docs/DemoThinkingLLM%20(1).gif)
![Node screenshot](docs/Screenshot%20of%20Notes.png)

> 🎥 Prefer a narrated version? **[Watch the demo on YouTube](https://youtu.be/K1JsvnzujOw)**

---

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

---

## Nodes at a glance

| Node | Path | Purpose |
| --- | --- | --- |
| `ThinkingLLM` | Transformers / HF (recommended) | Vision/video analysis with text output |
| `ThinkingLLM (Advanced)` | Transformers / HF | Vision/video + full sampling controls, `MASK_PREVIEW` |
| `ThinkingLLM Prompt Enhancer` | Transformers / HF | Text prompt enhancement |
| `ThinkingLLM (GGUF)` | GGUF / llama.cpp | Vision/video via GGUF models |
| `ThinkingLLM (GGUF Advanced)` | GGUF / llama.cpp | GGUF vision/video + advanced controls, `MASK_PREVIEW` |
| `ThinkingLLM Gemma 4 Audio (GGUF)` | GGUF / llama.cpp | Audio understanding (Gemma 4) |
| `ThinkingLLM Whisper ASR` | ASR | Speech-to-text transcription |
| `ThinkingLLM Prompt Enhancer (GGUF)` | GGUF / llama.cpp | Text prompt enhancement via GGUF |
| `ThinkingLLM System Prompt Preset` | Utility | Reusable system-prompt presets as `STRING` |
| `ThinkingLLM API (OpenAI Compatible)` | ThinkingLLM/API | Secure calls to OpenAI-compatible remote providers |

> **Need a remote/cloud model?** The **API node** calls approved OpenAI-compatible backends (OpenRouter, OrcaRouter, OpenAI, QwenCloud, Groq, …) with credentials kept server-side. See [docs/OPENAI_COMPATIBLE_API.md](docs/OPENAI_COMPATIBLE_API.md).

---

## Feature overview

### Model guidance box

ThinkingLLM nodes add a read-only `recommended_settings` info box.

- It explains the selected model family and recommended generation settings.
- For every LTX 2.3, MiniMax H3, and Wan 2.2 video preset, it adds a preset-specific `enable_thinking` and `max_tokens` recommendation with a short reason.
- Model sampling guidance and video-preset output budgets stay separate, so selecting a preset never replaces the selected model family's sampler advice.
- It does not change your saved widget values.
- For GGUF audio-capable nodes, it tells you whether the selected model actually supports audio.
- In Advanced GGUF nodes, if you connect or expose audio for a model without known audio support, the box warns you instead of silently implying that audio will work.

See [video preset settings](docs/VIDEO_PRESET_SETTINGS.md) for the complete recommendation table and source rationale.

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

The four ThinkingLLM vision nodes (`ThinkingLLM`, `ThinkingLLM (Advanced)`, `ThinkingLLM (GGUF)`, `ThinkingLLM (GGUF Advanced)`) accept an optional ComfyUI `MASK` input. White mask pixels select the area to analyze; black pixels preserve surrounding context, and soft mask values create a gradual transition. ThinkingLLM keeps the selected area at normal brightness while dimming everything outside it, helping the vision model focus on the mask without losing the scene's geometry, lighting, or background context.

**Object-removal workflow**
1. Connect the source image to `image` and the mask to `mask`.
2. Select `👁️ Remove (Mask Area)` from `preset_prompt`.
3. Optionally describe the desired replacement in `custom_prompt` (e.g. `continue the brick wall` or `fill with matching wooden floor`).
4. Connect `RESPONSE` to the positive prompt of your diffusion inpainting workflow.
5. Send the original image and mask separately to the diffusion model's inpainting inputs.

ThinkingLLM does not edit the image itself — it examines the masked area and surrounding scene, then generates a positive description of what should appear inside the mask. The downstream diffusion model performs the actual replacement.

**Previewing the LLM mask view**

`ThinkingLLM (Advanced)` and `ThinkingLLM (GGUF Advanced)` return a third output `MASK_PREVIEW` — connect it to a ComfyUI **Preview Image** node to inspect the exact image used for analysis. The GGUF nodes also expose `mask_mode`: `focus` keeps the selected area at full brightness and dims the surroundings to 20%, while `reconstruct` conceals the selected area and asks the model to infer a natural continuation from the visible surroundings. With no mask connected, the preview remains the original image.

Without a custom instruction, the preset infers the most natural background continuation — matching surfaces, perspective, materials, texture, color, lighting, shadows, reflections, and image style, while avoiding phrases such as "remove the object" or "erase this area."

Example output:

> Continuous warm beige plaster wall with subtle uneven texture, matching the existing perspective, soft window illumination, natural tonal variation, and the floor-edge shadows.

**Notes and limitations**

- Masks apply to still images only, not video input.
- Mask dimensions are resized automatically to match the image.
- An empty mask or a mask without an image produces a clear error.
- Image batches continue to use the first image and first mask.
- Precise masks with a small amount of surrounding context generally produce the best reconstruction prompt.

## Nodes

### Transformers / HF nodes

This is the recommended default path: `ThinkingLLM`, `ThinkingLLM (Advanced)`, `ThinkingLLM Prompt Enhancer`. The HF path is the most straightforward option for Windows and Linux when you want the fewest install complications.

HF tokenizer note: some Gemma and Qwen variants need `sentencepiece` or `tiktoken` in the ComfyUI Python environment. ThinkingLLM declares those dependencies and will attempt to install them automatically if a tokenizer backend is missing at first load.

#### Shared ComfyUI text encoders

Compatible single-file checkpoints already in `ComfyUI/models/text_encoders` appear as `[local] filename (text_encoders)` in every applicable model selector. ThinkingLLM loads them through ComfyUI, so it does not download or duplicate a Transformers snapshot. Older workflows saved with the former `[ComfyUI]` label remain compatible.

- Qwen3-VL 4B and 8B `.safetensors`, including `qwen3vl_8b_fp8_scaled.safetensors`
- Gemma 3 12B `.safetensors`, including `gemma-3-12b-it-heretic-v2_int8.safetensors`

The MiniMax H3/Qwen3-VL 32B conditioning checkpoint is intentionally excluded because it is not a complete text-generation model. The valid SafeTensors extension is `.safetensors` (plural). ComfyUI-native generation supports one-beam sampling; terminal output is printed after generation rather than streamed token by token.

### GGUF / llama.cpp nodes

- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`
- `ThinkingLLM Gemma 4 Audio (GGUF)`
- `ThinkingLLM Prompt Enhancer (GGUF)`

The GGUF path requires a multimodal-capable `llama-cpp-python` build. The normal PyPI package may not include the chat handlers needed for Qwen vision, Gemma 4, or MTMD audio. On Linux, ThinkingLLM auto-checks this at first GGUF use and attempts an automatic install of a matching JamePeng backend. Use the setup notes in [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md) if you need to override the wheel/package source.

Model labels are consistent across the standard, Advanced, audio, and Prompt Enhancer selectors: `[installed]` marks a catalog model found in any configured `LLM`, `GGUF`, or `Qwen-VL` location, while `[local]` marks an uncatalogued local model. These labels are display-only and do not change saved model identities.

### API node (OpenAI Compatible)

`ThinkingLLM API (OpenAI Compatible)` calls approved OpenAI-compatible backends (OpenRouter, OrcaRouter, OpenAI, QwenCloud, Groq, Together, Fireworks, DeepInfra, Featherless) with credentials and endpoints bound server-side. `model_name` offers the full curated per-provider list; use `custom_model_name` for anything not listed.

- Provider keys live in **server/user environment variables**, never in the workflow, profile JSON, or Git.
- Full setup, security model, and per-provider key names: [docs/OPENAI_COMPATIBLE_API.md](docs/OPENAI_COMPATIBLE_API.md).
- Image input: [docs/OPENAI_COMPATIBLE_API_IMAGES.md](docs/OPENAI_COMPATIBLE_API_IMAGES.md).

## Workflow tips

- **Pre-process input images** — use a resize or scale node before ThinkingLLM so large images don't blow up the context window.
- **Display the output** — connect RESPONSE or ENHANCED_OUTPUT to a **Show Text** or **Show Anything** node to see the generated text in the UI.
- **Inspect reasoning** — connect the RAW_TRACE output to a second Show Text node to see what the model was thinking.
- **Use the dedicated audio nodes first** — Gemma 4 Audio is for audio understanding; Whisper ASR is for transcription. The Advanced GGUF audio input is intentionally more flexible, but the info box will warn you when the selected model has no curated audio support.

### Duration-aware video prompts

The four vision nodes and both Prompt Enhancer nodes expose an optional `duration_seconds` input. It becomes visible only for registered LTX 2.3 and MiniMax H3 video presets; all other presets ignore it. Connect the same requested duration value to ThinkingLLM and your downstream video generator so prompt timing and generated length stay aligned.

Switching from a duration-aware preset back to `Custom Only` or another non-video preset hides the control without changing its ComfyUI serialization slot. The last valid duration is preserved; if an older workflow contains a corrupted non-numeric value in that slot, the frontend repairs it to the last valid value or `5.0` seconds. After updating ThinkingLLM, restart ComfyUI and use `Ctrl+F5` once so the corrected frontend code is loaded.

- LTX 2.3 uses the requested value directly and returns one chronological paragraph with a natural total-duration phrase.
- MiniMax H3 mirrors ComfyUI's 24 fps `17k+5` frame grid. For example, `5.0` seconds becomes 124 frames (`5.17` seconds), `8.0` stays 192 frames (`8.00` seconds), and `12.0` becomes 294 frames (`12.25` seconds). The effective duration is internal planning context: T2VA, I2VA, and full-reference outputs do not get an invented `Target duration:` field. FL2VA and L2VA use it only in the official image-alignment line.
- The duration input overrides conflicting duration wording in the free prompt for duration-aware presets. Older saved nodes that do not contain the optional input use `5.0` seconds.

See the [duration-aware video prompt guide](docs/VIDEO_PROMPT_DURATION.md) for node coverage, MiniMax frame normalization, Base and Full-Reference structures, cache behavior, best-effort normalization, and troubleshooting.

For LTX, use `📖 LTX 2.3 NSFW T2V Scene` in a text Prompt Enhancer, or choose the matching I2V, First/Last-to-Video, or Reference-to-Video preset in a vision node. The First/Last preset treats the first two connected pictures as exact boundary frames; the Reference preset treats connected media as reusable identity, style, prop, environment, motion, or camera anchors.

### MiniMax H3 prompt preset

Select the dedicated T2VA, I2VA, FL2VA, L2VA, or Reference-to-Video preset that matches the connected media. Vision nodes expose all five formats; text-only Prompt Enhancer nodes expose only T2VA. The preset instructs the LLM to use the official three-field Base or six-section Full-Reference structure. After generation, ThinkingLLM only applies safe best-effort cleanup: it removes obsolete `Target duration:` declarations, normalizes recognizable cut timestamps, separates uniquely identifiable fields, and converts clear `<d>English ...</d>` variants to `<d>[English] ...</d>`. It does not reject the result when fields, speaker IDs, dialogue length, or timing remain imperfect.

Standalone quoted dialogue blocks are presented to the LLM as verbatim source material. When the input is a longer master script, the preset asks the model to select one coherent moment that fits instead of compressing the complete story. The selected moment may be visual-only. This is prompt guidance, not a blocking post-generation validator; the downstream MiniMax pipeline receives the cleaned model output even if the model deviates slightly.

For example, a 99-word multi-scene script with `duration_seconds=8.0` is valid input. ThinkingLLM selects one feasible short scene, with or without dialogue; it does not require all 99 dialogue words to play in eight seconds and does not compress the full narrative into rapid cuts.

The effective duration remains authoritative planning context for the LLM. ThinkingLLM does not launch a schema-repair generation or block output because of an out-of-range timestamp, missing speaker tag, unexpected field, or estimated dialogue length; MiniMax performs its own downstream prompt interpretation. The only automatic retry is a direct finalization retry when generation ends before any usable final prompt, such as inside an unfinished `<think>` block.

Reference-to-Video requires a custom prompt that describes the target video and assigns each reference a role such as identity, style, motion, camera, or voice. To animate one image as the first frame, use Text-to-Video instead. Start with `enable_thinking=false` and `max_tokens=2048` for Base formats or `max_tokens=3072` for Reference-to-Video. If a model still spends the initial budget inside an unfinished `<think>` block, ThinkingLLM discards that incomplete reasoning and automatically performs a direct schema-prefilled retry instead of forwarding it as the video prompt.

See the official [MiniMax H3 ComfyUI workflows](https://docs.comfy.org/tutorials/video/minimax/minimax-h3), [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), and [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

## Supported models

ThinkingLLM ships pre-configured entries in `hf_models.json` and `gguf_models.json`. Local GGUF models in `models/LLM/GGUF/` with a matching mmproj file are auto-discovered.

Commercial-release dropdowns group entries by rights status and accept only reviewed, locked models. Community dropdowns keep normal model names and download behavior; gated Hugging Face repositories still require the model owner's access terms and a read token.

### Qwen — Vision-language
- **HF:** Qwen3.5 4B/9B · Qwen3-VL 4B/8B (Instruct, abliterated, unredacted) · Qwen3-VL 32B Heretic · Qwen2.5-VL 3B/7B
- **GGUF:** Qwen3.8 27B Blackfrost Abliterated (Q2–Q8, image/video) · Qwen3.5 4B/9B/27B (Q4 K M–BF16) · Qwen3-VL 4B/8B/32B (Instruct, abliterated, thinking) · Qwen2.5-VL 3B/7B

### Qwen — Text-only
- **HF:** Qwen3 0.6B/4B/8B · Qwen3.5 4B/9B (base + heretic + uncensored)
- **GGUF:** Qwen3.8 27B Blackfrost Abliterated (Q2–Q8) · Qwen3 4B/8B (abliterated, Josiefied, base) · Qwen3.5 4B/9B/27B (base + uncensored HauhauCS)

### Gemma 4 — Vision-language
- **HF:** Gemma-4-E2B/E4B-it/Uncensored · Gemma-4-12B · Gemma-4-26B-A4B-it/Heretic · Gemma-4-31B-it
- **GGUF:** Gemma-4-E2B/E4B (Q4 K M–BF16, uncensored) · Gemma-4-12B-it (Q4/Q5/Q6/Q8/BF16/UD-Q4) · Gemma-4-26B-A4B-it · Gemma-4-31B-it (Q4 K M–BF16)

### Gemma 4 — Text-only (HF)
- Gemma-4-E2B-it / Uncensored · Gemma-4-E4B-it / Uncensored · Gemma-4-12B · Gemma-4-26B-A4B-it / Heretic · Gemma-4-31B-it

### Gemma 4 — Audio (GGUF)
- Gemma-4-E2B / E4B when present in the GGUF multimodal catalog · Gemma-4-12B-it via `unsloth/gemma-4-12b-it-GGUF`

Use `ThinkingLLM Gemma 4 Audio (GGUF)` for the clean audio-only interface. Audio depends on a recent multimodal `llama-cpp-python`/`Gemma4ChatHandler` build — check `RAW_TRACE` if audio behaves unexpectedly.

### Whisper — ASR
tiny · base · small · medium · large-v3 · distil-large-v3

Use `ThinkingLLM Whisper ASR` with a connected ComfyUI `AUDIO` input or an `audio_file_path` (M4A/MP3/WAV/FLAC/FFmpeg-readable). The node uses `faster-whisper`; first use may download the model into the Hugging Face cache. Default `small` on CPU/int8 is reliable on Windows; switch to CUDA once the local CTranslate2 CUDA runtime is confirmed working.

## Model locations

Models are discovered automatically.

- HF models: `ComfyUI/models/LLM/Qwen-VL/`
- GGUF models: `ComfyUI/models/LLM/GGUF/`

For local GGUF models, keep the matching mmproj file beside the model file.

## Platform notes

- **HF is the recommended default path** and the simplest cross-platform option.
- **Shared ComfyUI text encoders**: compatible single-file checkpoints in `ComfyUI/models/text_encoders` appear as `[local] filename (text_encoders)` in every applicable model selector and load through ComfyUI (no duplicate Transformers snapshot). Qwen3-VL 4B/8B and Gemma 3 12B `.safetensors` are supported. The MiniMax H3/Qwen3-VL 32B conditioning checkpoint is intentionally excluded (not a complete text-generation model). Valid extension is `.safetensors`; native generation uses one-beam sampling and prints terminal output after generation.
- **GGUF on Windows** needs a matching `win_amd64` vision-capable wheel.
- **GGUF on Linux** auto-installs a matching backend on first use when possible. Override with `THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL` or `THINKINGLLM_LLAMA_CPP_LINUX_SPEC` when your server needs a different build.
- **GGUF audio** needs an audio-capable multimodal backend. `Gemma4ChatHandler` is used for Gemma 4.
- **Whisper ASR** needs `faster-whisper` and FFmpeg. The node returns an install hint if `faster-whisper` is missing.
- **Flash Attention** support is best on Linux. The nodes fall back when it is unavailable.
- **The thinking toggle** works best with Qwen3.5 and Qwen3 style models. Other architectures may ignore the steering.

## Production deployment

This repository provides ComfyUI nodes, not a complete production runtime. If you plan to expose a workflow through an API on RunPod Serverless, review the dependency, validation, testing, and rollback requirements in the [ComfyUI on RunPod production-readiness checklist](https://comfyrail.dev/production-readiness).

## FAQ

**Why does the node sometimes answer quickly and sometimes take longer?**

When thinking is enabled, the model may decide to reason before answering. For easy prompts it may skip explicit reasoning. That is expected behavior.

**How do I see what the model was thinking?**

Use the second `RAW_TRACE` output, or enable live terminal token streaming.

**How do I debug prompts that seem bad or stuck?**

Turn on `stream_tokens_to_terminal`. It is there specifically so you can see the generation live and catch bad prompt behavior sooner.

**Can I call remote/cloud models?**

Yes — the **ThinkingLLM API (OpenAI Compatible)** node calls approved OpenAI-compatible providers with credentials kept server-side. See [docs/OPENAI_COMPATIBLE_API.md](docs/OPENAI_COMPATIBLE_API.md).

## Documentation index

| Topic | Doc |
| --- | --- |
| OpenAI-compatible API node (setup, keys, security) | [docs/OPENAI_COMPATIBLE_API.md](docs/OPENAI_COMPATIBLE_API.md) |
| API node image inputs | [docs/OPENAI_COMPATIBLE_API_IMAGES.md](docs/OPENAI_COMPATIBLE_API_IMAGES.md) |
| GGUF / llama.cpp vision install (Windows + Linux) | [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md) |
| Video preset recommendation table | [docs/VIDEO_PRESET_SETTINGS.md](docs/VIDEO_PRESET_SETTINGS.md) |
| Duration-aware video prompts | [docs/VIDEO_PROMPT_DURATION.md](docs/VIDEO_PROMPT_DURATION.md) |
| Maintainer notes (fork lineage, publishability) | [docs/MAINTAINER_NOTES.md](docs/MAINTAINER_NOTES.md) |

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
