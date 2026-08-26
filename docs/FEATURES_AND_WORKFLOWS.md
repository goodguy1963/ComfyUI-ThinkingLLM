# Features and workflows

This guide explains how ThinkingLLM's user-facing controls affect inference and how to choose the right workflow. For installation and the short node list, start with the [README](../README.md).

## Model guidance box

ThinkingLLM nodes add a read-only `recommended_settings` box. It explains the selected model family and suggests generation settings without changing saved widget values.

For LTX 2.3, MiniMax H3, and Wan 2.2 presets, the box also provides preset-specific `enable_thinking` and `max_tokens` guidance. Model sampling recommendations and video output budgets remain separate so a video preset does not silently overwrite model-family advice.

Audio-capable GGUF nodes use the same box to distinguish models with curated audio support. The Advanced GGUF node warns when audio is connected to a model that is not known to support it.

The complete video recommendation table and its rationale are in [Video preset settings](VIDEO_PRESET_SETTINGS.md).

## Thinking mode

Every inference node exposes `enable_thinking`.

- When enabled, a compatible model may reason before answering. This can improve difficult responses but uses more time and output budget.
- A model may still answer directly when it considers the task simple.
- When disabled, ThinkingLLM applies the appropriate direct-answer steering for the selected model family, such as a `/no_think` directive where supported.
- For transcription, short classification, and concise prompt enhancement, begin with thinking disabled.

ThinkingLLM keeps incomplete or visible reasoning out of the cleaned primary response. When supported, reasoning remains available in `RAW_TRACE` for diagnostics.

## Live terminal streaming

Enable `stream_tokens_to_terminal` to display generation in the ComfyUI terminal as tokens arrive. This is useful when a run appears stuck, when you want to catch a poor prompt early, or when you need to distinguish model reasoning from slow loading.

Terminal streaming is diagnostic output. Downstream nodes should consume the normal response output instead.

## Response and raw trace

Inference nodes return a cleaned answer and a second string named `RAW_TRACE`.

- `RESPONSE` or `ENHANCED_OUTPUT` is intended for display and downstream workflow use.
- `RAW_TRACE` preserves the raw generation stream and backend diagnostics, including visible reasoning when the model emits it.

Connect `RAW_TRACE` to a separate text display when diagnosing empty output, unexpected thinking, backend selection, model loading, or generation formatting.

## Audio workflows

### Gemma 4 audio understanding

Use `ThinkingLLM Gemma 4 Audio (GGUF)` for audio questions, descriptions, classification, and transcription with a compatible Gemma 4 model.

The node accepts a ComfyUI `AUDIO` input or an `audio_file_path`. Audio is normalized to 16 kHz mono WAV before inference. Native Gemma 4 audio requires:

- multimodal `llama-cpp-python` 0.3.36 or newer;
- `Gemma4ChatHandler`;
- an mmproj that positively reports audio support.

ThinkingLLM verifies these conditions before sending audio to inference. Older or unknown native paths remain blocked because they can terminate the ComfyUI process instead of returning a Python exception. BF16 mmproj is recommended for Gemma 4 E2B/E4B audio.

### Whisper speech-to-text

Use `ThinkingLLM Whisper ASR` when the primary goal is dependable transcription. It accepts ComfyUI audio and common FFmpeg-readable paths such as M4A, MP3, WAV, and FLAC.

The node uses `faster-whisper`. The first run may download the selected model. The `small` model on CPU/int8 is a practical Windows starting point; use CUDA after confirming that the local CTranslate2 CUDA runtime works.

### Advanced GGUF audio input

The Advanced GGUF node exposes optional audio for expert workflows, but most Qwen and Qwen-VL models are not audio models. Prefer the dedicated nodes unless the selected model and mmproj explicitly support audio.

## Mask-focused image analysis

The standard and Advanced HF/GGUF vision nodes accept an optional ComfyUI `MASK` input.

- White pixels select the area of interest.
- Black pixels preserve surrounding context.
- Soft values create a gradual transition.
- Mask dimensions are resized to the image automatically.

In `focus` mode, ThinkingLLM keeps the selected area at normal brightness and dims the surroundings. This directs the vision model without removing spatial, lighting, or scene context.

In `reconstruct` mode, the selected area is concealed and the model is asked to infer a natural continuation from visible surroundings. This is useful for producing a positive inpainting prompt.

### Object-removal prompt workflow

1. Connect the original image and mask to a ThinkingLLM vision node.
2. Select `Remove (Mask Area)` from `preset_prompt`.
3. Optionally describe the desired replacement in `custom_prompt`.
4. Connect `RESPONSE` to the positive prompt of a diffusion inpainting workflow.
5. Send the original image and mask separately to the diffusion model's inpainting inputs.

ThinkingLLM does not edit the image. It describes what should naturally appear inside the masked region; the downstream diffusion model performs the replacement.

Advanced nodes return `MASK_PREVIEW`. Connect it to **Preview Image** to inspect exactly what the LLM receives.

Mask limitations:

- Masks apply to still images, not video.
- An empty mask or a mask without an image returns a clear error.
- Image batches use the first image and first mask.
- Precise masks with a modest amount of surrounding context generally produce the best reconstruction prompt.

## Video prompt workflows

ThinkingLLM provides preset-specific guidance for LTX 2.3, MiniMax H3, and Wan 2.2 workflows. The optional `duration_seconds` input appears only for registered duration-aware presets.

Use the preset that matches the downstream workflow and connected media. MiniMax H3 offers separate T2VA, I2VA, FL2VA, L2VA, and Reference-to-Video contracts; text-only Prompt Enhancer nodes expose T2VA.

Detailed duration resolution, MiniMax frame-grid normalization, output structures, long-script handling, best-effort cleanup, and recommended budgets are documented in [Duration-aware video prompts](VIDEO_PROMPT_DURATION.md).

## Shared model files and caching

ThinkingLLM reuses compatible model files discovered in configured ComfyUI locations. Fixed-seed local generations may reuse saved prompt output when all relevant inputs match. Audio-node results skip prompt persistence because the external audio content must not be mistaken for a reusable text-only result.

For supported file layouts and model labels, see [Models and backends](MODELS_AND_BACKENDS.md).
