# Duration-aware video prompts

ThinkingLLM can make LTX 2.3 and MiniMax H3 prompts match the duration configured in the downstream video generator. The duration affects prompt planning and best-effort formatting only; ThinkingLLM does not change the generator's own frame or length setting.

## Supported nodes

The optional `duration_seconds` input is available on:

- `ThinkingLLM`
- `ThinkingLLM (Advanced)`
- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`
- `ThinkingLLM Prompt Enhancer`
- `ThinkingLLM Prompt Enhancer (GGUF)`

The widget is shown only for registered LTX 2.3 and MiniMax H3 video presets. It remains hidden and has no effect for image, analysis, tagging, and other non-video presets. Older workflows without the input use `5.0` seconds for a duration-aware preset.

Connect the same requested duration to ThinkingLLM and the downstream video generator. No separate enable switch is required.

## Duration resolution

LTX uses the requested value directly. Its final prompt contains the duration once as natural language, for example `an 8-second continuous shot`, and remains one chronological paragraph.

MiniMax uses the same 24 fps `17k+5` frame grid as the ComfyUI H3 node:

1. Convert the requested seconds to frames with `round(seconds × 24)`.
2. Use at least five frames.
3. Raise the result to the next valid `17k+5` frame count.
4. Divide the valid frame count by 24 to obtain the effective prompt duration.

Examples:

| Requested | Effective frames | Effective duration |
| ---: | ---: | ---: |
| 5.0 s | 124 | 5.17 s |
| 8.0 s | 192 | 8.00 s |
| 12.0 s | 294 | 12.25 s |

Two requested MiniMax values that resolve to the same effective duration share the same duration-aware cache identity. A change to a different effective duration invalidates the cached and saved prompt state.

## MiniMax H3 output contracts

### Base formats

T2VA, I2VA, FL2VA, and L2VA use the official base fields in this order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

The preset asks for `[Shot 1]` without a timestamp and later cuts as `[Shot N] At MM:SS.mmm`, with sequential shot numbers and timestamps below the effective duration. I2VA, FL2VA, and L2VA additionally request their official picture-alignment preamble. These are generation instructions rather than post-generation rejection rules.

The effective duration is internal planning context. T2VA and I2VA do not receive an invented `Target duration:` field. FL2VA and L2VA print the duration only where the official picture-alignment sentence requires it.

### Full-reference format

Reference-to-Video uses exactly these six sections in order:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

Use Reference-to-Video only when the request assigns connected assets a reusable role such as identity, wardrobe, prop, environment, style, motion, camera, or voice. To animate one image as the exact first frame, use the I2VA preset.

## Long scripts and short clips

A long multi-scene script is treated as source material, not as content that must all be compressed into one short generation. For an 8-second target, ThinkingLLM instructs the model to select one coherent contiguous moment that fits instead of turning the complete story into a rapid montage.

Standalone quoted paragraphs are recognized as source dialogue. The rules are:

- Scenes and dialogue outside the selected moment may be omitted.
- The selected moment may be visual-only even when the master script contains dialogue.
- Every selected dialogue block must match a source block verbatim and remain in source order.
- Dialogue cannot be translated, paraphrased, merged, shortened, or invented.
- Selected speech uses `<d>[Language] exact source text</d>` and a stable speaker ID such as `(S1)`. The square brackets are literal, so English is written as `<d>[English] Exact source text.</d>`, not `<d>English ...</d>`.
- `<cutoff>` is allowed only when the user explicitly requests truncated speech.

For example, a 99-word comedy script and an 8-second target do not fail before or after model generation. The preset asks the LLM to select one short exchange and its directly related action, or a visual-only beat. Any remaining mismatch is forwarded for MiniMax's downstream prompt processing.

## Best-effort normalization

After generation, ThinkingLLM deterministically normalizes only safe, unambiguous formatting differences:

- uniquely identifiable field boundaries;
- recognizable timestamp precision and `At` capitalization;
- removal of the redundant zero timestamp from `[Shot 1]`;
- the common `<d>English ...</d>` variant to `<d>[English] ...</d>` syntax;
- removal of an obsolete standalone `Target duration:` declaration.

ThinkingLLM does **not** enforce exact field count or order, speaker IDs, dialogue length or source fidelity, sequential or in-range timestamps, or the full reference-section schema. It does not run an automatic schema-repair pass for minor mismatches; those are forwarded to the downstream MiniMax generator, which performs its own prompt interpretation. It retries only when no usable final prompt exists, such as an unfinished `<think>` block. The system prompt still strongly requests the official format without turning ThinkingLLM into a blocking schema gateway.

## Recommended settings

- The read-only info box shows the active video preset's recommendation without changing node values. See the [complete video preset settings table](VIDEO_PRESET_SETTINGS.md).
- Connect the same requested duration to ThinkingLLM and the H3 generator.
- Start with `enable_thinking=false` for direct prompt conversion.
- Start with `max_tokens=2048` for Base/T2VA, I2VA, FL2VA, and L2VA prompts.
- Start with `max_tokens=3072` for full-reference prompts.
- If a video-preset response ends inside an unfinished `<think>` block, ThinkingLLM discards that reasoning and automatically retries with thinking disabled and the native output-field prefill. It never forwards the incomplete reasoning as the main prompt.
- Increase the token limit only when a useful response is genuinely truncated.
- Inspect `RAW_TRACE` for model reasoning and connect the main output to Show Text or Show Anything to inspect the exact downstream string.

After updating the custom node, restart ComfyUI completely and refresh the browser with `Ctrl+F5` so Python node definitions and frontend widget visibility are both reloaded.

## Official references

- [ComfyUI MiniMax H3 workflow guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [MiniMax H3 Base prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 Full-Reference prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [LTX-Video repository](https://github.com/Lightricks/LTX-Video)
