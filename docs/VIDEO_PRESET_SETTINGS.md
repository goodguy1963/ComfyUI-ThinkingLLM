# Video preset settings

ThinkingLLM shows two independent recommendation blocks in the read-only `recommended_settings` info box:

1. The selected model family's sampler guidance.
2. The selected video preset's output budget and thinking-mode guidance.

The info box never changes saved widget values. Preset guidance is centrally defined in `AILab_System_Prompts.json` and mirrored to the web payload used by ComfyUI.

## Preset recommendations

| Preset family | `enable_thinking` | `max_tokens` |
| --- | ---: | ---: |
| LTX 2.3 T2V, I2V, First/Last, Reference | `false` | 512 |
| MiniMax H3 T2VA, I2VA, FL2VA, L2VA | `false` | 2048 |
| MiniMax H3 Full Reference | `false` | 3072 |
| Wan 2.2 Scene 3s/5s, T2V and I2V | `false` | 512 |
| Wan 2.2 Timeline 3s/5s, T2V and I2V | `false` | 768 |
| Wan 2.2 Scene 20s, T2V and I2V | `false` | 1536 |
| Wan 2.2 Timeline 20s, T2V and I2V | `false` | 2048 |

Thinking is disabled in these recommendations because the selected system prompts already define the planning and output format. Direct generation is faster and less likely to spend the output budget on a reasoning trace or deviate from the requested final structure.

The token budgets are upper bounds for the expected final format rather than settings from the downstream video generator. Increase the budget only when a useful final prompt is genuinely truncated.

## Sampling settings

Temperature, `top_p`, `top_k`, and repetition penalties remain model- and input-mode recommendations:

- Qwen3/Qwen3.5 vision input generally uses `temperature=0.7`, `top_p=0.8`, `top_k=20`, and `repetition_penalty=1.0`.
- Text-only input follows the selected model card. Qwen3.5 uses `temperature=1.0`, `top_p=1.0`, `top_k=20`, and `repetition_penalty=1.0`; Qwen3-VL Instruct uses `temperature=1.0`, `top_p=1.0`, `top_k=40`, and `repetition_penalty=1.0`.
- Widgets not exposed by the selected node are omitted from the info box summary.

For a connected reference video, begin around two sampled frames per source second and stay within the node's 64-frame limit. Image-only presets do not need a `frame_count` recommendation.

## Format rationale

- LTX requests one literal chronological paragraph under 200 words, so 512 output tokens provide sufficient headroom.
- MiniMax Base uses three structured fields; Full Reference uses six sections and needs a larger budget.
- Wan single-scene prompts fit within 512 tokens. Timeline formats and four-part 20-second sequences need progressively larger budgets to avoid truncation.

Runtime-setting instructions such as `max_tokens`, context length, batch size, and temperature are kept out of the system prompts themselves. They belong in the info box and must not become part of the generated video prompt.

## Official references

- [Qwen3-VL 4B Instruct generation hyperparameters](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen3.5 sampling recommendations](https://huggingface.co/Qwen/Qwen3.5-2B)
- [LTX-2 prompting guide](https://github.com/Lightricks/LTX-2/blob/main/README.md#%EF%B8%8F-prompting-for-ltx-2)
- [MiniMax H3 Base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 Full-Reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Wan 2.2 prompt extension](https://github.com/Wan-Video/Wan2.2#2-using-prompt-extention)
