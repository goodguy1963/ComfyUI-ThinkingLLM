# ThinkingLLM API (OpenAI Compatible)

`ThinkingLLM API (OpenAI Compatible)` calls remote `/chat/completions` endpoints while keeping the same two-output pattern used by the local ThinkingLLM nodes:

- `RESPONSE` — final assistant text
- `RAW_TRACE` — reasoning plus final text when the provider exposes a separate reasoning channel

The node supports provider presets for QwenCloud, OrcaRouter, and Featherless, plus a fully custom OpenAI-compatible endpoint.

## API keys

API keys are intentionally read from environment variables instead of a ComfyUI text widget. This avoids serializing a secret into workflow JSON files.

Default variables:

| Provider | Environment variable |
| --- | --- |
| QwenCloud | `DASHSCOPE_API_KEY` |
| OrcaRouter | `ORCAROUTER_API_KEY` |
| Featherless | `FEATHERLESS_API_KEY` |
| Custom | `OPENAI_API_KEY` |

You can override the environment-variable name with the optional `api_key_env` input.

### Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY="your-key"
python main.py
```

For a persistent user variable:

```powershell
setx DASHSCOPE_API_KEY "your-key"
```

Restart the terminal/ComfyUI process after using `setx`.

### Linux / macOS

```bash
export DASHSCOPE_API_KEY="your-key"
python main.py
```

## Provider presets

### QwenCloud

Default base URL:

```text
https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

The default model field is `qwen3.8-max-preview`. `enable_thinking` and `thinking_budget` are sent as Qwen-compatible request fields.

### OrcaRouter

Default base URL:

```text
https://api.orcarouter.ai/v1
```

Paste the exact OrcaRouter model ID into `model_name`. The node does not hard-code community model IDs because availability can change independently of ThinkingLLM.

### Featherless

Default base URL:

```text
https://api.featherless.ai/v1
```

Paste the exact Featherless model ID into `model_name`.

### Custom

Set `base_url` to any service that implements an OpenAI-compatible `/chat/completions` endpoint. Examples include self-hosted vLLM, llama.cpp server, LiteLLM, and compatible gateways.

## Provider-specific parameters

Use `extra_body_json` for parameters that are not part of the common OpenAI Chat Completions shape. The value must be a JSON object and is merged into the outgoing request.

Example:

```json
{
  "top_k": 40,
  "min_p": 0.05
}
```

The node always forces `stream: true` after merging the extra JSON. Streaming is used so reasoning and final-answer channels can be captured independently when the backend exposes them.

## Qwen3.8 27B / uncensored variants

Do not assume a community model name from another provider will work unchanged. When a Qwen3.8 27B uncensored/abliterated model becomes available through an OpenAI-compatible provider, select that provider and paste its exact published model ID into `model_name`.

No code change is required unless the provider uses a non-compatible protocol.
