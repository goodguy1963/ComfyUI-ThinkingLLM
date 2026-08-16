# ThinkingLLM API (OpenAI Compatible)

`ThinkingLLM API (OpenAI Compatible)` calls server-approved `/chat/completions` endpoints while keeping the same two-output pattern used by the local ThinkingLLM nodes:

- `RESPONSE` — final assistant text with visible `<think>...</think>` blocks removed
- `RAW_TRACE` — reasoning plus final text when the provider exposes a reasoning channel or literal think blocks

The node is designed so ComfyUI can safely be used as an API backend without putting cloud API keys into workflow JSON.

## Security model

Treat ComfyUI workflow/API input as untrusted. The workflow is therefore not allowed to provide:

- API keys
- bearer tokens
- environment-variable names
- arbitrary remote `base_url` values
- custom HTTP headers

Instead, the workflow only selects an `api_profile` alias. Each profile is resolved on the ComfyUI server and binds the provider endpoint and authentication policy together.

This prevents a workflow from combining a sensitive environment variable with an attacker-controlled URL and also removes arbitrary outbound URLs from prompt JSON.

The server-side profile file contains no secrets. Secrets remain in process environment variables.

## Built-in profiles

The node includes fixed profiles for:

- QwenCloud (Singapore)
- OpenRouter
- Together AI
- Fireworks AI
- DeepInfra
- Groq
- Featherless
- OrcaRouter
- Ollama on `127.0.0.1:11434`
- vLLM on `127.0.0.1:8000`
- llama.cpp on `127.0.0.1:8080`

Authenticated built-in profiles use a fixed HTTPS endpoint and a fixed server-side environment-variable name. Local profiles use `auth: none` and fixed loopback addresses.

## Environment variables

Built-in cloud profiles use:

| Profile | Environment variable |
| --- | --- |
| QwenCloud | `DASHSCOPE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Fireworks AI | `FIREWORKS_API_KEY` |
| DeepInfra | `DEEPINFRA_TOKEN` |
| Groq | `GROQ_API_KEY` |
| Featherless | `FEATHERLESS_API_KEY` |
| OrcaRouter | `ORCAROUTER_API_KEY` |

### Windows PowerShell

```powershell
$env:OPENROUTER_API_KEY="your-key"
python main.py
```

For a persistent user variable:

```powershell
setx OPENROUTER_API_KEY "your-key"
```

Restart the terminal/ComfyUI process after using `setx`.

### Linux / macOS

```bash
export OPENROUTER_API_KEY="your-key"
python main.py
```

## Server-side custom profiles

Copy `thinkingllm_api_profiles.example.json` to:

```text
thinkingllm_api_profiles.json
```

The real profile file is ignored by Git and must stay server-side.

A profile can bind:

- `provider` — display/provider label
- `base_url` — fixed server-approved API base URL
- `auth` — `bearer_env` or `none`
- `api_key_env` — environment-variable name; only valid with `bearer_env`
- `allowed_models` — optional exact model allowlist
- `max_tokens_limit` — optional server-side output-token ceiling
- `allow_extra_body` — whether workflow-supplied provider-specific JSON is accepted; defaults to `false`
- `send_seed` — whether the common `seed` value is sent; defaults to `false`
- `thinking_mode` — set to `qwen` for Qwen-compatible `enable_thinking` and `thinking_budget`

Example:

```json
{
  "profiles": {
    "OpenRouter Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": ["provider/model-id"],
      "max_tokens_limit": 8192,
      "allow_extra_body": false,
      "send_seed": false
    }
  }
}
```

Do not put a literal `api_key`, token, `Authorization` header, or arbitrary headers in this file. The loader rejects those fields.

Authenticated profiles must use HTTPS. If an internal authenticated endpoint only supports HTTP, put it behind TLS/reverse proxy before using it with this node.

## Production lock-down mode

For a public-facing or multi-user ComfyUI backend, the safest setup is to disable all convenience built-in profiles and explicitly define only the profiles the server is allowed to use:

```text
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
```

Then create only approved entries in `thinkingllm_api_profiles.json`, ideally with both `allowed_models` and `max_tokens_limit` configured.

You can keep the profile file outside the repository/container image by setting:

```text
THINKINGLLM_API_PROFILES_FILE=/secure/path/thinkingllm_api_profiles.json
```

## ComfyUI API usage

A client calling ComfyUI sends only the profile alias and normal generation inputs. For example, conceptually:

```json
{
  "api_profile": "OpenRouter Production",
  "model_name": "provider/model-id",
  "prompt": "Describe the scene",
  "max_tokens": 2048
}
```

The ComfyUI process resolves the profile, reads the configured environment variable, and adds the bearer header only when making the outbound provider request.

The client never receives or transmits the provider API key.

Important: this protects the provider secret, but it does not replace access control for ComfyUI itself. Anyone who is authorized to submit workflows using a configured profile can consume that profile's provider quota. Put authentication/network controls in front of a remotely accessible ComfyUI API and use profile model/token limits where appropriate.

## `extra_body_json`

`extra_body_json` is disabled by default for server-defined profiles. The server administrator must explicitly set:

```json
"allow_extra_body": true
```

before workflow-supplied provider-specific parameters are accepted.

Even when enabled, `extra_body_json` cannot override protected fields including:

- `model`
- `messages`
- `stream`
- `max_tokens`
- `max_completion_tokens`
- `temperature`
- `top_p`
- `seed`
- `enable_thinking`
- `thinking_budget`

This prevents a workflow from bypassing server-side model/token controls or changing the core request contract.

## Qwen thinking

The built-in QwenCloud profile uses Qwen-compatible thinking parameters. When `thinking_mode` is `qwen`, the node sends `enable_thinking` and, when greater than zero, `thinking_budget`.

For other providers, provider-specific reasoning controls should only be enabled through a server-approved profile with `allow_extra_body: true`.

## Qwen3.8 27B / community variants

The node deliberately does not hard-code a community model ID. Model availability and IDs can change independently at each provider. Use the exact provider-published model ID in `model_name`, or lock it down with `allowed_models` in the server-side profile.
