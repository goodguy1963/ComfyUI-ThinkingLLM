# ThinkingLLM API (OpenAI Compatible)

`ThinkingLLM API (OpenAI Compatible)` lets a ComfyUI workflow call approved OpenAI-compatible chat-completions backends without exposing provider credentials to the workflow or to clients that submit ComfyUI API jobs.

The node returns:

- `RESPONSE` — the final assistant text, with literal `<think>...</think>` blocks removed.
- `RAW_TRACE` — the reasoning channel plus final text when the provider exposes reasoning separately, or the raw literal think block when a compatible backend emits one.

> [!IMPORTANT]
> **Provider API keys are host/runtime secrets, not ThinkingLLM configuration.** A real API key should exist only in the environment or secret store of the machine/service/container that launches ComfyUI. Never put a real provider key in a workflow, ComfyUI `/prompt` request, profile JSON, repository file, Dockerfile, README, issue, PR comment, or Git commit.

## Recommended production architecture

```text
External client / application
        |
        | ComfyUI API request
        | api_profile = "OpenRouter Production"
        | model_name  = "approved/model-id"
        | prompt      = "..."
        v
     ComfyUI
        |
        | loads trusted server-side profile
        | validates model, input size, output/thinking limits and timeout
        | reads provider credential from process environment
        v
OpenRouter / OpenAI / QwenCloud / other provider
```

The external client never sends or receives the provider key. The workflow only names a server-approved profile.

For a public or multi-user deployment, the recommended baseline is:

1. Keep provider keys in the host/service/container secret environment.
2. Keep the real profile file outside the Git checkout.
3. Set `THINKINGLLM_API_PROFILES_FILE` to that external file.
4. Set `THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1`.
5. Configure exact `allowed_models` values.
6. Configure `max_tokens_limit`, `max_input_chars`, and `max_timeout_seconds`; for Qwen thinking profiles also configure `max_thinking_budget`.
7. Keep `allowed_extra_body_fields` empty unless specific provider options are required.
8. Protect ComfyUI itself with authentication/network controls and rate/spending limits.

---

## Security boundaries

Treat workflow/API input as untrusted. A workflow is intentionally unable to provide:

- API keys or bearer tokens
- the name of the environment variable containing a key
- arbitrary remote `base_url` values
- custom HTTP headers
- a provider credential directly

Instead, the workflow provides only `api_profile`, `model_name`, prompts, and generation controls. The profile is read from trusted server configuration and binds the endpoint and authentication policy together.

This design prevents a workflow from combining an arbitrary server environment variable with an attacker-controlled URL.

### The profile file is not a secret, but it is security-sensitive

`thinkingllm_api_profiles.json` must contain configuration only, never a literal API key. Nevertheless, its **integrity matters**: a profile chooses the destination endpoint and the environment-variable reference. If an untrusted user can modify this file, they could redirect an approved credential to a different endpoint.

Therefore:

- only the ComfyUI administrator/service account should be able to modify the production profile file;
- use restrictive filesystem permissions where practical;
- for production, store the file outside the Git checkout;
- do not let workflow/API clients choose its path;
- restart ComfyUI after changing profile configuration.

### Redirects are deliberately rejected

The API transport does **not follow HTTP redirects**. This is intentional. Redirecting an authenticated request is dangerous because an authorization header can otherwise be forwarded beyond the originally approved endpoint.

Use the provider's canonical final API base URL in the server-side profile. A 3xx response is treated as an error and should be fixed by correcting the profile URL.

### HTTPS and local HTTP

Authenticated profiles (`auth: "bearer_env"`) must use HTTPS.

Unauthenticated plain HTTP is accepted automatically only for loopback hosts such as:

```text
http://127.0.0.1:8000/v1
http://localhost:11434/v1
```

For an intentionally cleartext LAN/internal endpoint with `auth: "none"`, the administrator must explicitly opt in:

```json
"allow_insecure_http": true
```

Do not enable this for Internet-facing endpoints.

---

## Built-in profiles

Convenience profiles are included for:

| Profile | Base URL | Server environment variable |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| QwenCloud (Singapore) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Together AI | `https://api.together.ai/v1` | `TOGETHER_API_KEY` |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
| DeepInfra | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_TOKEN` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Featherless | `https://api.featherless.ai/v1` | `FEATHERLESS_API_KEY` |
| OrcaRouter | `https://api.orcarouter.ai/v1` | `ORCAROUTER_API_KEY` |
| Ollama (local) | `http://127.0.0.1:11434/v1` | none |
| vLLM (local) | `http://127.0.0.1:8000/v1` | none |
| llama.cpp (local) | `http://127.0.0.1:8080/v1` | none |

The built-ins are convenient for local single-user use. For a remotely accessible/multi-user ComfyUI server, prefer explicit custom profiles and disable the built-ins.

### QwenCloud Singapore note

The built-in Singapore profile uses the shared `dashscope-intl` endpoint because a built-in profile cannot know your Alibaba Model Studio workspace ID. Alibaba currently keeps that shared endpoint functional, but recommends the workspace-dedicated Singapore domain for production:

```text
https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
```

For production QwenCloud use, create a custom server-side profile with your workspace-specific URL instead of relying on the convenience built-in.

---

## Environment variables and secret ownership

### Windows PowerShell — current process

```powershell
$env:OPENROUTER_PRODUCTION_API_KEY="your-secret-key"
$env:THINKINGLLM_DISABLE_BUILTIN_API_PROFILES="1"
$env:THINKINGLLM_API_PROFILES_FILE="D:\ComfyUI-Secrets\thinkingllm_api_profiles.json"
python main.py
```

The variable exists only for that PowerShell process and child processes.

### Windows — persistent service/user setup

If ComfyUI is launched by a service, task scheduler, wrapper, or another account, configure the secret in the environment of the **actual process that launches ComfyUI**. Setting a variable in your interactive terminal does not modify an already-running service.

After changing persistent environment values, restart ComfyUI.

### Linux/macOS shell

```bash
export OPENROUTER_PRODUCTION_API_KEY='your-secret-key'
export THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
export THINKINGLLM_API_PROFILES_FILE=/etc/thinkingllm/api_profiles.json
python main.py
```

### systemd/service deployments

Prefer a service-specific environment file or secret mechanism with restrictive permissions. Keep it outside the Git checkout. The unit should expose the key to the ComfyUI process without copying the key into the ThinkingLLM source tree.

### Containers

Do **not** bake provider credentials into the image:

```dockerfile
# Do not do this
ENV OPENROUTER_PRODUCTION_API_KEY=real-secret
```

Inject credentials at runtime through your container/orchestrator secret mechanism. Mount the profile file read-only and point `THINKINGLLM_API_PROFILES_FILE` at the mounted path.

---

## Server-side profile file

Start from `thinkingllm_api_profiles.example.json`, but keep your real production file outside the repository when practical.

Example:

```json
{
  "profiles": {
    "OpenRouter Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": [
        "provider/model-id"
      ],
      "max_tokens_limit": 8192,
      "max_input_chars": 131072,
      "max_timeout_seconds": 300,
      "allowed_extra_body_fields": [],
      "send_seed": false
    }
  }
}
```

The string `OPENROUTER_PRODUCTION_API_KEY` is only the **name** of an environment variable. The actual provider key must not appear in this JSON.

### Supported profile fields

| Field | Meaning |
| --- | --- |
| `provider` | Display/log label. |
| `base_url` | Fixed administrator-approved API base URL. |
| `auth` | `bearer_env` or `none`. |
| `api_key_env` | Environment-variable name used only with `bearer_env`. |
| `allowed_models` | Optional exact model allowlist. Recommended for production. |
| `max_tokens_limit` | Maximum output-token request accepted from a workflow. |
| `max_input_chars` | Maximum combined `system_prompt + prompt` character count. Default: 1,000,000. |
| `max_timeout_seconds` | Maximum timeout a workflow may request. Default: 300. |
| `max_thinking_budget` | Maximum Qwen `thinking_budget` accepted from a workflow. Default/max: 262,144; set lower for production. |
| `allowed_extra_body_fields` | Exact top-level provider-specific request fields allowed from `extra_body_json`. Empty means disabled. |
| `send_seed` | Whether to send the common `seed` parameter. Must be a JSON boolean. |
| `thinking_mode` | Currently empty or `qwen`. |
| `max_tokens_field` | `max_tokens` or `max_completion_tokens`. |
| `allow_insecure_http` | Explicit opt-in for unauthenticated non-loopback HTTP. |

Unknown profile fields are rejected instead of silently ignored. This is intentional: a typo such as `max_token_limit` must not quietly disable a security limit.

A custom profile with the same name as a built-in is treated as an override. If that override is invalid, the built-in with that name is removed rather than silently used as a fallback. This is fail-closed behavior: a broken restrictive `OpenRouter` override must not accidentally reactivate the permissive convenience built-in.

Literal secret/header fields such as `api_key`, `token`, `Authorization`, `headers`, or `secret` are rejected.

---

## Production lock-down mode

For public-facing or multi-user ComfyUI:

```text
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
```

Then expose only explicit entries in your trusted profile file.

Recommended location examples:

```text
Windows: D:\ComfyUI-Secrets\thinkingllm_api_profiles.json
Linux:   /etc/thinkingllm/api_profiles.json
```

Point ThinkingLLM to the external file:

```text
THINKINGLLM_API_PROFILES_FILE=/secure/path/thinkingllm_api_profiles.json
```

This also keeps credentials/profile policy independent of `git pull`, branch changes, reinstalls, or replacing the custom-node directory.

---

## ComfyUI UI usage

Add:

```text
ThinkingLLM/API
  ThinkingLLM API (OpenAI Compatible)
```

Typical values:

```text
api_profile: OpenRouter Production
model_name: provider/approved-model-id
system_prompt: You are a helpful assistant.
prompt: ...
max_tokens: 2048
temperature: 0.6
top_p: 0.95
stream_tokens_to_terminal: false
```

The node never asks for the real provider credential.

Terminal streaming defaults to **off** for the remote API node. When enabled, ThinkingLLM prints a display-only copy with C0/C1 control characters removed; the actual `RESPONSE`/`RAW_TRACE` strings remain unchanged.

---

## ComfyUI API usage

A client calling ComfyUI sends only profile/model/prompt/generation settings. Conceptually:

```json
{
  "api_profile": "OpenRouter Production",
  "model_name": "provider/approved-model-id",
  "system_prompt": "You are a helpful assistant.",
  "prompt": "Describe the scene",
  "max_tokens": 2048,
  "temperature": 0.6,
  "top_p": 0.95,
  "timeout_seconds": 120
}
```

It must not contain:

```text
api_key
bearer_token
api_key_env
base_url
Authorization
custom secret headers
```

### Remote calls always execute

ComfyUI normally caches a custom node when its inputs have not changed. That is wrong for a remote API call because the remote service is external state and can return a different answer even for identical inputs.

This node therefore declares itself changed on every queue run via `IS_CHANGED`, ensuring a repeated workflow actually performs a new provider request instead of reusing a previous response.

---

## Cost and abuse controls

Hiding a key is only one layer. Anyone authorized to submit a workflow that can select a cloud profile can potentially consume provider quota.

### Model allowlist

```json
"allowed_models": [
  "provider/approved-model-a",
  "provider/approved-model-b"
]
```

### Output-token ceiling

```json
"max_tokens_limit": 8192
```

### Input-size ceiling

```json
"max_input_chars": 131072
```

This limit is character based, not tokenizer based. It is a defensive bound against extremely large API payloads and uncontrolled input cost; provider/model context limits still apply separately.

### Timeout ceiling

```json
"max_timeout_seconds": 300
```

A workflow may request a shorter timeout but cannot exceed the server-defined ceiling.

### Qwen thinking-budget ceiling

For profiles using `"thinking_mode": "qwen"`, also cap the maximum thinking budget that a workflow may request:

```json
"max_thinking_budget": 8192
```

`thinking_budget` controls the maximum token budget available to Qwen's thinking process. The node enforces the profile ceiling before sending the provider request. A production deployment should choose a value appropriate to its model/cost policy rather than leaving the broad default maximum.

Also use provider-side budget caps/alerts and gateway/reverse-proxy rate limiting where available.

---

## Provider-specific request fields (`extra_body_json`)

The old boolean `allow_extra_body` switch is intentionally **not supported**. It was too broad: once enabled, a workflow could send arbitrary top-level provider parameters.

Instead, every extra top-level field must be individually approved:

```json
"allowed_extra_body_fields": [
  "reasoning"
]
```

Then the workflow may send, for example:

```json
{
  "reasoning": {
    "effort": "high"
  }
}
```

A field not on the allowlist is rejected.

The following core fields can never be delegated through `extra_body_json`:

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

`extra_body_json` also has a size limit and rejects non-finite JSON constants such as `NaN`.

---

## Thinking / reasoning traces

The node recognizes common OpenAI-compatible reasoning channels, including:

- `reasoning_content`
- `reasoning`
- `thinking`
- OpenRouter-style `reasoning_details` text/summary entries

For OpenRouter streaming, `reasoning_details` may appear in `choices[].delta.reasoning_details`; the node extracts readable text/summary portions for `RAW_TRACE`.

If a provider returns a mid-stream error inside an HTTP-200 SSE stream, the node detects a top-level `error` / `finish_reason: "error"` event and fails the ComfyUI execution rather than returning silently truncated output.

### Qwen thinking

Profiles with:

```json
"thinking_mode": "qwen"
```

send `enable_thinking` and, when greater than zero, `thinking_budget`. `max_thinking_budget` is the server-side ceiling on the workflow-requested thinking budget.

Example production policy:

```json
{
  "thinking_mode": "qwen",
  "max_thinking_budget": 8192
}
```

For other providers, expose only the specific provider reasoning field you need via `allowed_extra_body_fields`.

---

## Key rotation

Rotating a provider credential should require **zero workflow changes**:

1. Create a new provider key.
2. Replace the value of the existing server environment variable.
3. Restart/reload the ComfyUI process so it inherits the new value.
4. Test the configured profile.
5. Revoke the old provider key.

Workflows continue using the same alias:

```text
api_profile = OpenRouter Production
```

---

## Updating or reinstalling ThinkingLLM safely

Keep repository-managed source and host-managed secrets separate:

```text
Git repository                         Host configuration
-------------------------------        ---------------------------------
ThinkingLLM source code                provider API key environment variable
example profile                        production profile JSON
public documentation                   service/container secret settings

         git pull / reinstall                 remains unchanged
```

Before updating:

```bash
git status
git check-ignore -v thinkingllm_api_profiles.json
```

Do not use `git add -f` on secret/config files.

After updating, restart ComfyUI so it reloads current profiles and process environment. No provider key needs to be copied into the repository.

### Gitignore is a secondary safeguard, not a secret manager

The repository ignores `.env`, `.env.*`, and local `thinkingllm_api_profiles*.json` variants while explicitly keeping `thinkingllm_api_profiles.example.json` trackable.

Git ignore rules do not protect a file that was already committed. If a real provider key ever entered Git history, logs, an issue, or a PR comment, treat it as compromised and rotate/revoke it at the provider.

---

## Multiple environments/accounts

Use distinct aliases and environment-variable names:

```json
{
  "profiles": {
    "OpenRouter Development": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_DEV_API_KEY",
      "allowed_models": ["provider/dev-model"],
      "max_tokens_limit": 4096,
      "max_input_chars": 65536,
      "max_timeout_seconds": 120
    },
    "OpenRouter Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": ["provider/prod-model"],
      "max_tokens_limit": 8192,
      "max_input_chars": 131072,
      "max_timeout_seconds": 300
    }
  }
}
```

The actual keys remain separate host secrets.

---

## Troubleshooting

### `Unknown or unavailable server-side API profile`

Check:

- `THINKINGLLM_API_PROFILES_FILE` points to the intended file;
- the JSON parses correctly;
- the alias matches exactly;
- if built-ins are disabled, the profile exists in the custom file;
- no unknown/deprecated profile field caused the entry to be rejected.

Invalid profiles are ignored and the reason is printed in the ComfyUI server terminal. If a custom profile uses the same name as a built-in and the custom override is invalid, that built-in name is removed rather than used as a fallback.

### `API credential is not configured for the selected server-side profile`

The ComfyUI process cannot see the required environment variable. Configure it in the environment of the **actual process/service/container** that starts ComfyUI, then restart ComfyUI.

### Redirect rejected

The configured provider URL returned HTTP 3xx. Set `base_url` to the provider's canonical final API URL. Redirect following is intentionally disabled for credential safety.

### LAN HTTP profile is rejected

Plain HTTP is loopback-only by default. Prefer HTTPS. If an administrator intentionally uses an unauthenticated trusted-LAN endpoint, set `allow_insecure_http: true` in that server-side profile.

### Model rejected

The exact `model_name` is absent from `allowed_models`. Model IDs are provider-specific.

### Input rejected

The combined system/user prompt exceeds `max_input_chars`.

### Output limit rejected

The workflow's `max_tokens` exceeds `max_tokens_limit`.

### Thinking budget rejected

For a Qwen thinking profile, the workflow's `thinking_budget` exceeds `max_thinking_budget`.

### Timeout rejected

The workflow's `timeout_seconds` exceeds `max_timeout_seconds`.

### `extra_body_json` rejected

Each top-level field must appear in `allowed_extra_body_fields`. The old `allow_extra_body` flag is deliberately rejected.

### Works in a terminal but not as a service

Your interactive shell and service likely have different environments. Configure secrets in the service/container/runtime environment rather than relying on the login shell.

---

## Production security checklist

Before exposing a ThinkingLLM API profile through ComfyUI:

- [ ] The real provider key is absent from the Git working tree and Git history.
- [ ] The real provider key is absent from workflow/API JSON and profile JSON.
- [ ] Production profile configuration is stored outside the repository where practical.
- [ ] The production profile file is writable only by trusted administrators/service accounts.
- [ ] `THINKINGLLM_API_PROFILES_FILE` points to the intended trusted file.
- [ ] `THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1` is enabled for public/multi-user deployments.
- [ ] Every cloud profile uses HTTPS.
- [ ] Every cloud profile has an exact `allowed_models` list.
- [ ] Every cloud profile has a reasonable `max_tokens_limit`.
- [ ] Every cloud profile has a reasonable `max_input_chars`.
- [ ] Every cloud profile has a reasonable `max_timeout_seconds`.
- [ ] Every Qwen thinking profile has a reasonable `max_thinking_budget`.
- [ ] `allowed_extra_body_fields` is empty unless specific fields are needed.
- [ ] Redirects are not required by the configured provider URL.
- [ ] Terminal token streaming is disabled unless intentionally needed for debugging.
- [ ] ComfyUI itself is protected by authentication/network controls and rate limits.
- [ ] Provider-side spending limits/alerts are configured where available.
- [ ] Key rotation can be performed by changing only the host secret and restarting/reloading ComfyUI.

Following this model keeps provider credentials local to the ComfyUI runtime, constrains what an API client can spend/change, and keeps secret ownership independent from ThinkingLLM source-code updates.
