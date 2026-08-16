# ThinkingLLM API (OpenAI Compatible)

`ThinkingLLM API (OpenAI Compatible)` calls server-approved OpenAI-compatible `/chat/completions` endpoints while keeping the same two-output pattern used by the local ThinkingLLM nodes:

- `RESPONSE` — final assistant text with visible `<think>...</think>` blocks removed
- `RAW_TRACE` — reasoning plus final text when the provider exposes a reasoning channel or literal think blocks

The node is designed so ComfyUI can safely be used as an API backend **without putting cloud-provider API keys into workflow JSON, prompt JSON, saved workflows, node widgets, or the ThinkingLLM Git repository**.

> [!IMPORTANT]
> A provider API key is a server secret. Keep it on the machine/container that runs ComfyUI. ThinkingLLM only reads the key from the ComfyUI process environment at request time. Do not paste a real provider key into a workflow, profile JSON, README, issue, commit, Dockerfile, launch script that is tracked by Git, or any other repository file.

## TL;DR — recommended production setup

For a remotely accessible ComfyUI server, use this layout:

```text
External client / app
        |
        | ComfyUI API request
        | api_profile = "OpenRouter Production"
        | model_name  = "approved/model-id"
        | prompt      = "..."
        v
     ComfyUI
        |
        | resolves server-side profile
        | validates model + token limit
        | reads OPENROUTER_PRODUCTION_API_KEY from process environment
        v
   OpenRouter / provider
```

Recommended rules:

1. Keep the real API key only in the ComfyUI host/container environment or a host-level secret manager.
2. Keep the real `thinkingllm_api_profiles.json` outside the Git checkout for production.
3. Set `THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1` on public or multi-user deployments.
4. Point `THINKINGLLM_API_PROFILES_FILE` to the external profile file.
5. Use `allowed_models` and `max_tokens_limit` on every production cloud profile.
6. Leave `allow_extra_body` disabled unless a provider feature specifically requires it.
7. Protect the ComfyUI API itself with authentication/network controls and rate limiting.

With this arrangement, updating ThinkingLLM with `git pull`, switching branches, reinstalling the custom node, or replacing the repository directory does **not** require putting the provider secret into the repository. The secret remains owned by the host environment.

---

## Security model

Treat ComfyUI workflow/API input as untrusted. The workflow is therefore not allowed to provide:

- API keys
- bearer tokens
- environment-variable names
- arbitrary remote `base_url` values
- custom HTTP headers

Instead, the workflow only selects an `api_profile` alias. Each profile is resolved on the ComfyUI server and binds the provider endpoint and authentication policy together.

This prevents a workflow from combining a sensitive environment variable with an attacker-controlled URL and also removes arbitrary outbound URLs from prompt JSON.

The server-side profile file contains **configuration, not secrets**. Secrets remain in process environment variables.

### What is protected

The design is intended to prevent accidental or workflow-driven disclosure of provider credentials through:

- exported ComfyUI workflow JSON
- ComfyUI API `/prompt` payloads
- node widgets
- Git commits
- copied example workflows
- client-side applications that call ComfyUI
- arbitrary workflow-controlled outbound URLs

### What this does not replace

Protecting the provider key does not authenticate your ComfyUI installation. A user who is allowed to submit workflows using an approved profile can still consume that profile's provider quota.

For an Internet-facing deployment, also use appropriate controls around ComfyUI itself, for example:

- VPN/private network access
- authenticated reverse proxy
- API gateway
- IP allowlisting where appropriate
- rate limits / concurrency limits
- provider-side spending limits and alerts

---

## Secret ownership: the API key must stay local to the ComfyUI server

The provider key should have a simple lifecycle:

```text
Provider dashboard
      |
      | copy once during server setup / rotation
      v
Host secret store or process environment
      |
      | inherited by ComfyUI process
      v
ThinkingLLM API node
      |
      | Authorization: Bearer <secret>
      v
Provider
```

The key should **not** travel in the opposite direction. It must not be returned to the browser/client, serialized into a workflow, or stored in `thinkingllm_api_profiles.json`.

### Never put a real key in these places

Do not store a real API key in:

```text
ComfyUI workflow JSON
ComfyUI API request JSON
thinkingllm_api_profiles.json
thinkingllm_api_profiles.example.json
README.md or docs/
Python source files
Dockerfile
tracked docker-compose.yml
tracked startup scripts
GitHub issues or pull requests
screenshots / logs shared publicly
```

### Environment-variable names are not secrets

A profile may contain:

```json
"api_key_env": "OPENROUTER_PRODUCTION_API_KEY"
```

That is only the **name** of the server environment variable. The real key is stored separately in the environment:

```text
OPENROUTER_PRODUCTION_API_KEY=<real secret value>
```

ThinkingLLM resolves the name server-side and reads the value only when the node makes the provider request.

---

## Repository safety and `.gitignore`

The repository ignores the normal local secret files:

```text
.env
.env.*
thinkingllm_api_profiles.json
```

`.env.example` is allowed so documentation can show variable names without real values.

This is a safety net, not the primary secret-storage mechanism.

### Important Git rule

`.gitignore` only prevents **untracked** files from being added accidentally. It does not automatically protect a file that was already committed/tracked in the past.

Before storing anything sensitive inside the working tree, check that Git ignores it:

```bash
git check-ignore -v thinkingllm_api_profiles.json
```

For a production server, the safer approach is still to keep the profile file and secret material completely outside the repository.

### If a real key was ever committed

Deleting the line from the latest commit is not enough. Treat the key as compromised:

1. Revoke/rotate the key at the provider immediately.
2. Remove the secret from current repository content.
3. If required, clean it from Git history using an appropriate history-rewrite process.
4. Verify forks, CI logs, artifacts, backups, and deployment logs as appropriate.
5. Create a new key and place it only in the server-side secret environment.

Do not continue using a key that has appeared in Git history.

---

## Built-in profiles

The node includes fixed convenience profiles for:

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

Built-ins are convenient for a private desktop installation. For a public or multi-user API server, explicit custom profiles with allowlists are safer.

## Environment variables used by built-in cloud profiles

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

---

## Recommended production file layout

Keeping the profile file outside the Git checkout makes upgrades and reinstalls much safer.

### Windows example

```text
D:\ComfyUI\
  ComfyUI\
    custom_nodes\
      ComfyUI-ThinkingLLM\       <-- Git checkout, contains no secret

D:\ComfyUI-Secrets\
  thinkingllm_api_profiles.json   <-- server configuration, no literal API key
```

The real provider key is stored in the process/user/service environment, not in either file tree.

### Linux example

```text
/opt/comfyui/ComfyUI/custom_nodes/ComfyUI-ThinkingLLM/  <-- Git checkout
/etc/thinkingllm/api_profiles.json                      <-- root/admin-managed config
/etc/thinkingllm/provider-secrets.env                   <-- optional service env file, mode 600
```

If an environment file is used, it is loaded by your service manager or container runtime. ThinkingLLM itself does not need to parse a `.env` file.

---

## Server-side custom profiles

The repository includes:

```text
thinkingllm_api_profiles.example.json
```

For a simple local test, you may copy it to:

```text
thinkingllm_api_profiles.json
```

That real filename is ignored by Git.

For production, prefer an external location and set:

```text
THINKINGLLM_API_PROFILES_FILE=/secure/path/thinkingllm_api_profiles.json
```

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

Example production profile:

```json
{
  "profiles": {
    "OpenRouter Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": [
        "provider/approved-model-id"
      ],
      "max_tokens_limit": 8192,
      "allow_extra_body": false,
      "send_seed": false
    }
  }
}
```

There is deliberately no field containing the real key.

### Forbidden profile fields

Do not put a literal `api_key`, token, `Authorization` header, arbitrary headers, or secret field into the profile file. The profile loader rejects these secret/header fields.

Authenticated profiles must use HTTPS. If an internal authenticated endpoint only supports HTTP, put it behind TLS/reverse proxy before using it with this node.

---

## Production lock-down mode

For a public-facing or multi-user ComfyUI backend, disable all convenience built-in profiles and explicitly define only the profiles the server is allowed to use:

```text
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
```

Then create only approved entries in the external profile file, ideally with both `allowed_models` and `max_tokens_limit` configured.

Example production environment:

```text
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
THINKINGLLM_API_PROFILES_FILE=/etc/thinkingllm/api_profiles.json
OPENROUTER_PRODUCTION_API_KEY=<real secret value>
```

Only the first two values describe ThinkingLLM configuration. The third value is the actual provider secret and must remain private to the host/runtime.

---

## Windows setup

### Temporary PowerShell session

Useful for testing:

```powershell
$env:OPENROUTER_PRODUCTION_API_KEY="your-real-key"
$env:THINKINGLLM_DISABLE_BUILTIN_API_PROFILES="1"
$env:THINKINGLLM_API_PROFILES_FILE="D:\ComfyUI-Secrets\thinkingllm_api_profiles.json"

python main.py
```

The variables exist only in that PowerShell process and child processes. Closing the shell removes the temporary values.

### Persistent Windows environment variable

You can store a user-level variable with:

```powershell
setx OPENROUTER_PRODUCTION_API_KEY "your-real-key"
```

`setx` affects future processes. Restart the terminal/launcher/service before starting ComfyUI.

For a production service, prefer configuring the secret in the account/service environment or a dedicated Windows secret-management/deployment mechanism rather than placing it in a tracked `.bat`/`.ps1` file.

### Verify without printing the secret

Do not `echo` the real key into shared logs. You can check only whether it exists:

```powershell
if ($env:OPENROUTER_PRODUCTION_API_KEY) { "OpenRouter key is configured" } else { "OpenRouter key is missing" }
```

---

## Linux setup

### Interactive shell

```bash
export OPENROUTER_PRODUCTION_API_KEY='your-real-key'
export THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
export THINKINGLLM_API_PROFILES_FILE=/etc/thinkingllm/api_profiles.json

python main.py
```

### systemd-style deployment

A common production pattern is to keep the secret in a root/admin-managed environment file outside the repository, for example:

```text
/etc/thinkingllm/provider-secrets.env
```

Example content:

```text
OPENROUTER_PRODUCTION_API_KEY=<real secret value>
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
THINKINGLLM_API_PROFILES_FILE=/etc/thinkingllm/api_profiles.json
```

Restrict permissions appropriately, for example:

```bash
chmod 600 /etc/thinkingllm/provider-secrets.env
```

Then configure your service manager to load that environment file for the ComfyUI process. The exact service unit depends on how ComfyUI is installed.

### Verify without printing the secret

```bash
if [ -n "$OPENROUTER_PRODUCTION_API_KEY" ]; then
  echo "OpenRouter key is configured"
else
  echo "OpenRouter key is missing"
fi
```

---

## Container / Docker deployment

The same rule applies: inject the key at runtime; do not bake it into the image.

Avoid:

```dockerfile
ENV OPENROUTER_PRODUCTION_API_KEY=real-secret
```

A secret baked into an image can remain recoverable from image layers or deployment metadata.

Prefer runtime secret injection, for example an external environment/secret file managed outside the Git checkout, container-orchestrator secrets, or your hosting provider's secret/environment settings.

Conceptually:

```text
container runtime / secret manager
        |
        | injects OPENROUTER_PRODUCTION_API_KEY
        v
ComfyUI container process
        |
        v
ThinkingLLM API node
```

The profile JSON may be mounted read-only into the container and referenced with `THINKINGLLM_API_PROFILES_FILE`.

---

## ComfyUI UI usage

After ComfyUI starts with the desired profiles, add:

```text
ThinkingLLM/API
  ThinkingLLM API (OpenAI Compatible)
```

Select only the server-approved profile alias, for example:

```text
api_profile: OpenRouter Production
model_name: provider/approved-model-id
```

The node never asks for the real provider key.

If the key is missing, the client-facing error is intentionally generic. The ComfyUI server terminal contains the administrator-facing detail needed to identify which environment variable is missing.

---

## ComfyUI API usage

A client calling ComfyUI sends only the profile alias and normal generation inputs. Conceptually:

```json
{
  "api_profile": "OpenRouter Production",
  "model_name": "provider/approved-model-id",
  "system_prompt": "You are a helpful assistant.",
  "prompt": "Describe the scene",
  "max_tokens": 2048,
  "temperature": 0.6,
  "top_p": 0.95
}
```

The API request must **not** contain:

```text
api_key
bearer_token
api_key_env
base_url
Authorization
custom secret headers
```

The ComfyUI process resolves the profile, validates the request, reads the configured environment variable, and adds the bearer header only when making the outbound provider request.

The external client never receives or transmits the provider API key.

### Why this matters for generated API workflow JSON

ComfyUI API-format workflows are often:

- saved to disk
- generated programmatically
- logged during debugging
- copied between machines
- embedded in applications
- sent over queues or HTTP

Keeping the provider credential out of that JSON means all of those operations can occur without duplicating the provider secret.

---

## Model and spending controls

A hidden key is not enough. A user with permission to select a profile can still spend against that provider account.

Use an exact model allowlist:

```json
"allowed_models": [
  "provider/approved-model-a",
  "provider/approved-model-b"
]
```

Use a server-side token ceiling:

```json
"max_tokens_limit": 8192
```

If a workflow requests an unapproved model or a higher token limit, the node rejects the request before sending it to the provider.

Provider-side budget limits and alerts are also recommended where available.

---

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

For production, keep it `false` unless a specific provider feature requires it.

---

## Qwen thinking

The built-in QwenCloud profile uses Qwen-compatible thinking parameters. When `thinking_mode` is `qwen`, the node sends `enable_thinking` and, when greater than zero, `thinking_budget`.

For other providers, provider-specific reasoning controls should only be enabled through a server-approved profile with `allow_extra_body: true` where necessary.

---

## Qwen3.8 27B / community variants

The node deliberately does not hard-code a community model ID. Model availability and IDs can change independently at each provider.

Use the exact provider-published model ID in `model_name`, or lock it down with `allowed_models` in the server-side profile.

The provider key remains unchanged by model selection. The key identifies/authenticates the provider account; the `model_name` identifies the requested model.

---

## Updating ThinkingLLM without losing or committing secrets

This is one of the main reasons to keep secrets external.

### Safe update model

```text
Git repository                         Host configuration
-------------------------------        ---------------------------------
ComfyUI-ThinkingLLM source code        API key environment variable
example profile                        production profile JSON
public documentation                   service/container secret settings

         git pull / reinstall                 remains unchanged
```

A normal ThinkingLLM update should only replace/update repository-managed source files. It should not need to read, modify, migrate, or commit your real provider key.

### Best practice

Keep production secrets and profile configuration in a separate location such as:

```text
Windows: D:\ComfyUI-Secrets\
Linux:   /etc/thinkingllm/
```

Then point ThinkingLLM to the profile file with `THINKINGLLM_API_PROFILES_FILE`.

This avoids accidental deletion if the custom-node directory is removed and recloned.

### Before updating

You can verify that the Git working tree contains no secret configuration:

```bash
git status
```

If you intentionally use the ignored local `thinkingllm_api_profiles.json`, verify its ignore rule:

```bash
git check-ignore -v thinkingllm_api_profiles.json
```

Do not use `git add -f` on ignored secret/config files.

### After updating

Restart ComfyUI so it inherits the current environment and reloads profile configuration. There is no need to re-enter the provider key if the service/host environment still contains it.

---

## Key rotation

Rotating a provider key should not require any workflow changes.

1. Create a new provider key.
2. Replace the value of the existing server environment variable, for example `OPENROUTER_PRODUCTION_API_KEY`.
3. Restart/reload the ComfyUI process so it inherits the new value.
4. Test the configured profile.
5. Revoke the old provider key.

Because workflows reference only the profile alias, they continue to use:

```text
api_profile = OpenRouter Production
```

No workflow JSON needs to contain or learn the new secret.

---

## Adding another provider account or environment

Use separate environment-variable names and separate profile aliases.

Example:

```json
{
  "profiles": {
    "OpenRouter Development": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_DEV_API_KEY",
      "allowed_models": ["provider/dev-model"],
      "max_tokens_limit": 4096
    },
    "OpenRouter Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": ["provider/prod-model"],
      "max_tokens_limit": 8192
    }
  }
}
```

The two real keys remain separate host secrets.

---

## Troubleshooting

### `Unknown or unavailable server-side API profile`

Check:

- `THINKINGLLM_API_PROFILES_FILE` points to the correct file
- the JSON parses correctly
- the requested alias matches exactly
- if built-ins were disabled, the desired profile is present in the custom profile file

Invalid profiles are ignored and an explanation is printed to the ComfyUI server terminal.

### `API credential is not configured for the selected server-side profile`

The profile exists, but the ComfyUI process cannot see the required environment variable.

Check the environment of the **actual process that launches ComfyUI**. Setting a variable in one terminal does not automatically inject it into an already-running ComfyUI process or a service launched by another account.

Restart ComfyUI after changing persistent environment/service settings.

### Works in terminal but not when ComfyUI starts as a service

The interactive shell and the service likely have different environments. Configure the key in the service/container/runtime secret mechanism rather than relying on your login shell.

### Model rejected

Check the profile's exact `allowed_models` list. Model IDs are provider-specific and must match exactly.

### Token request rejected

The requested `max_tokens` is greater than the profile's server-side `max_tokens_limit`.

### Key changed but requests still use the old credential

Restart or reload the ComfyUI process. Environment variables are normally inherited when the process starts.

---

## Production security checklist

Before exposing a ThinkingLLM API profile through ComfyUI:

- [ ] The real provider key is not present anywhere in the Git working tree.
- [ ] The real provider key has never been committed to Git history.
- [ ] The real provider key is not present in workflow/API JSON.
- [ ] The real provider key is not present in `thinkingllm_api_profiles.json`.
- [ ] Production profile configuration is stored outside the repository where practical.
- [ ] `THINKINGLLM_API_PROFILES_FILE` points to the intended server-side file.
- [ ] `THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1` is enabled for public/multi-user deployments.
- [ ] Every cloud profile has an exact `allowed_models` list.
- [ ] Every cloud profile has a reasonable `max_tokens_limit`.
- [ ] `allow_extra_body` is `false` unless explicitly required.
- [ ] Authenticated remote profiles use HTTPS.
- [ ] ComfyUI itself is protected by appropriate authentication/network controls.
- [ ] Provider-side spending limits/alerts are configured where available.
- [ ] Secret files have restrictive filesystem permissions where applicable.
- [ ] Key rotation can be performed by changing only the server secret and restarting/reloading ComfyUI.

Following this model keeps provider credentials local to the ComfyUI runtime and independent from ThinkingLLM source-code updates.