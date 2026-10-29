# Decision Router API mode

The Image and Text Decision Router nodes support both local in-process inference and native API inference. Local remains the default. API mode sends each item in a batch as a separate decision request and routes the original input item according to the provider's typed `choice` response.

## Supported native providers

| Profile | Protocol | Endpoint | Initial model selection | Image input |
| --- | --- | --- | --- | --- |
| OpenAI | `openai_decisions` | `https://api.openai.com/v1/decisions` | `gpt-6-luna` | Supported |
| OpenRouter | `openrouter_decisions` | `https://openrouter.ai/api/alpha/decisions` | Decision models returned by its Models API | Only models whose input modalities include `image` |

Other providers in the OpenAI-compatible chat node are not exposed as native Decision APIs by this feature. The router does not fall back to a chat completion that generates JSON.

## Credentials and profile setup

The built-in `OpenAI` and `OpenRouter` profiles use `OPENAI_API_KEY` and `OPENROUTER_API_KEY`. Set the key in the environment of the process that launches ComfyUI, then restart ComfyUI. Restart after adding or changing a custom profile as well, so the node's profile list is refreshed. The key and endpoint do not enter the workflow.

Custom profiles can opt into a native protocol using the server-side profile file:

```json
{
  "profiles": {
    "OpenRouter Decision Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "decision_protocol": "openrouter_decisions",
      "decision_endpoint": "https://openrouter.ai/api/alpha/decisions",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": ["cloudflare/clef-flash", "typesafe/jev-1.14"],
      "allow_images": true,
      "max_image_pixels": 16777216,
      "max_image_bytes": 20000000,
      "max_timeout_seconds": 300
    }
  }
}
```

`decision_protocol` and `decision_endpoint` must appear together. The endpoint is trusted server configuration, validated at profile load, and is not a node input. For shared ComfyUI servers, use an exact `allowed_models` list.

## Model and image capability handling

The node's API model list is filtered by decision output, required input modality, the profile's `allow_images` policy, and its optional `allowed_models` list. OpenRouter metadata is cached for up to one hour. If a stale cached list exists during a metadata outage, it can still be displayed; the selected model is checked against that list before inference. A model absent from the capability catalog is rejected rather than guessed to support images.

Image tensors are converted to bounded PNG data URLs using the selected server profile's `allow_images`, `max_image_pixels`, and `max_image_bytes` settings. OpenAI receives an `input_image` part; OpenRouter receives an `image_url` part in `state`. The profile's `max_input_chars` also bounds the system prompt, input text, and configured route names/descriptions. Provider-specific image limits can still apply and are reported as inference errors.

## Output and limits

The APIs return typed route selections and probabilities, which are normalized into the router's existing `decision_json` output. Provider confidence is included when returned. Route descriptions become the choice criteria, and the system prompt becomes the decision instruction. The existing 52-route cap and dynamic output sockets continue to apply.

Native Decisions APIs do not generate a rationale. The local-only `include_reason` control is hidden in API mode; if it remains enabled in a saved workflow, the result explicitly marks the reason as unavailable. `keep_model_loaded` has no effect on remote inference, and switching to API mode releases a local model previously retained by the node.
