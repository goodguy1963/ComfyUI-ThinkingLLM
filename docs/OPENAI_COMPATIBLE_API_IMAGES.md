# ThinkingLLM API image inputs

`ThinkingLLM API (OpenAI Compatible)` supports a single image input in addition to text. Video input is intentionally not part of this node.

The image path follows the common OpenAI-style chat-completions multimodal message format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      }
    }
  ]
}
```

## Two mutually exclusive image inputs

The node exposes two optional inputs:

- `image` — a normal ComfyUI `IMAGE`. This is the **default/recommended path** for local images.
- `image_url` — an alternative public HTTPS URL for an image already hosted elsewhere.

Use one or the other. If both are supplied, the node rejects the request before contacting the provider.

There is no `video` or `video_url` input.

## Default path: ComfyUI IMAGE -> PNG -> Base64 data URL

When `image` is connected, ThinkingLLM:

1. accepts exactly one ComfyUI image (`[1,H,W,C]` or `[H,W,C]`);
2. validates dimensions and server-side pixel limits before conversion;
3. converts the image to 8-bit PNG in memory;
4. validates the PNG byte size;
5. Base64-encodes the PNG;
6. sends it as `data:image/png;base64,...` in an OpenAI-style `image_url` content item.

No temporary public file, upload service, or image-hosting credential is required.

The Base64 value exists only in the outbound provider request. It is not an API key and is not written to the server-side profile file.

### Why PNG

PNG avoids an additional lossy JPEG conversion and keeps the ComfyUI image content deterministic. The tradeoff is a larger request payload, which is why production profiles should set explicit image-size limits.

## Alternative path: public HTTPS `image_url`

When `image_url` is used, ThinkingLLM does **not** download the image itself. It validates the URL and passes it to the selected provider in the multimodal request.

Accepted URL form:

```text
https://example.com/path/image.png
```

Signed HTTPS URLs with query parameters are supported.

Rejected forms include:

```text
http://example.com/image.png
data:image/png;base64,...
file:///path/to/image.png
URLs containing embedded user:password credentials
URLs containing control characters
```

The URL alternative is intentionally restricted to public HTTPS URLs. Base64 data URLs must come from the local ComfyUI `IMAGE` path so the node can enforce local pixel/byte limits.

Because ThinkingLLM does not fetch `image_url`, the URL is not a server-side HTTP fetch performed by the ComfyUI host. The downstream provider decides whether it can retrieve and process the URL.

## Server-side profile policy

Custom production profiles are fail-closed for images. Image support is disabled unless the administrator explicitly enables it.

Example vision profile:

```json
{
  "profiles": {
    "OpenRouter Vision Production": {
      "provider": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "auth": "bearer_env",
      "api_key_env": "OPENROUTER_PRODUCTION_API_KEY",
      "allowed_models": [
        "provider/vision-model-id"
      ],
      "max_tokens_limit": 8192,
      "max_input_chars": 131072,
      "max_timeout_seconds": 300,
      "allow_images": true,
      "allow_image_url": true,
      "max_image_pixels": 16777216,
      "max_image_bytes": 20000000,
      "allowed_extra_body_fields": [],
      "send_seed": false
    }
  }
}
```

### `allow_images`

```json
"allow_images": true
```

Enables image input for the profile. Default for custom profiles is `false`.

If false, both local `IMAGE` and `image_url` requests are rejected before any provider request is sent.

### `allow_image_url`

```json
"allow_image_url": true
```

Allows the public HTTPS `image_url` alternative.

This option requires `allow_images: true`. A profile cannot enable URL images while general image input is disabled.

For production deployments that only need locally supplied ComfyUI images, use:

```json
"allow_images": true,
"allow_image_url": false
```

That is the tighter policy.

### `max_image_pixels`

```json
"max_image_pixels": 16777216
```

Maximum width x height accepted for a local ComfyUI `IMAGE` before PNG encoding.

The default is 16,777,216 pixels (4096 x 4096). Administrators can lower this substantially for API workloads.

This protects server RAM/CPU and limits accidentally huge multimodal requests.

### `max_image_bytes`

```json
"max_image_bytes": 20000000
```

Maximum PNG byte size before Base64 encoding. Default is 20,000,000 bytes.

Base64 adds approximately one third of transport overhead, so choose this limit with provider request-size limits and network costs in mind.

This limit applies to the locally encoded ComfyUI image. ThinkingLLM cannot know the remote byte size of an `image_url` without downloading it, which the node deliberately does not do.

## Built-in profiles

The convenience built-in profiles allow the standard image transport so a local/single-user installation can try vision-capable models without creating a custom profile.

That does **not** mean every model exposed by a provider supports images. Image capability is model-specific.

For public/multi-user deployments, disable convenience built-ins:

```text
THINKINGLLM_DISABLE_BUILTIN_API_PROFILES=1
```

Then create explicit image-enabled production profiles with an exact `allowed_models` list.

## What happens with a text-only model

A provider can expose both text-only and vision-capable models through the same OpenAI-compatible API.

ThinkingLLM therefore cannot infer image capability reliably from the provider name alone. It sends the standard multimodal request when the profile permits images.

If the provider/model rejects a request containing an image with a typical image/content validation response, the node raises an actionable ComfyUI error similar to:

```text
Image input was rejected by API profile 'OpenRouter Production' / model '...'.
Use a vision-capable model supported by that provider, or remove the IMAGE/image_url input.
```

The provider's detailed response remains available in the ComfyUI server log, subject to existing credential redaction.

## ComfyUI UI usage

Typical local-image workflow:

```text
Load Image
    |
    v
ThinkingLLM API (OpenAI Compatible)
    image = connected IMAGE
    prompt = Describe the scene in detail
    api_profile = OpenRouter Vision Production
    model_name = provider/vision-model-id
    |
    +--> RESPONSE
    +--> RAW_TRACE
```

Typical URL workflow:

```text
ThinkingLLM API (OpenAI Compatible)
    image = not connected
    image_url = https://example.com/image.jpg
    prompt = Describe the scene in detail
```

## ComfyUI API-format workflow

For a local ComfyUI image, the normal ComfyUI graph connection supplies the `IMAGE` tensor. The external application does not need to manually Base64-encode the image for the provider; ThinkingLLM performs that conversion after the upstream ComfyUI image-loading/input node has produced an `IMAGE`.

For the URL alternative, the API-format workflow may contain the public URL string:

```json
{
  "image_url": "https://example.com/image.jpg"
}
```

Provider credentials remain server-side exactly as for text-only requests.

## Current limitations

- One image per API-node execution.
- Image batches are rejected rather than silently using only the first frame.
- Video is intentionally unsupported by this node.
- Local image transport is PNG Base64 only.
- `image_url` is HTTPS only.
- Actual model vision support is determined by the provider/model, not by the provider profile name.
- Remote URL image size cannot be preflighted by ThinkingLLM because the ComfyUI host does not fetch the URL.

## Production checklist for image-enabled profiles

- [ ] Use an exact vision-capable model ID in `allowed_models`.
- [ ] Set `allow_images: true` only on profiles that need image input.
- [ ] Keep `allow_image_url: false` unless public URL input is actually needed.
- [ ] Set a realistic `max_image_pixels` for the application.
- [ ] Set a realistic `max_image_bytes` for provider/network limits.
- [ ] Keep the provider API key only in the ComfyUI host/runtime secret environment.
- [ ] Keep production profile configuration outside the Git checkout where practical.
- [ ] Test an intentionally text-only model once to verify the user-facing image rejection is understandable.
- [ ] Keep provider-side spend limits/alerts enabled where available.
