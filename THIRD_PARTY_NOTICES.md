# Third-party source and commercial release notice

ComfyUI-ThinkingLLM is distributed under GPL-3.0. Its README and source headers
credit this fork lineage:

- https://github.com/Deaquay/ComfyUI-Qwen3.5-Uncensored
- https://github.com/huchukato/ComfyUI-QwenVL-Mod
- https://github.com/1038lab/ComfyUI-QwenVL

The repository history begins with root commit
`0ca0d2448c6445a1addf87a55bde2308244380a2` and does not record the exact
upstream revisions used for the inherited files. The inherited source remains
under GPL-3.0; this notice does not replace its copyright notices.

The Comfy Rail commercial release pins an exact commit from this repository,
recorded in Comfy Rail's custom-node lock. Commercial release mode limits the
runtime to the locally installed, release-locked Apache-2.0 models below and
disables runtime downloads, runtime package installation, remote model code,
GGUF, and Whisper registration:

- `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`
- `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec70b05090aa6165b016eac8ec08e71b17`

The exact GPL-3.0 source and its modification notices must be provided together
with distributed commercial images.

Catalog status labels are operational safeguards, not legal advice or a model
license grant. External/gated models remain subject to their Hugging Face access
terms, and rights-unclear weights are neither shipped nor automatically fetched
by commercial Comfy Rail releases. Only catalog entries marked `cleared`, pinned by full revision and
component ID, are eligible for the locked commercial runtime.
