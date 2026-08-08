"""Standalone system-prompt preset selector for ThinkingLLM."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "AILab_System_Prompts.json"
NO_PRESET_PROMPT = "🚫 No preset (image-only)"


def _load_system_prompt_presets() -> tuple[list[str], dict[str, str]]:
    """Load the same preset order and prompt strings used by ThinkingLLM."""
    try:
        with SYSTEM_PROMPTS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file) or {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ThinkingLLM System Prompt Preset] Failed to load presets: {exc}")
        return [NO_PRESET_PROMPT], {NO_PRESET_PROMPT: ""}

    raw_prompts = data.get("qwenvl") or {}
    system_prompts = raw_prompts if isinstance(raw_prompts, dict) else {}
    system_prompts = {
        str(name): str(prompt)
        for name, prompt in system_prompts.items()
        if isinstance(prompt, str)
    }
    system_prompts.setdefault(NO_PRESET_PROMPT, "")

    raw_presets = data.get("_preset_prompts") or []
    preset_prompts = [
        name
        for name in raw_presets
        if isinstance(name, str) and name in system_prompts
    ]
    if not preset_prompts:
        preset_prompts = list(system_prompts)
    if NO_PRESET_PROMPT not in preset_prompts:
        preset_prompts.insert(0, NO_PRESET_PROMPT)

    return preset_prompts, system_prompts


class ThinkingLLMSystemPromptPreset:
    """Select a ThinkingLLM preset and return its full system prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        preset_prompts, _ = _load_system_prompt_presets()
        return {
            "required": {
                "preset_prompt": (
                    preset_prompts,
                    {
                        "default": preset_prompts[0],
                        "tooltip": "Select a ThinkingLLM system-prompt preset.",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    OUTPUT_TOOLTIPS = ("The full system prompt text for the selected preset.",)
    FUNCTION = "get_system_prompt"
    CATEGORY = "ThinkingLLM/Utils"
    DESCRIPTION = "Outputs the full system prompt string for a ThinkingLLM preset."

    def get_system_prompt(self, preset_prompt: str):
        _, system_prompts = _load_system_prompt_presets()
        return (system_prompts.get(preset_prompt, ""),)


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_SystemPromptPreset": ThinkingLLMSystemPromptPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_SystemPromptPreset": "ThinkingLLM System Prompt Preset",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
