"""
{
    "name": "ComfyUI-ThinkingLLM",
    "description": "A multimodal ComfyUI AI node with Qwen3.5, Qwen3-VL, Qwen2.5-VL, Qwen3, and Gemma 4 integrations. Features live thinking in the terminal to see what the LLM is doing in real time.",
    "author": "goodguy1963",
    "version": "2.4.0",
    "url": "https://github.com/goodguy1963/ComfyUI-ThinkingLLM",
    "category": "image"
}
"""

import importlib.util
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
COMMERCIAL_RELEASE = os.environ.get("THINKINGLLM_COMMERCIAL_RELEASE") == "1"
WEB_DIRECTORY = None if COMMERCIAL_RELEASE else "./web"
COMMERCIAL_MODULES = {
    "AILab_OutputCleaner",
    "AILab_QwenVL",
    "AILab_QwenVL_PromptEnhancer",
    "AILab_StreamDisplay",
}
COMMERCIAL_NODE_MODULES = {
    "story_split_node",
    "system_prompt_preset",
    "vram_cleanup",
}

LEGACY_NODE_REPLACEMENTS = {
    "AILab_QwenVL": "ThinkingLLM_QwenVL",
    "AILab_QwenVL_Advanced": "ThinkingLLM_QwenVL_Advanced",
    "AILab_QwenVL_GGUF": "ThinkingLLM_QwenVL_GGUF",
    "AILab_QwenVL_GGUF_Advanced": "ThinkingLLM_QwenVL_GGUF_Advanced",
    "AILab_QwenVL_PromptEnhancer": "ThinkingLLM_QwenVL_PromptEnhancer",
    "AILab_QwenVL_GGUF_PromptEnhancer": "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
}
if COMMERCIAL_RELEASE:
    LEGACY_NODE_REPLACEMENTS = {
        old: new for old, new in LEGACY_NODE_REPLACEMENTS.items() if "GGUF" not in old
    }

def load_modules_from_directory(directory):
    for file in os.listdir(directory):
        if file.endswith(".py"):
            file_path = os.path.join(directory, file)
            module_name = os.path.basename(file)[:-3]
            if module_name == os.path.basename(__file__)[:-3]:
                continue
            if COMMERCIAL_RELEASE:
                allowed = (
                    COMMERCIAL_MODULES
                    if os.path.abspath(directory) == os.path.abspath(current_dir)
                    else COMMERCIAL_NODE_MODULES
                )
                if module_name not in allowed:
                    continue

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to create import spec for {module_name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")

load_modules_from_directory(current_dir)

# Also load from nodes subdirectory
nodes_dir = os.path.join(current_dir, "nodes")
if os.path.exists(nodes_dir):
    load_modules_from_directory(nodes_dir)

for legacy_node_name in LEGACY_NODE_REPLACEMENTS:
    NODE_CLASS_MAPPINGS.pop(legacy_node_name, None)
    NODE_DISPLAY_NAME_MAPPINGS.pop(legacy_node_name, None)


def _register_legacy_node_replacements():
    try:
        from comfy_api.latest._io import NodeReplace
        from server import PromptServer
    except Exception:
        return

    manager = getattr(getattr(PromptServer, "instance", None), "node_replace_manager", None)
    if manager is None:
        return

    for old_node_id, new_node_id in LEGACY_NODE_REPLACEMENTS.items():
        manager.register(NodeReplace(new_node_id=new_node_id, old_node_id=old_node_id))


_register_legacy_node_replacements()

NODE_CLASS_MAPPINGS = dict(sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0])))
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: x[1]))

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "LEGACY_NODE_REPLACEMENTS",
]
