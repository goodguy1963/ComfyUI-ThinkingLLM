"""Lightweight regression tests for the local ThinkingLLM fork.

Run from the ComfyUI-ThinkingLLM folder:
    python tests/test_new_features.py

These tests intentionally avoid importing the model modules so they can run in a
plain shell without ML dependencies.
"""

import ast
import base64
import contextlib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WORKFLOW_DIR = PKG / "example_workflows"

STREAMING_FILES = {
    "AILab_QwenVL_GGUF_PromptEnhancer.py": {
        "stage": "INITIAL GENERATION",
        "label": "QwenVL GGUF",
    },
    "AILab_QwenVL_PromptEnhancer.py": {
        "stage": "HF TEXT GENERATION",
        "label": "QwenVL HF",
    },
    "AILab_QwenVL.py": {
        "stage": None,
        "label": "QwenVL HF",
    },
    "AILab_QwenVL_GGUF.py": {
        "stage": "STREAMING",
        "label": "QwenVL GGUF",
    },
}


def read_source(filename: str) -> str:
    return (PKG / filename).read_text(encoding="utf-8")


def parse_source(filename: str) -> ast.AST:
    return ast.parse(read_source(filename), filename=filename)


def build_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubQuantization:
    FP16 = types.SimpleNamespace(value="fp16")

    @classmethod
    def get_values(cls):
        return [cls.FP16.value]


def build_loader_test_stubs() -> dict[str, types.ModuleType]:
    pil_image = build_stub_module("PIL.Image", Image=object)
    pil_package = build_stub_module("PIL", Image=pil_image)
    comfy_model_management = build_stub_module(
        "comfy.model_management",
        throw_exception_if_processing_interrupted=lambda: None,
    )
    comfy_package = build_stub_module("comfy", model_management=comfy_model_management)
    output_clean_config = type("OutputCleanConfig", (), {"__init__": lambda self, *args, **kwargs: None})
    terminal_stream_display = type("TerminalStreamDisplay", (), {})
    stream_degeneration_error = type("StreamDegenerationError", (RuntimeError,), {})
    qwen_base = type("QwenVLBase", (), {"__init__": lambda self, *args, **kwargs: None})
    return {
        "numpy": build_stub_module("numpy", ndarray=object),
        "torch": build_stub_module(
            "torch",
            Tensor=object,
            dtype=object,
            float16=object,
            float32=object,
            device=type("device", (), {}),
            cuda=types.SimpleNamespace(device_count=lambda: 0, is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            no_grad=lambda func=None, *args, **kwargs: (lambda inner: inner) if func is None else func,
            inference_mode=lambda func=None, *args, **kwargs: (lambda inner: inner) if func is None else func,
        ),
        "huggingface_hub": build_stub_module(
            "huggingface_hub",
            hf_hub_download=lambda *args, **kwargs: "",
            snapshot_download=lambda *args, **kwargs: "",
        ),
        "PIL": pil_package,
        "PIL.Image": pil_image,
        "folder_paths": build_stub_module(
            "folder_paths",
            models_dir=str(PKG),
            folder_names_and_paths={},
            get_folder_paths=lambda *args, **kwargs: [],
        ),
        "AILab_StreamDisplay": build_stub_module(
            "AILab_StreamDisplay",
            StreamDegenerationError=stream_degeneration_error,
            StreamDegenerationGuard=type("StreamDegenerationGuard", (), {"__init__": lambda self, *args, **kwargs: None, "push": lambda self, text: None}),
            TerminalStreamDisplay=terminal_stream_display,
            extract_stream_token=lambda chunk: {"reasoning": "", "content": (chunk.get("choices") or [{}])[0].get("delta", {}).get("content", "")},
            strip_degenerate_repetition=lambda text, *args, **kwargs: text,
        ),
        "AILab_LlamaCppInstaller": build_stub_module(
            "AILab_LlamaCppInstaller",
            ensure_llama_cpp_backend=lambda *args, **kwargs: object(),
            format_llama_cpp_backend_info=lambda info=None: "stub backend",
            get_last_llama_cpp_backend_info=lambda: {"gpu_offload": None, "vision_handlers": ["Qwen3VLChatHandler"]},
            relax_windows_dll_directory_for_long_paths=contextlib.nullcontext,
        ),
        "AILab_OutputCleaner": build_stub_module(
            "AILab_OutputCleaner",
            OutputCleanConfig=output_clean_config,
            clean_model_output=lambda text, *args, **kwargs: text,
            prompt_output_guard=lambda text, *args, **kwargs: text,
        ),
        "transformers": build_stub_module(
            "transformers",
            AutoModelForCausalLM=object,
            AutoModelForVision2Seq=object,
            AutoModelForImageTextToText=object,
            AutoProcessor=object,
            AutoTokenizer=object,
            BitsAndBytesConfig=type("BitsAndBytesConfig", (), {}),
        ),
        "comfy": comfy_package,
        "comfy.model_management": comfy_model_management,
        "AILab_QwenVL": build_stub_module(
            "AILab_QwenVL",
            ATTENTION_MODES=["auto"],
            HF_ALL_MODELS={"stub-model": {}},
            HF_TEXT_MODELS={},
            HF_VL_MODELS={},
            NODE_PROMPT_STATE={},
            PROMPT_CACHE={},
            _make_node_state_key=lambda *args, **kwargs: "node-key",
            _build_workflow_fingerprint=lambda *args, **kwargs: "workflow-fingerprint",
            _default_model_from_config=lambda models, fallback: next(
                (
                    name
                    for name, info in models.items()
                    if isinstance(info, dict) and info.get("default")
                ),
                next(iter(models.keys()), fallback),
            ),
            apply_qwen_soft_thinking_directive=lambda *args, **kwargs: None,
            build_node_input_signature=lambda *args, **kwargs: {},
            download_hf_file_to_path=lambda *args, **kwargs: None,
            ensure_model=lambda *args, **kwargs: None,
            ensure_cuda_vram_headroom=lambda *args, **kwargs: None,
            estimate_qwen_text_tokens=lambda *args, **kwargs: 0,
            get_cache_key=lambda *args, **kwargs: "cache-key",
            get_alternative_cache_key=lambda *args, **kwargs: "alt-cache-key",
            get_image_hash=lambda *args, **kwargs: "image-hash",
            get_video_hash=lambda *args, **kwargs: "video-hash",
            save_prompt_cache=lambda *args, **kwargs: None,
            get_node_saved_prompt=lambda *args, **kwargs: None,
            get_node_saved_prompt_with_seed=lambda *args, **kwargs: None,
            resolve_qwen_thinking_mode=lambda *args, **kwargs: False,
            resolve_qwen_context_window=lambda *args, **kwargs: 0,
            set_node_saved_prompt=lambda *args, **kwargs: None,
            load_node_prompt_state=lambda *args, **kwargs: None,
            QwenVLBase=qwen_base,
            Quantization=_StubQuantization,
            TOOLTIPS={},
        ),
    }


def load_thinkingllm_loader_subset(module_filenames: list[str]):
    package_name = "thinkingllm_loader_smoke"
    tracked_names = [package_name, *[Path(filename).stem for filename in module_filenames]]
    previous_modules = {
        name: sys.modules[name]
        for name in tracked_names
        if name in sys.modules
    }
    real_listdir = os.listdir
    pkg_path = PKG.resolve()
    nodes_path = (PKG / "nodes").resolve()

    def fake_listdir(path):
        resolved = Path(path).resolve()
        if resolved == pkg_path:
            return ["__init__.py", *module_filenames]
        if resolved == nodes_path:
            return []
        return real_listdir(path)

    try:
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            with mock.patch("os.listdir", side_effect=fake_listdir):
                spec = importlib.util.spec_from_file_location(package_name, PKG / "__init__.py")
                if spec is None or spec.loader is None:
                    raise AssertionError("failed to create loader spec for ThinkingLLM package")
                module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = module
                spec.loader.exec_module(module)
                return module
    finally:
        for name in tracked_names:
            if name in previous_modules:
                sys.modules[name] = previous_modules[name]
            else:
                sys.modules.pop(name, None)


def load_module_from_file(filename: str, module_name: str):
    previous_module = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(module_name, PKG / filename)
        if spec is None or spec.loader is None:
            raise AssertionError(f"failed to create loader spec for {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def extract_dict_node(filename: str, name: str) -> ast.Dict:
    tree = parse_source(filename)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Dict):
                    return node.value
                raise AssertionError(f"{name} in {filename} is not a dict literal")
    raise AssertionError(f"{name} not found in {filename}")


def extract_mapping_target_names(filename: str, name: str) -> dict[str, str]:
    mapping = {}
    dict_node = extract_dict_node(filename, name)
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        key = ast.literal_eval(key_node)
        if isinstance(value_node, ast.Name):
            mapping[key] = value_node.id
        else:
            raise AssertionError(f"{name}[{key}] in {filename} is not a direct name reference")
    return mapping


def extract_literal_dict(filename: str, name: str) -> dict[str, str]:
    mapping = {}
    dict_node = extract_dict_node(filename, name)
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        key = ast.literal_eval(key_node)
        mapping[key] = ast.literal_eval(value_node)
    return mapping


def iter_workflow_types(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        node.get("type")
        for node in payload.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("type"), str)
    ]


def load_model_recommendations() -> dict:
    return json.loads((PKG / "web" / "model_recommendations.json").read_text(encoding="utf-8"))


def matches_recommendation_rule(model_name: str, rules: list[dict]) -> bool:
    lowered = model_name.lower()
    ordered = sorted(rules, key=lambda rule: rule.get("priority", 0), reverse=True)
    for rule in ordered:
        include = [token.lower() for token in rule.get("match_any", [])]
        exclude = [token.lower() for token in rule.get("exclude_any", [])]
        if any(token in lowered for token in include) and not any(token in lowered for token in exclude):
            return True
    return False


def iter_catalog_display_names() -> list[str]:
    names: list[str] = []
    hf_payload = json.loads((PKG / "hf_models.json").read_text(encoding="utf-8"))
    for section in ("hf_vl_models", "hf_text_models"):
        names.extend(list((hf_payload.get(section) or {}).keys()))

    gguf_payload = json.loads((PKG / "gguf_models.json").read_text(encoding="utf-8"))
    for section in ("Qwen_model", "qwenVL_model"):
        entries = gguf_payload.get(section) or {}
        for repo_key, repo in entries.items():
            if not isinstance(repo, dict):
                continue
            for model_file in repo.get("model_files") or []:
                names.append(str(Path(model_file).name))
    return names


class TestTerminalDisplayHelper(unittest.TestCase):
    def test_helper_file_exists(self):
        self.assertTrue((PKG / "AILab_StreamDisplay.py").exists())

    def test_helper_class_and_methods_exist(self):
        tree = parse_source("AILab_StreamDisplay.py")
        helper_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "TerminalStreamDisplay":
                helper_class = node
                break
        self.assertIsNotNone(helper_class, "TerminalStreamDisplay class missing")
        method_names = {
            node.name for node in helper_class.body if isinstance(node, ast.FunctionDef)
        }
        for method in ["start_stage", "push", "flush", "end_stage", "push_compact", "end_compact"]:
            self.assertIn(method, method_names)


class TestBufferedStreamingIntegration(unittest.TestCase):
    def test_old_rolling_tail_patterns_removed(self):
        old_patterns = [
            r'print\(f"\\r\{tail\}"',
            r"tail\s*=\s*full_streamed\[-200:\]",
            r"tail\s*=\s*full_text\[-200:\]",
        ]
        for filename in STREAMING_FILES:
            source = read_source(filename)
            for pattern in old_patterns:
                with self.subTest(filename=filename, pattern=pattern):
                    self.assertIsNone(re.search(pattern, source))

    def test_all_streaming_files_use_helper(self):
        for filename, expectations in STREAMING_FILES.items():
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn("TerminalStreamDisplay", source)
                self.assertIn(expectations["label"], source)
                if expectations["stage"]:
                    self.assertIn(expectations["stage"], source)
                self.assertIn(".push_compact(", source, f"{filename} should use compact streaming display")
                self.assertIn(".end_compact(", source, f"{filename} should finish compact streaming cleanly")

    def test_gguf_prompt_enhancer_has_no_off_toggle_text_stream_fallback(self):
        source = read_source("AILab_QwenVL_GGUF_PromptEnhancer.py")
        self.assertNotIn("_maybe_emit_prompt_compact_progress", source)


class TestModelRecommendations(unittest.TestCase):
    def test_recommendation_rules_have_sources(self):
        payload = load_model_recommendations()
        rules = payload.get("rules") or []
        self.assertTrue(rules, "model_recommendations.json should define at least one rule")
        for rule in rules:
            with self.subTest(rule=rule.get("id")):
                self.assertTrue(rule.get("sources"), "each recommendation rule should record its research source")

    def test_recommendation_notes_are_creative_work_focused(self):
        payload = load_model_recommendations()
        banned_terms = ("coding", "programming", "benchmark", "webdev", "math competition")
        for rule in payload.get("rules") or []:
            for note in rule.get("notes") or []:
                lowered = note.lower()
                with self.subTest(rule=rule.get("id"), note=note):
                    self.assertFalse(
                        any(term in lowered for term in banned_terms),
                        "recommendation notes should focus on creative prompt generation rather than coding/benchmarking",
                    )

    def test_recommendations_cover_all_catalog_models(self):
        payload = load_model_recommendations()
        rules = payload.get("rules") or []
        uncovered = [name for name in iter_catalog_display_names() if not matches_recommendation_rule(name, rules)]
        self.assertEqual([], uncovered)

    def test_qwen36_hauhau_exists_in_text_and_vision_catalogs(self):
        payload = json.loads((PKG / "gguf_models.json").read_text(encoding="utf-8"))
        text_catalog = payload.get("Qwen_model") or {}
        vision_catalog = payload.get("qwenVL_model") or {}

        text_key = next((key for key in text_catalog if "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive" in key), None)
        vision_key = next((key for key in vision_catalog if "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive" in key), None)

        self.assertIsNotNone(text_key)
        self.assertIsNotNone(vision_key)
        self.assertIn("mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf", vision_catalog[vision_key]["mmproj_file"])
        self.assertGreaterEqual(len(text_catalog[text_key].get("model_files") or []), 11)
        self.assertGreaterEqual(len(vision_catalog[vision_key].get("model_files") or []), 11)

    def test_gemma4_12b_is_available_for_hf_and_gguf_vision_and_text(self):
        hf_payload = json.loads((PKG / "hf_models.json").read_text(encoding="utf-8"))
        gguf_payload = json.loads((PKG / "gguf_models.json").read_text(encoding="utf-8"))

        self.assertTrue(any("Gemma-4-12B" in key for key in hf_payload.get("hf_vl_models", {})))
        self.assertTrue(any("Gemma-4-12B" in key for key in hf_payload.get("hf_text_models", {})))

        gguf_text = gguf_payload.get("Qwen_model") or {}
        gguf_vision = gguf_payload.get("qwenVL_model") or {}
        text_key = next((key for key in gguf_text if "Gemma-4-12B" in key), None)
        vision_key = next((key for key in gguf_vision if "Gemma-4-12B" in key), None)

        self.assertIsNotNone(text_key)
        self.assertIsNotNone(vision_key)
        self.assertIn("mmproj-F16.gguf", gguf_vision[vision_key]["mmproj_file"])
        self.assertGreaterEqual(len(gguf_text[text_key].get("model_files") or []), 8)
        self.assertGreaterEqual(len(gguf_vision[vision_key].get("model_files") or []), 8)

    def test_qwen3_asr_gguf_catalog_is_removed(self):
        gguf_payload = json.loads((PKG / "gguf_models.json").read_text(encoding="utf-8"))
        vision_catalog = gguf_payload.get("qwenVL_model") or {}
        asr_entries = {key: value for key, value in vision_catalog.items() if "Qwen3-ASR" in key}

        self.assertEqual({}, asr_entries)

    def test_legacy_gguf_model_file_entries_expand_to_runnable_models(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_legacy_gguf_model_files_expand_test",
            )

        models = module.GGUF_VL_CATALOG.get("models") or {}
        self.assertIn("gemma-4-12b-it-Q4_K_M.gguf [~7.5GB]", models)
        resolved = module._resolve_model_entry("gemma-4-12b-it-Q4_K_M.gguf [~7.5GB]")

        self.assertEqual(resolved.model_filename, "gemma-4-12b-it-Q4_K_M.gguf")
        self.assertEqual(resolved.mmproj_filename, "mmproj-F16.gguf")

    def test_ltx_presets_are_available_for_vision_and_prompt_enhancers(self):
        data = json.loads(Path("AILab_System_Prompts.json").read_text(encoding="utf-8"))

        self.assertIn("🎦 LTX 2.3 NSFW I2V Scene", data["_preset_prompts"])
        self.assertIn("🎦 LTX 2.3 NSFW I2V Scene", data["qwenvl"])
        self.assertIn("📖 LTX 2.3 NSFW T2V Scene", data["qwen_text"]["styles"])

        vision_prompt = data["qwenvl"]["🎦 LTX 2.3 NSFW I2V Scene"]
        text_prompt = data["qwen_text"]["styles"]["📖 LTX 2.3 NSFW T2V Scene"]["system_prompt"]
        self.assertIn("LTX 2.3", vision_prompt)
        self.assertIn("image first", vision_prompt)
        self.assertIn("LTX 2.3", text_prompt)
        self.assertIn("Do not change the user input intent", text_prompt)
        self.assertNotIn("Example:", vision_prompt)
        self.assertNotIn("Examples:", text_prompt)

    def test_custom_prompt_image_no_preset_alias_is_available_for_saved_workflows(self):
        label = "💬 Custom prompt + image (no preset)"
        data = json.loads(Path("AILab_System_Prompts.json").read_text(encoding="utf-8"))

        self.assertIn(label, data["_preset_prompts"])
        self.assertEqual(data["qwenvl"][label], "")

        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            package = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py"])

        values = package.NODE_CLASS_MAPPINGS["ThinkingLLM_QwenVL_GGUF"].INPUT_TYPES()["required"]["preset_prompt"][0]
        self.assertIn("🚫 No preset (image-only)", values)
        self.assertIn(label, values)

    def test_ltx_presets_are_exposed_by_relevant_nodes(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            package = load_thinkingllm_loader_subset(
                [
                    "AILab_QwenVL.py",
                    "AILab_QwenVL_GGUF.py",
                    "AILab_QwenVL_PromptEnhancer.py",
                    "AILab_QwenVL_GGUF_PromptEnhancer.py",
                ]
            )

        expectations = {
            "ThinkingLLM_QwenVL": "preset_prompt",
            "ThinkingLLM_QwenVL_Advanced": "preset_prompt",
            "ThinkingLLM_QwenVL_GGUF": "preset_prompt",
            "ThinkingLLM_QwenVL_GGUF_Advanced": "preset_prompt",
            "ThinkingLLM_QwenVL_PromptEnhancer": "enhancement_style",
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer": "preset_system_prompt",
        }
        for node_name, widget_name in expectations.items():
            with self.subTest(node_name=node_name):
                values = package.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["required"][widget_name][0]
                self.assertTrue(any("LTX 2.3" in value for value in values))

    def test_preset_tooltip_payload_matches_system_prompts(self):
        system_prompts = json.loads(Path("AILab_System_Prompts.json").read_text(encoding="utf-8"))
        tooltip_payload = json.loads(Path("web/preset_tooltips.json").read_text(encoding="utf-8"))

        expected = {
            "preset_prompt": system_prompts["qwenvl"],
            "enhancement_style": {
                name: entry["system_prompt"]
                for name, entry in system_prompts["qwen_text"]["styles"].items()
            },
            "preset_system_prompt": {
                name: entry["system_prompt"]
                for name, entry in system_prompts["qwen_text"]["styles"].items()
            },
        }
        self.assertEqual(tooltip_payload, expected)

    def test_appearance_js_hooks_preset_dropdown_tooltips(self):
        source = read_source("web/js/appearance.js")
        self.assertIn("preset_tooltips.json", source)
        self.assertIn("hookPresetTooltipPreviews(node)", source)
        for widget_name in ["preset_prompt", "enhancement_style", "preset_system_prompt"]:
            with self.subTest(widget_name=widget_name):
                self.assertIn(f'"{widget_name}"', source)

    def test_hf_default_flag_controls_default_model(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            package = load_thinkingllm_loader_subset(["AILab_QwenVL.py"])
        required = package.NODE_CLASS_MAPPINGS["ThinkingLLM_QwenVL"].INPUT_TYPES()["required"]
        _, meta = required["model_name"]
        self.assertEqual(meta["default"], "Qwen3-VL-4B-Instruct-Abliterated [DL: 7.5GB, VRAM: 6.0GB]")

    def test_appearance_js_loads_recommendation_widget(self):
        source = read_source("web/js/appearance.js")
        self.assertIn("model_recommendations.json", source)
        self.assertIn("ComfyWidgets", source)
        self.assertIn("recommended_settings", source)
        self.assertIn("support_notes?.audio", source)
        self.assertIn("AUDIO_UNSUPPORTED_MESSAGE", source)
        self.assertIn("else if (hasAudioInput(node))", source)
        self.assertIn("ThinkingLLM_Gemma4_Audio_GGUF", source)
        self.assertIn("ThinkingLLM_Whisper_ASR", source)
        self.assertIn("Whisper ASR transcription", source)
        self.assertNotIn("Prompt enhancer compact terminal progress is active", source)
        self.assertIn("queueRecommendationRefresh", source)

    def test_appearance_js_styles_all_visible_plugin_nodes(self):
        source = read_source("web/js/appearance.js")
        self.assertIn("if (!node.color && theme.nodeColor)", source)
        self.assertIn("if (!node.bgcolor && theme.nodeBgColor)", source)
        expected_node_names = [
            "AILab_QwenVL",
            "AILab_QwenVL_Advanced",
            "AILab_QwenVL_PromptEnhancer",
            "AILab_QwenVL_GGUF",
            "AILab_QwenVL_GGUF_Advanced",
            "AILab_QwenVL_GGUF_PromptEnhancer",
            "ThinkingLLM_QwenVL",
            "ThinkingLLM_QwenVL_Advanced",
            "ThinkingLLM_QwenVL_PromptEnhancer",
            "ThinkingLLM_QwenVL_GGUF",
            "ThinkingLLM_QwenVL_GGUF_Advanced",
            "ThinkingLLM_Gemma4_Audio_GGUF",
            "ThinkingLLM_Whisper_ASR",
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
            "VRAMCleanup",
            "StorySplitNode",
        ]
        for node_name in expected_node_names:
            with self.subTest(node_name=node_name):
                self.assertIn(f'"{node_name}"', source)

    def test_gguf_prompt_enhancer_caps_context_and_safe_constructs_llama(self):
        source = read_source("AILab_QwenVL_GGUF_PromptEnhancer.py")
        self.assertIn("_PROMPT_ENHANCER_MAX_CONTEXT_LENGTH = 32768", source)
        self.assertIn("_resolve_prompt_enhancer_context_length", source)
        self.assertIn("construct_llama_safely(Llama, kwargs", source)

    def test_gguf_nodes_guard_failed_llama_construction(self):
        source = read_source("AILab_QwenVL_GGUF.py")
        self.assertIn("def construct_llama_safely", source)
        self.assertIn("_thinkingllm_llama_init_complete", source)
        self.assertIn("construct_llama_safely(Llama, llm_kwargs_filtered", source)

    def test_gguf_streaming_files_use_shared_extractor(self):
        for filename in ("AILab_QwenVL_GGUF.py", "AILab_QwenVL_GGUF_PromptEnhancer.py"):
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn("extract_stream_token", source, f"{filename} should use shared chunk extraction")
                self.assertIn("display_token = reasoning_token + content_token", source)
                self.assertNotIn("display_token = reasoning_token or content_token", source)
                self.assertNotIn("stream_display.push(display_token)", source)

    def test_stream_token_extractor_handles_reasoning_content(self):
        from AILab_StreamDisplay import extract_stream_token
        # Reasoning chunk
        chunk = {"choices": [{"delta": {"reasoning_content": "Let me think...", "content": ""}}]}
        self.assertEqual(extract_stream_token(chunk), {"reasoning": "Let me think...", "content": ""})
        # Content-only chunk
        chunk2 = {"choices": [{"delta": {"reasoning_content": "", "content": "The answer is 42."}}]}
        self.assertEqual(extract_stream_token(chunk2), {"reasoning": "", "content": "The answer is 42."})
        # Mixed chunk
        chunk3 = {"choices": [{"delta": {"reasoning_content": "Hmm...", "content": "OK"}}]}
        self.assertEqual(extract_stream_token(chunk3), {"reasoning": "Hmm...", "content": "OK"})
        # Empty chunk
        self.assertEqual(extract_stream_token({}), {"reasoning": "", "content": ""})

        # llama.cpp/OpenAI-compatible variants seen across releases
        top_level_reasoning = {"reasoning_content": "thinking..."}
        self.assertEqual(extract_stream_token(top_level_reasoning), {"reasoning": "thinking...", "content": ""})

        message_chunk = {"choices": [{"message": {"content": "final answer"}}]}
        self.assertEqual(extract_stream_token(message_chunk), {"reasoning": "", "content": "final answer"})

        object_chunk = types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(reasoning_content="why", content=" then"))]
        )
        self.assertEqual(extract_stream_token(object_chunk), {"reasoning": "why", "content": " then"})

    def test_compact_stream_display_does_not_flush_unstyled_tail(self):
        from AILab_StreamDisplay import TerminalStreamDisplay
        import io

        display = TerminalStreamDisplay("UnitTest", flush_interval=999, min_chunk_chars=999, compact=True, line_width=24)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            display.start_stage("STREAMING")
            display.push_compact("partial compact token output with visible words")
            display.end_stage()
        output = stream.getvalue()
        self.assertNotIn("[UnitTest]", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("…", output)
        self.assertIn("partial compact token", output)
        self.assertIn("output with visible", output)
        self.assertIn("words", output)

    def test_stream_degeneracy_guard_catches_repeated_word_loop(self):
        from AILab_StreamDisplay import StreamDegenerationError, StreamDegenerationGuard, strip_degenerate_repetition

        guard = StreamDegenerationGuard(repeated_word_limit=5)
        with self.assertRaises(StreamDegenerationError):
            guard.push("mightiest, mightiest, mightiest, mightiest, mightiest, ")
        self.assertEqual(
            strip_degenerate_repetition(
                "A strong start, mightiest, mightiest, mightiest, mightiest, mightiest",
                repeated_word_limit=5,
            ),
            "A strong start",
        )

    def test_visible_streaming_has_no_reasoning_heartbeat_pollution(self):
        for filename in STREAMING_FILES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                if "GGUF" in filename or filename in {"AILab_QwenVL.py", "AILab_QwenVL_PromptEnhancer.py"}:
                    stream_pushes = [line for line in source.splitlines() if "push_compact" in line]
                    self.assertTrue(stream_pushes)
                self.assertNotIn("last_status_at = _maybe_emit_answer_stream_heartbeat", source)
                self.assertNotIn("last_status_at = _maybe_emit_hf_stream_heartbeat", source)
                self.assertNotIn("last_status_at = _maybe_emit_hf_prompt_stream_heartbeat", source)
                self.assertNotIn("last_status_at = _maybe_emit_prompt_stream_heartbeat", source)
                self.assertNotIn("terminal stream shows readable progress", source)


class TestPromptEnhancerMetadata(unittest.TestCase):
    def test_hf_prompt_enhancer_metadata_matches_runtime(self):
        source = read_source("AILab_QwenVL_PromptEnhancer.py")
        self.assertIn('RETURN_TYPES = ("STRING", "STRING")', source)
        self.assertIn('RETURN_NAMES = ("ENHANCED_OUTPUT", "RAW_TRACE")', source)
        self.assertIn('"stream_tokens_to_terminal": ("BOOLEAN"', source)
        self.assertIn('"unique_id": "UNIQUE_ID"', source)
        self.assertIn('"extra_pnginfo": "EXTRA_PNGINFO"', source)

    def test_raw_trace_still_exposed_across_nodes(self):
        expectations = {
            "AILab_QwenVL_PromptEnhancer.py": 'RETURN_NAMES = ("ENHANCED_OUTPUT", "RAW_TRACE")',
            "AILab_QwenVL_GGUF_PromptEnhancer.py": 'RETURN_NAMES = ("ENHANCED_OUTPUT", "RAW_TRACE")',
            "AILab_QwenVL.py": 'RETURN_NAMES = ("RESPONSE", "RAW_TRACE")',
            "AILab_QwenVL_GGUF.py": 'RETURN_NAMES = ("RESPONSE", "RAW_TRACE")',
        }
        for filename, marker in expectations.items():
            with self.subTest(filename=filename):
                self.assertIn(marker, read_source(filename))


class TestHuggingFaceTokenSupport(unittest.TestCase):
    TOKEN_NODE_FILES = [
        "AILab_QwenVL.py",
        "AILab_QwenVL_GGUF.py",
        "AILab_QwenVL_PromptEnhancer.py",
        "AILab_QwenVL_GGUF_PromptEnhancer.py",
    ]

    def _load_qwenvl_with_stubs(self):
        stub_modules = build_loader_test_stubs()
        stub_modules.pop("AILab_QwenVL", None)
        with mock.patch.dict(sys.modules, stub_modules, clear=False):
            return load_module_from_file("AILab_QwenVL.py", "thinkingllm_qwenvl_hf_token_test")

    def test_hf_token_widget_exists_in_download_nodes(self):
        for filename in self.TOKEN_NODE_FILES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn('"hf_token": ("STRING"', source)
                self.assertRegex(source, r'"hf_token": \("STRING", \{[^\n]+"tooltip"')

    def test_hf_token_is_not_part_of_prompt_or_cache_signatures(self):
        for filename in self.TOKEN_NODE_FILES:
            tree = parse_source(filename)
            with self.subTest(filename=filename):
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    if isinstance(node.func, ast.Name):
                        function_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        function_name = node.func.attr
                    else:
                        continue
                    if function_name not in {"build_node_input_signature", "get_cache_key"}:
                        continue
                    keyword_names = {keyword.arg for keyword in node.keywords if keyword.arg}
                    self.assertNotIn("hf_token", keyword_names)
                    for arg in node.args:
                        self.assertNotEqual(getattr(arg, "id", None), "hf_token")

    def test_hf_download_prefers_exact_file_and_passes_token(self):
        module = self._load_qwenvl_with_stubs()
        captured_kwargs = []

        def fake_hf_hub_download(**kwargs):
            captured_kwargs.append(dict(kwargs))
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            downloaded_path = local_dir / kwargs["filename"]
            downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.write_bytes(b"GGUF")
            return str(downloaded_path)

        def fail_snapshot_download(**kwargs):
            raise AssertionError("snapshot_download should not run when exact file download succeeds")

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "model.gguf"
            with mock.patch.object(module, "hf_hub_download", side_effect=fake_hf_hub_download):
                with mock.patch.object(module, "snapshot_download", side_effect=fail_snapshot_download):
                    module.download_hf_file_to_path(
                        ["private/repo"],
                        "model.gguf",
                        target_path,
                        hf_token="hf_secret_test_token",
                    )

        self.assertEqual(captured_kwargs[0]["token"], "hf_secret_test_token")
        self.assertEqual(captured_kwargs[0]["filename"], "model.gguf")

    def test_hf_download_errors_redact_token_and_add_access_hint(self):
        module = self._load_qwenvl_with_stubs()
        secret = "hf_secret_test_token"
        message = module._hf_download_error_message(
            RuntimeError(f"401 Unauthorized: {secret}"),
            repo_id="private/repo",
            hf_token=secret,
        )

        self.assertNotIn(secret, message)
        self.assertIn("<redacted HF token>", message)
        self.assertIn("access was rejected", message)

    def test_hf_token_can_fall_back_to_environment(self):
        module = self._load_qwenvl_with_stubs()
        with mock.patch.dict(module.os.environ, {"HF_TOKEN": "hf_env_secret"}, clear=True):
            self.assertEqual(module._clean_hf_token(""), "hf_env_secret")
            message = module._hf_download_error_message(
                RuntimeError("401 Unauthorized: hf_env_secret"),
                repo_id="private/repo",
                hf_token="",
            )

        self.assertNotIn("hf_env_secret", message)
        self.assertIn("<redacted HF token>", message)
        self.assertIn("access was rejected", message)

    def test_hf_file_download_failure_raises_redacted_message(self):
        module = self._load_qwenvl_with_stubs()
        secret = "hf_secret_test_token"

        def fake_snapshot_download(**kwargs):
            raise RuntimeError(f"403 Forbidden: {secret}")

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "missing.gguf"
            with mock.patch.object(module, "hf_hub_download", side_effect=fake_snapshot_download):
                with mock.patch.object(module, "snapshot_download", side_effect=fake_snapshot_download):
                    with self.assertRaises(FileNotFoundError) as raised:
                        module.download_hf_file_to_path(
                            ["private/repo"],
                            "missing.gguf",
                            target_path,
                            hf_token=secret,
                        )

        message = str(raised.exception)
        self.assertNotIn(secret, message)
        self.assertIn("<redacted HF token>", message)
        self.assertIn("access was rejected", message)

    def test_frontend_hides_only_legacy_gguf_advanced_shims(self):
        source = read_source("web/js/appearance.js")
        for widget_name in [
            "legacy_seed_mode",
            "legacy_unload_after_run",
        ]:
            with self.subTest(widget_name=widget_name):
                self.assertIn(f'"{widget_name}"', source)
        internal_widgets_block = source[
            source.index("const GGUF_ADVANCED_INTERNAL_WIDGETS = new Set(["):
            source.index("]);", source.index("const GGUF_ADVANCED_INTERNAL_WIDGETS = new Set([")):
        ]
        for widget_name in [
            "n_ubatch",
            "n_threads",
            "n_threads_batch",
            "flash_attn",
            "offload_kqv",
            "ctx_checkpoints",
        ]:
            with self.subTest(widget_name=widget_name):
                self.assertNotIn(f'"{widget_name}"', internal_widgets_block)
        self.assertNotIn('GGUF_ADVANCED_INTERNAL_WIDGETS = new Set([\n    "hf_token"', source)
        self.assertIn("hideStableWidget", source)
        self.assertIn("clearHfTokenAfterExecution", source)


class TestInterruptSupport(unittest.TestCase):
    INTERRUPT_FILES = [
        "AILab_QwenVL.py",
        "AILab_QwenVL_GGUF.py",
        "AILab_QwenVL_PromptEnhancer.py",
        "AILab_QwenVL_GGUF_PromptEnhancer.py",
    ]

    def test_interrupt_import_present(self):
        for filename in self.INTERRUPT_FILES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn("from comfy.model_management import throw_exception_if_processing_interrupted", source)

    def test_interrupt_check_in_loops(self):
        for filename in self.INTERRUPT_FILES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn("throw_exception_if_processing_interrupted()", source)

    def test_interrupt_after_blocking_calls(self):
        """Verify interrupt checks exist after non-streaming generate/create_chat_completion calls."""
        for filename in self.INTERRUPT_FILES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                # Every file should have at least one interrupt check outside streaming loops
                # (after blocking calls + in stream loops = at least 2 checks per file,
                #  but simple nodes may have only 1 non-streaming path)
                count = source.count("throw_exception_if_processing_interrupted()")
                self.assertGreaterEqual(
                    count, 1,
                    f"{filename} has only {count} interrupt check(s); expected >= 1"
                )


class TestNoResidualGlobals(unittest.TestCase):
    def test_no_global_last_saved_prompt_statement(self):
        for filename in [
            "AILab_QwenVL.py",
            "AILab_QwenVL_GGUF.py",
            "AILab_QwenVL_PromptEnhancer.py",
            "AILab_QwenVL_GGUF_PromptEnhancer.py",
        ]:
            tree = parse_source(filename)
            visitor = _GlobalCheck()
            visitor.visit(tree)
            with self.subTest(filename=filename):
                self.assertFalse(visitor.found)


class TestLoaderModuleRegistration(unittest.TestCase):
    def test_loader_registers_qwenvl_modules_with_stubbed_dependencies(self):
        package = load_thinkingllm_loader_subset(
            [
                "AILab_QwenVL_GGUF.py",
                "AILab_WhisperASR.py",
                "AILab_QwenVL_PromptEnhancer.py",
                "AILab_QwenVL_GGUF_PromptEnhancer.py",
            ]
        )

        for node_name in [
            "ThinkingLLM_QwenVL_GGUF",
            "ThinkingLLM_QwenVL_GGUF_Advanced",
            "ThinkingLLM_Gemma4_Audio_GGUF",
            "ThinkingLLM_Whisper_ASR",
            "ThinkingLLM_QwenVL_PromptEnhancer",
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
        ]:
            with self.subTest(node_name=node_name):
                self.assertIn(node_name, package.NODE_CLASS_MAPPINGS)
        self.assertNotIn("ThinkingLLM_Qwen3_ASR_GGUF", package.NODE_CLASS_MAPPINGS)


class TestLegacyNodeNameCompatibility(unittest.TestCase):
    CURRENT_TO_LEGACY = {
        "ThinkingLLM_QwenVL": "AILab_QwenVL",
        "ThinkingLLM_QwenVL_Advanced": "AILab_QwenVL_Advanced",
        "ThinkingLLM_QwenVL_GGUF": "AILab_QwenVL_GGUF",
        "ThinkingLLM_QwenVL_GGUF_Advanced": "AILab_QwenVL_GGUF_Advanced",
        "ThinkingLLM_QwenVL_PromptEnhancer": "AILab_QwenVL_PromptEnhancer",
        "ThinkingLLM_QwenVL_GGUF_PromptEnhancer": "AILab_QwenVL_GGUF_PromptEnhancer",
    }

    MAPPING_FILES = {
        "AILab_QwenVL.py": [
            "ThinkingLLM_QwenVL",
            "ThinkingLLM_QwenVL_Advanced",
        ],
        "AILab_QwenVL_GGUF.py": [
            "ThinkingLLM_QwenVL_GGUF",
            "ThinkingLLM_QwenVL_GGUF_Advanced",
        ],
        "AILab_QwenVL_PromptEnhancer.py": [
            "ThinkingLLM_QwenVL_PromptEnhancer",
        ],
        "AILab_QwenVL_GGUF_PromptEnhancer.py": [
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
        ],
    }

    def test_current_names_remain_canonical(self):
        for filename, canonical_names in self.MAPPING_FILES.items():
            mappings = extract_mapping_target_names(filename, "NODE_CLASS_MAPPINGS")
            for canonical_name in canonical_names:
                with self.subTest(filename=filename, canonical_name=canonical_name):
                    self.assertIn(canonical_name, mappings)

    def test_legacy_aliases_map_to_same_runtime_class_name(self):
        for filename, canonical_names in self.MAPPING_FILES.items():
            mappings = extract_mapping_target_names(filename, "NODE_CLASS_MAPPINGS")
            displays = extract_literal_dict(filename, "NODE_DISPLAY_NAME_MAPPINGS")
            for canonical_name in canonical_names:
                legacy_name = self.CURRENT_TO_LEGACY[canonical_name]
                with self.subTest(filename=filename, legacy_name=legacy_name):
                    self.assertEqual(mappings[canonical_name], mappings[legacy_name])
                    self.assertEqual(displays[canonical_name], displays[legacy_name])

    def test_example_workflow_legacy_types_are_covered(self):
        resolved_types = set()
        for filename in self.MAPPING_FILES:
            mappings = extract_mapping_target_names(filename, "NODE_CLASS_MAPPINGS")
            resolved_types.update(mappings.keys())

        workflow_files = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertTrue(workflow_files, "expected shipped example workflows")
        for workflow_file in workflow_files:
            for node_type in iter_workflow_types(workflow_file):
                if node_type.startswith("AILab_QwenVL"):
                    with self.subTest(workflow=workflow_file.name, node_type=node_type):
                        self.assertIn(node_type, resolved_types)


class TestGGUFAdvancedWorkflowCompatibility(unittest.TestCase):
    LEGACY_REQUIRED_PREFIX = [
        "model_name",
        "device",
        "preset_prompt",
        "custom_prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "repetition_penalty",
        "frame_count",
        "ctx",
        "n_batch",
        "gpu_layers",
        "image_max_tokens",
        "top_k",
        "pool_size",
        "keep_model_loaded",
        "seed",
    ]

    def _load_gguf_advanced_class(self):
        package = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py"])
        return package.NODE_CLASS_MAPPINGS["ThinkingLLM_QwenVL_GGUF_Advanced"]

    def test_gguf_advanced_required_widgets_preserve_legacy_order(self):
        node_cls = self._load_gguf_advanced_class()
        required_keys = list(node_cls.INPUT_TYPES()["required"].keys())

        self.assertEqual(
            required_keys[: len(self.LEGACY_REQUIRED_PREFIX)],
            self.LEGACY_REQUIRED_PREFIX,
        )
        self.assertEqual(
            required_keys[len(self.LEGACY_REQUIRED_PREFIX): len(self.LEGACY_REQUIRED_PREFIX) + 2],
            ["legacy_seed_mode", "legacy_unload_after_run"],
        )
        self.assertGreater(required_keys.index("n_ubatch"), required_keys.index("legacy_unload_after_run"))

    def test_gguf_advanced_process_signature_matches_required_widget_order(self):
        node_cls = self._load_gguf_advanced_class()
        signature = inspect.signature(node_cls.process)
        parameter_names = [
            name
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        expected_prefix = self.LEGACY_REQUIRED_PREFIX + [
            "legacy_seed_mode",
            "legacy_unload_after_run",
            "n_ubatch",
            "n_threads",
            "n_threads_batch",
            "flash_attn",
            "offload_kqv",
            "ctx_checkpoints",
        ]
        self.assertEqual(parameter_names[: len(expected_prefix)], expected_prefix)

    def test_gguf_advanced_accepts_legacy_serialized_values(self):
        node_cls = self._load_gguf_advanced_class()
        required = node_cls.INPUT_TYPES()["required"]

        legacy_seed_mode_options, legacy_seed_mode_meta = required["legacy_seed_mode"]
        n_ubatch_type, n_ubatch_meta = required["n_ubatch"]

        self.assertIn(False, legacy_seed_mode_options)
        self.assertEqual(legacy_seed_mode_meta["default"], "fixed")
        self.assertEqual(n_ubatch_type, "INT")
        self.assertEqual(n_ubatch_meta["min"], 0)

    def test_gguf_runtime_normalizes_legacy_zero_ubatch(self):
        source = read_source("AILab_QwenVL_GGUF.py")
        self.assertIn("n_ubatch_val = int(n_ubatch) if n_ubatch is not None else 0", source)
        self.assertIn("if n_ubatch_val <= 0:", source)
        self.assertIn("n_ubatch_val = min(n_batch_val, 512)", source)

    def test_gguf_vision_nodes_accept_optional_audio_without_legacy_order_change(self):
        node_cls = self._load_gguf_advanced_class()
        input_types = node_cls.INPUT_TYPES()

        self.assertEqual(input_types["optional"]["audio"], ("AUDIO",))

        simple_cls = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py"]).NODE_CLASS_MAPPINGS["ThinkingLLM_QwenVL_GGUF"]
        self.assertEqual(simple_cls.INPUT_TYPES()["optional"]["audio"], ("AUDIO",))

    def test_gguf_vision_advanced_allows_larger_reasoning_output_budget(self):
        node_cls = self._load_gguf_advanced_class()
        max_tokens_type, max_tokens_meta = node_cls.INPUT_TYPES()["required"]["max_tokens"]

        self.assertEqual(max_tokens_type, "INT")
        self.assertGreaterEqual(max_tokens_meta["max"], 32768)

    def test_simple_gguf_node_exposes_raw_trace(self):
        node_cls = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py"]).NODE_CLASS_MAPPINGS["ThinkingLLM_QwenVL_GGUF"]

        self.assertEqual(node_cls.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(node_cls.RETURN_NAMES, ("RESPONSE", "RAW_TRACE"))

    def test_gemma4_audio_node_is_audio_only_and_model_filtered(self):
        node_cls = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py"]).NODE_CLASS_MAPPINGS["ThinkingLLM_Gemma4_Audio_GGUF"]
        input_types = node_cls.INPUT_TYPES()

        optional = input_types["optional"]
        self.assertEqual(optional["audio"], ("AUDIO",))
        self.assertIn("audio_file_path", optional)
        self.assertNotIn("image", optional)
        self.assertNotIn("video", optional)

        model_keys, model_meta = input_types["required"]["model_name"]
        self.assertTrue(model_keys, "audio node should expose at least one Gemma 4 audio model")
        self.assertTrue(all("gemma-4" in key.lower() for key in model_keys))
        self.assertTrue(all("26b" not in key.lower() and "31b" not in key.lower() for key in model_keys))
        self.assertIn("gemma-4-12b", model_meta["default"].lower())
        self.assertIn("Audio", input_types["required"]["custom_prompt"][1]["default"])

    def test_gguf_nodes_disable_auto_finalization_retry_by_default(self):
        package = load_thinkingllm_loader_subset([
            "AILab_QwenVL_GGUF.py",
            "AILab_QwenVL_GGUF_PromptEnhancer.py",
        ])

        for node_name in (
            "ThinkingLLM_QwenVL_GGUF",
            "ThinkingLLM_QwenVL_GGUF_Advanced",
            "ThinkingLLM_Gemma4_Audio_GGUF",
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
        ):
            with self.subTest(node_name=node_name):
                required = package.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["required"]
                self.assertIn("auto_finalization_retry", required)
                self.assertEqual(required["auto_finalization_retry"][0], "BOOLEAN")
                self.assertEqual(required["auto_finalization_retry"][1]["default"], False)

    def test_qwen3_asr_node_is_not_registered(self):
        package = load_thinkingllm_loader_subset(["AILab_QwenVL_GGUF.py", "AILab_WhisperASR.py"])

        self.assertNotIn("ThinkingLLM_Qwen3_ASR_GGUF", package.NODE_CLASS_MAPPINGS)

    def test_whisper_asr_node_is_audio_only(self):
        node_cls = load_thinkingllm_loader_subset(["AILab_WhisperASR.py"]).NODE_CLASS_MAPPINGS["ThinkingLLM_Whisper_ASR"]
        input_types = node_cls.INPUT_TYPES()

        self.assertEqual(input_types["optional"]["audio"], ("AUDIO",))
        self.assertIn("audio_file_path", input_types["optional"])
        model_keys, model_meta = input_types["required"]["model_size"]
        self.assertIn("base", model_keys)
        self.assertIn("large-v3", model_keys)
        self.assertEqual(model_meta["default"], "small")
        self.assertEqual(input_types["required"]["device"][1]["default"], "cpu")
        self.assertEqual(input_types["required"]["compute_type"][1]["default"], "int8")
        self.assertNotIn("image", input_types["optional"])
        self.assertNotIn("video", input_types["optional"])

    def test_whisper_asr_returns_install_message_when_backend_missing(self):
        node_cls = load_thinkingllm_loader_subset(["AILab_WhisperASR.py"]).NODE_CLASS_MAPPINGS["ThinkingLLM_Whisper_ASR"]
        with mock.patch.dict(sys.modules, {"faster_whisper": None}, clear=False):
            transcript, segments_json, raw_trace = node_cls().process(
                model_size="base",
                language="auto",
                task="transcribe",
                device="auto",
                compute_type="auto",
                beam_size=5,
                vad_filter=True,
                audio_file_path="dummy.wav",
            )

        self.assertIn("faster-whisper is not installed", transcript)
        self.assertEqual(segments_json, "[]")
        self.assertIn("pip install faster-whisper", raw_trace)

    def test_whisper_asr_transcribes_with_mock_backend(self):
        module = load_module_from_file(
            "AILab_WhisperASR.py",
            "thinkingllm_whisper_mock_backend_test",
        )

        class FakeSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        fake_info = types.SimpleNamespace(language="de", language_probability=0.99, duration=1.0)
        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_size, device, compute_type, local_files_only=False):
                captured["init"] = (model_size, device, compute_type)

            def transcribe(self, path, language, task, beam_size, vad_filter):
                captured["transcribe"] = (Path(path).suffix, language, task, beam_size, vad_filter)
                return [FakeSegment(0.0, 1.0, " Hallo Welt. ")], fake_info

        fake_backend = build_stub_module("faster_whisper", WhisperModel=FakeWhisperModel)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 1600)

            with mock.patch.dict(sys.modules, {"faster_whisper": fake_backend}, clear=False):
                transcript, segments_json, raw_trace = module.ThinkingLLM_Whisper_ASR().process(
                    model_size="base",
                    language="de",
                    task="transcribe",
                    device="cpu",
                    compute_type="int8",
                    beam_size=3,
                    vad_filter=True,
                    audio_file_path=str(source),
                )

        self.assertEqual(transcript, "Hallo Welt.")
        self.assertEqual(json.loads(segments_json)[0]["text"], "Hallo Welt.")
        self.assertEqual(captured["init"], ("base", "cpu", "int8"))
        self.assertEqual(captured["transcribe"], (".wav", "de", "transcribe", 3, True))
        self.assertIn("language=de", raw_trace)

    def test_whisper_asr_prefers_cached_model_without_hf_ping(self):
        module = load_module_from_file(
            "AILab_WhisperASR.py",
            "thinkingllm_whisper_cached_model_test",
        )

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = " cached transcript "

        fake_info = types.SimpleNamespace(language="en", language_probability=0.99, duration=1.0)
        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_size, device, compute_type, local_files_only=False):
                captured["init"] = (model_size, device, compute_type, local_files_only)

            def transcribe(self, path, language, task, beam_size, vad_filter):
                return [FakeSegment()], fake_info

        fake_backend = build_stub_module("faster_whisper", WhisperModel=FakeWhisperModel)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 1600)

            with mock.patch.dict(sys.modules, {"faster_whisper": fake_backend}, clear=False):
                transcript, _, raw_trace = module.ThinkingLLM_Whisper_ASR().process(
                    model_size="small",
                    language="auto",
                    task="transcribe",
                    device="cpu",
                    compute_type="int8",
                    beam_size=5,
                    vad_filter=True,
                    audio_file_path=str(source),
                )

        self.assertEqual(transcript, "cached transcript")
        self.assertEqual(captured["init"], ("small", "cpu", "int8", True))
        self.assertIn("local_files_only=True", raw_trace)

    def test_whisper_asr_downloads_once_when_cache_is_missing(self):
        module = load_module_from_file(
            "AILab_WhisperASR.py",
            "thinkingllm_whisper_cache_miss_test",
        )

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = " downloaded transcript "

        fake_info = types.SimpleNamespace(language="en", language_probability=0.99, duration=1.0)
        attempts = []

        class FakeWhisperModel:
            def __init__(self, model_size, device, compute_type, local_files_only=False):
                attempts.append(local_files_only)
                if local_files_only:
                    raise RuntimeError("local_files_only=True and no cached snapshot was found")

            def transcribe(self, path, language, task, beam_size, vad_filter):
                return [FakeSegment()], fake_info

        fake_backend = build_stub_module("faster_whisper", WhisperModel=FakeWhisperModel)
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 1600)

            with mock.patch.dict(sys.modules, {"faster_whisper": fake_backend}, clear=False):
                transcript, _, raw_trace = module.ThinkingLLM_Whisper_ASR().process(
                    model_size="small",
                    language="auto",
                    task="transcribe",
                    device="cpu",
                    compute_type="int8",
                    beam_size=5,
                    vad_filter=True,
                    audio_file_path=str(source),
                )

        self.assertEqual(transcript, "downloaded transcript")
        self.assertEqual(attempts, [True, False])
        self.assertIn("local_files_only=False", raw_trace)
        self.assertIn("Local Whisper cache miss", raw_trace)

    def test_audio_to_wav_base64_resamples_to_16khz_mono(self):
        import numpy as real_numpy

        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_audio_resample_test",
            )
        module.np = real_numpy

        waveform = real_numpy.stack([
            real_numpy.linspace(-0.5, 0.5, 800, dtype=real_numpy.float32),
            real_numpy.linspace(0.5, -0.5, 800, dtype=real_numpy.float32),
        ])

        encoded = module._audio_to_wav_base64({"waveform": waveform, "sample_rate": 8000})

        self.assertEqual(len(encoded), 1)
        with wave.open(io.BytesIO(base64.b64decode(encoded[0])), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 1600)

    def test_audio_to_wav_base64_rejects_invalid_audio_metadata(self):
        import numpy as real_numpy

        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_audio_validation_test",
            )
        module.np = real_numpy

        self.assertEqual(module._audio_to_wav_base64({"waveform": real_numpy.zeros((1, 10)), "sample_rate": 0}), [])
        self.assertEqual(module._audio_to_wav_base64({"sample_rate": 16000}), [])

    def test_audio_file_path_to_wav_base64_decodes_m4a_with_ffmpeg(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required for audio file path decoding")

        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_audio_file_path_test",
            )

        with tempfile.TemporaryDirectory(prefix="thinkingllm audio ") as temp_dir:
            m4a_path = Path(temp_dir) / "sample voice memo.m4a"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.25",
                    "-c:a",
                    "aac",
                    str(m4a_path),
                ],
                check=True,
                capture_output=True,
            )

            encoded = module._audio_file_path_to_wav_base64(str(m4a_path))

        self.assertEqual(len(encoded), 1)
        with wave.open(io.BytesIO(base64.b64decode(encoded[0])), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertGreater(wav.getnframes(), 3000)
            self.assertLess(wav.getnframes(), 5000)

    def test_audio_nodes_skip_saved_prompt_cache(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_audio_nodes_skip_saved_prompt_cache_test",
            )

        class DummyLoader(module.QwenVLGGUFBase):
            def _load_model(self, *args, **kwargs):
                self.chat_handler = object()
                self.last_backend_trace = ""

            def _encode_media(self, image, video, frame_count):
                return []

            def _invoke(self, *args, **kwargs):
                return "real transcript", "raw transcript"

        with mock.patch.object(
            module,
            "get_node_saved_prompt_with_seed",
            side_effect=AssertionError("audio nodes must not reuse saved prompts"),
        ), mock.patch.object(
            module,
            "set_node_saved_prompt",
            side_effect=AssertionError("audio nodes must not store saved prompts"),
        ), mock.patch.object(
            module,
            "_audio_inputs_to_wav_base64",
            return_value=(["BASE64WAV"], []),
        ), mock.patch.object(
            module,
            "_resolve_model_entry",
            return_value=types.SimpleNamespace(context_length=4096),
        ):
            text, raw_trace = DummyLoader().run(
                model_name="Gemma-4-12B-it-Q4_K_M.gguf [~7.5GB]",
                preset_prompt="🚫 No preset (image-only)",
                custom_prompt="Analyze the audio.",
                image=None,
                video=None,
                frame_count=1,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                seed=1,
                keep_model_loaded=False,
                device="auto",
                node_class="ThinkingLLM_Gemma4_Audio_GGUF",
            )

        self.assertEqual(text, "real transcript")
        self.assertEqual(raw_trace, "raw transcript")


class TestLlamaCppInstaller(unittest.TestCase):
    def _resolve_linux_install_spec(self, installer, *, py_minor, cuda_version, machine="x86_64"):
        with mock.patch.dict(installer.os.environ, {}, clear=True):
            with mock.patch.object(
                installer.sys,
                "version_info",
                types.SimpleNamespace(major=3, minor=py_minor),
            ):
                with mock.patch.object(installer.platform, "machine", return_value=machine):
                    cuda_versions = {
                        "toolkit": "",
                        "torch": cuda_version,
                        "nvidia_smi": "",
                        "selected": cuda_version,
                        "selected_source": "torch" if cuda_version else "unknown",
                    }
                    with mock.patch.object(installer, "_detect_cuda_versions", return_value=cuda_versions):
                        return installer._resolve_linux_install_spec()

    def test_linux_install_spec_prefers_known_cuda_128_wheels_for_verified_python_tags(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_linux_spec_test",
        )

        for py_tag in ("cp310", "cp311", "cp312", "cp313", "cp314"):
            with self.subTest(py_tag=py_tag):
                install_spec, install_reason = self._resolve_linux_install_spec(
                    installer,
                    py_minor=int(py_tag[-2:]),
                    cuda_version="12.8",
                )
                expected_spec = (
                    "https://github.com/JamePeng/llama-cpp-python/releases/download/"
                    "v0.3.35-cu128-Basic-linux-20260406/"
                    f"llama_cpp_python-0.3.35+cu128.basic-{py_tag}-{py_tag}-linux_x86_64.whl"
                )

                self.assertEqual(install_spec, expected_spec)
                self.assertEqual(install_reason, "known Linux wheel (torch CUDA 12.8)")
                self.assertNotEqual(install_spec, installer.DEFAULT_JAMEPENG_GIT_SPEC)

    def test_linux_install_spec_keeps_existing_known_cuda_124_wheel(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_linux_spec_124_test",
        )

        install_spec, install_reason = self._resolve_linux_install_spec(
            installer,
            py_minor=12,
            cuda_version="12.4",
        )

        self.assertEqual(
            install_spec,
            "https://github.com/JamePeng/llama-cpp-python/releases/download/"
            "v0.3.34-cu124-Basic-linux-20260331/"
            "llama_cpp_python-0.3.34+cu124.basic-cp312-cp312-linux_x86_64.whl",
        )
        self.assertEqual(install_reason, "known Linux wheel (torch CUDA 12.4)")
        self.assertNotEqual(install_spec, installer.DEFAULT_JAMEPENG_GIT_SPEC)

    def test_linux_install_spec_blocks_unverified_combinations_by_default(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_linux_spec_fallback_test",
        )

        install_spec, install_reason = self._resolve_linux_install_spec(
            installer,
            py_minor=11,
            cuda_version="12.4",
        )

        self.assertEqual(install_spec, "")
        self.assertEqual(install_reason, "no verified Linux wheel match")

    def test_linux_install_spec_source_build_requires_explicit_opt_in(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_linux_spec_source_opt_in_test",
        )

        with mock.patch.dict(installer.os.environ, {"THINKINGLLM_LLAMA_CPP_ALLOW_SOURCE_BUILD": "1"}, clear=True):
            with mock.patch.object(installer.sys, "version_info", types.SimpleNamespace(major=3, minor=11)):
                with mock.patch.object(installer.platform, "machine", return_value="x86_64"):
                    with mock.patch.object(
                        installer,
                        "_detect_cuda_versions",
                        return_value={"toolkit": "", "torch": "12.4", "nvidia_smi": "", "selected": "12.4", "selected_source": "torch"},
                    ):
                        install_spec, install_reason = installer._resolve_linux_install_spec()

        self.assertEqual(install_spec, installer.DEFAULT_JAMEPENG_GIT_SPEC)
        self.assertEqual(install_reason, "source-build fallback (explicit opt-in)")

    def test_source_build_env_enables_cuda_when_toolkit_exists(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_source_env_test",
        )

        with mock.patch.dict(installer.os.environ, {"CMAKE_ARGS": "-DOTHER=1"}, clear=True):
            with mock.patch.object(installer, "_detect_cuda_home_version", return_value="12.8"):
                install_env = installer._build_source_install_env()

        self.assertIn("-DOTHER=1", install_env["CMAKE_ARGS"])
        self.assertIn("-DGGML_CUDA=on", install_env["CMAKE_ARGS"])
        self.assertEqual(install_env["FORCE_CMAKE"], "1")

    def test_windows_runtime_dll_dirs_include_torch_lib(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_dirs_test",
        )
        torch_module = types.SimpleNamespace(__file__=r"C:\stub\torch\__init__.py")

        with mock.patch.dict(sys.modules, {"torch": torch_module}, clear=False):
            with mock.patch.object(installer.os.path, "isdir", return_value=True):
                self.assertEqual(
                    installer._get_windows_runtime_dll_dirs(),
                    [r"C:\stub\torch\lib"],
                )

    def test_windows_long_path_dll_fallback_uses_path(self):
        installer = load_module_from_file(
            "AILab_LlamaCppInstaller.py",
            "thinkingllm_llama_installer_test",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = os.environ.get("PATH", "")

            def raise_long_path_error(path):
                error = FileNotFoundError(f"[WinError 206] path too long: {path}")
                error.winerror = 206
                raise error

            with mock.patch.object(installer.platform, "system", return_value="Windows"):
                with mock.patch.object(installer.os, "add_dll_directory", side_effect=raise_long_path_error):
                    with mock.patch.dict(installer.os.environ, {"PATH": original_path}, clear=False):
                        with installer._relax_windows_dll_directory_for_long_paths():
                            handle = installer.os.add_dll_directory(temp_dir)
                            self.assertTrue(installer.os.environ["PATH"].startswith(temp_dir))

                        self.assertFalse(installer.os.environ["PATH"].startswith(temp_dir + ";"))
                        self.assertTrue(hasattr(handle, "close"))


class TestGGUFWindllRuntimeFallback(unittest.TestCase):
    def test_gguf_loader_reuses_catalog_model_from_extra_llm_path_without_download(self):
        calls = []

        class DummyChatHandler:
            def __init__(self, **kwargs):
                calls.append(("chat", kwargs.get("clip_model_path")))

        class DummyLlama:
            def __init__(self, **kwargs):
                calls.append(("llama", kwargs.get("model_path")))

        chat_format_module = build_stub_module(
            "llama_cpp.llama_chat_format",
            Qwen35VLChatHandler=DummyChatHandler,
            Qwen3VLChatHandler=DummyChatHandler,
        )
        installer_module = build_stub_module(
            "AILab_LlamaCppInstaller",
            ensure_llama_cpp_backend=lambda *args, **kwargs: DummyLlama,
            format_llama_cpp_backend_info=lambda info=None: "stub backend",
            get_last_llama_cpp_backend_info=lambda: {
                "gpu_offload": True,
                "vision_handlers": ["Qwen35VLChatHandler"],
            },
            relax_windows_dll_directory_for_long_paths=contextlib.nullcontext,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dev_models = temp_root / "dev" / "models"
            shared_llm = temp_root / "shared" / "models" / "LLM"
            shared_model_dir = shared_llm / "GGUF" / "HauhauCS" / "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive"
            shared_model_dir.mkdir(parents=True)
            model_filename = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"
            mmproj_filename = "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
            (shared_model_dir / model_filename).write_bytes(b"GGUF")
            (shared_model_dir / mmproj_filename).write_bytes(b"mmproj")

            stubs = build_loader_test_stubs()
            stubs.update(
                {
                    "folder_paths": build_stub_module(
                        "folder_paths",
                        models_dir=str(dev_models),
                        folder_names_and_paths={"LLM": ([str(shared_llm)], set())},
                        get_folder_paths=lambda name: [str(shared_llm)] if name == "LLM" else [],
                    ),
                    "AILab_LlamaCppInstaller": installer_module,
                    "llama_cpp": build_stub_module("llama_cpp", llama_chat_format=chat_format_module),
                    "llama_cpp.llama_chat_format": chat_format_module,
                }
            )

            with mock.patch.dict(sys.modules, stubs, clear=False):
                module = load_module_from_file(
                    "AILab_QwenVL_GGUF.py",
                    "thinkingllm_gguf_extra_llm_reuse_test",
                )

            model_name = next(
                name
                for name in module.GGUF_VL_CATALOG["models"]
                if name.startswith(model_filename)
            )
            loader = module.QwenVLGGUFBase()

            with mock.patch.dict(sys.modules, stubs, clear=False), mock.patch.object(
                module,
                "_download_single_file",
                side_effect=AssertionError("download should not be called for shared local GGUF"),
            ), mock.patch.object(
                module,
                "_pick_device",
                return_value="cpu",
            ), mock.patch.object(
                module,
                "read_gguf_architecture",
                return_value="qwen35",
            ):
                loader._load_model(
                    model_name=model_name,
                    device="cpu",
                    ctx=128,
                    n_batch=16,
                    n_ubatch=None,
                    gpu_layers=0,
                    image_max_tokens=64,
                    top_k=0,
                    pool_size=1024,
                    n_threads=None,
                    n_threads_batch=None,
                    flash_attn=False,
                    offload_kqv=False,
                    ctx_checkpoints=None,
                    enable_thinking=True,
                    unique_id=None,
                )

        self.assertEqual(calls[0][0], "chat")
        self.assertTrue(str(calls[0][1]).endswith(mmproj_filename))
        self.assertEqual(calls[1][0], "llama")
        self.assertTrue(str(calls[1][1]).endswith(model_filename))

    def test_gguf_runtime_reuses_windows_dll_relaxation_for_handler_and_llama(self):
        enter_events = []

        @contextlib.contextmanager
        def tracking_relaxation():
            enter_events.append("enter")
            try:
                yield
            finally:
                enter_events.append("exit")

        class DummyChatHandler:
            def __init__(self, **kwargs):
                enter_events.append(("chat", sorted(kwargs)))

        class DummyLlama:
            def __init__(self, **kwargs):
                enter_events.append(("llama", sorted(kwargs)))

        chat_format_module = build_stub_module(
            "llama_cpp.llama_chat_format",
            Qwen3VLChatHandler=DummyChatHandler,
        )
        installer_module = build_stub_module(
            "AILab_LlamaCppInstaller",
            ensure_llama_cpp_backend=lambda *args, **kwargs: DummyLlama,
            format_llama_cpp_backend_info=lambda info=None: "stub backend",
            get_last_llama_cpp_backend_info=lambda: {"gpu_offload": True, "vision_handlers": ["Qwen3VLChatHandler"]},
            relax_windows_dll_directory_for_long_paths=tracking_relaxation,
        )
        stub_modules = build_loader_test_stubs()
        stub_modules.update(
            {
                "AILab_LlamaCppInstaller": installer_module,
                "llama_cpp": build_stub_module("llama_cpp", llama_chat_format=chat_format_module),
                "llama_cpp.llama_chat_format": chat_format_module,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.gguf"
            mmproj_path = Path(temp_dir) / "mmproj.gguf"
            model_path.write_bytes(b"GGUF")
            mmproj_path.write_bytes(b"mmproj")

            with mock.patch.dict(sys.modules, stub_modules, clear=False):
                module = load_module_from_file(
                    "AILab_QwenVL_GGUF.py",
                    "thinkingllm_gguf_runtime_dll_test",
                )
                loader = module.QwenVLGGUFBase()
                resolved = module.GGUFVLResolved(
                    display_name="stub",
                    repo_id=None,
                    alt_repo_ids=[],
                    author=None,
                    repo_dirname="stub",
                    model_filename=str(model_path),
                    mmproj_filename=str(mmproj_path),
                    context_length=4096,
                    image_max_tokens=1024,
                    n_batch=64,
                    gpu_layers=0,
                    top_k=20,
                    pool_size=1024,
                )

                with mock.patch.object(module, "_resolve_model_entry", return_value=resolved):
                    with mock.patch.object(module, "_pick_device", return_value="cpu"):
                        with mock.patch.object(module, "read_gguf_architecture", return_value="qwen3"):
                            loader._load_model(
                                model_name="stub",
                                device="cpu",
                                ctx=None,
                                n_batch=None,
                                n_ubatch=None,
                                gpu_layers=None,
                                image_max_tokens=None,
                                top_k=None,
                                pool_size=None,
                                n_threads=None,
                                n_threads_batch=None,
                                flash_attn=False,
                                offload_kqv=False,
                                ctx_checkpoints=None,
                                enable_thinking=True,
                                unique_id=None,
                            )

        self.assertEqual(enter_events[0:3], [
            "enter",
            ("chat", ["clip_model_path", "force_reasoning", "image_max_tokens", "verbose"]),
            "exit",
        ])
        self.assertEqual(enter_events[3], "enter")
        self.assertEqual(enter_events[5], "exit")
        self.assertEqual(enter_events[4][0], "llama")
        self.assertTrue(
            {
                "chat_handler",
                "model_path",
                "n_ctx",
                "image_max_tokens",
                "image_min_tokens",
            }.issubset(set(enter_events[4][1]))
        )

    def test_gguf_thinking_toggle_updates_without_reload_signature(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_gguf_thinking_toggle_test",
            )
        source = read_source("AILab_QwenVL_GGUF.py")
        self.assertNotIn("ctx_checkpoints_val,\n            bool(enable_thinking),", source)

        captured_kwargs = []

        class DummyLlama:
            def __init__(self):
                self.chat_template_kwargs = {"enable_thinking": True}

            def create_chat_completion(self, **kwargs):
                captured_kwargs.append(kwargs)
                return {"choices": [{"message": {"content": "ok"}}]}

        loader = module.QwenVLGGUFBase()
        loader.llm = DummyLlama()
        loader.uses_qwen_template_thinking = True
        loader._create_chat_completion(
            enable_thinking=False,
            messages=[],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stream=False,
        )

        self.assertEqual(loader.llm.chat_template_kwargs["enable_thinking"], False)
        self.assertEqual(captured_kwargs[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_create_chat_completion_retries_without_unexpected_kwarg(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_gguf_unexpected_kwarg_retry_test",
            )

        captured_kwargs = []

        class DummyLlama:
            def create_chat_completion(self, **kwargs):
                captured_kwargs.append(dict(kwargs))
                if "repeat_last_n" in kwargs:
                    raise TypeError("Llama.create_chat_completion() got an unexpected keyword argument 'repeat_last_n'")
                return {"choices": [{"message": {"content": "ok"}}]}

        loader = module.QwenVLGGUFBase()
        loader.llm = DummyLlama()
        loader.uses_qwen_template_thinking = False

        result = loader._create_chat_completion(
            enable_thinking=False,
            messages=[],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            stream=True,
            repeat_last_n=48,
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertIn("repeat_last_n", captured_kwargs[0])
        self.assertNotIn("repeat_last_n", captured_kwargs[1])

    def test_gguf_invoke_resets_before_each_completion_call(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_gguf_reset_before_completion_test",
            )

        class DummyLlama:
            def __init__(self):
                self.events = []
                self.reset_count = 0

            def reset(self):
                self.events.append("reset")
                self.reset_count += 1

            def create_chat_completion(self, **kwargs):
                self.events.append("completion")
                return iter([
                    {"choices": [{"delta": {"content": "ok"}}]},
                ])

        loader = module.QwenVLGGUFBase()
        loader.llm = DummyLlama()

        text, raw_trace = loader._invoke(
            system_prompt="system",
            user_prompt="user",
            images_b64=[],
            audio_b64=[],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
            stream_to_terminal=False,
            enable_thinking=False,
        )

        self.assertEqual(text, "ok")
        self.assertIn("ok", raw_trace)
        self.assertEqual(loader.llm.reset_count, 1)
        self.assertEqual(loader.llm.events, ["reset", "completion"])

    def test_gguf_invoke_does_not_auto_finalize_by_default(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_gguf_no_default_finalization_test",
            )

        class DummyLlama:
            def __init__(self):
                self.events = []
                self.calls = 0

            def reset(self):
                self.events.append(("reset", self.calls))

            def create_chat_completion(self, **kwargs):
                self.calls += 1
                self.events.append(("completion", self.calls))
                return iter([
                    {"choices": [{"delta": {"reasoning_content": "hidden reasoning only"}}]},
                ])

        loader = module.QwenVLGGUFBase()
        loader.llm = DummyLlama()

        text, raw_trace = loader._invoke(
            system_prompt="system",
            user_prompt="user",
            images_b64=[],
            audio_b64=[],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
            stream_to_terminal=False,
            enable_thinking=True,
        )

        self.assertEqual(text, "")
        self.assertIn("[STREAMING]", raw_trace)
        self.assertNotIn("FINALIZATION ATTEMPT", raw_trace)
        self.assertEqual(loader.llm.events, [("reset", 0), ("completion", 1)])

    def test_gguf_invoke_resets_before_finalization_retry_completion(self):
        with mock.patch.dict(sys.modules, build_loader_test_stubs(), clear=False):
            module = load_module_from_file(
                "AILab_QwenVL_GGUF.py",
                "thinkingllm_gguf_reset_finalization_test",
            )

        class DummyLlama:
            def __init__(self):
                self.events = []
                self.calls = 0

            def reset(self):
                self.events.append(("reset", self.calls))

            def create_chat_completion(self, **kwargs):
                self.calls += 1
                self.events.append(("completion", self.calls))
                if self.calls == 1:
                    return iter([
                        {"choices": [{"delta": {"reasoning_content": "hidden reasoning only"}}]},
                    ])
                return iter([
                    {"choices": [{"delta": {"content": "final answer"}}]},
                ])

        loader = module.QwenVLGGUFBase()
        loader.llm = DummyLlama()

        text, raw_trace = loader._invoke(
            system_prompt="system",
            user_prompt="user",
            images_b64=[],
            audio_b64=[],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=1,
            stream_to_terminal=False,
            enable_thinking=True,
            auto_finalization_retry=True,
        )

        self.assertEqual(text, "final answer")
        self.assertIn("[STREAMING]", raw_trace)
        self.assertIn("[FINALIZATION ATTEMPT 2/3]", raw_trace)
        self.assertEqual(
            loader.llm.events,
            [("reset", 0), ("completion", 1), ("reset", 1), ("completion", 2)],
        )


class _GlobalCheck(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Global(self, node):
        if "LAST_SAVED_PROMPT" in node.names:
            self.found = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
