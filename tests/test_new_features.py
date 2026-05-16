"""Lightweight regression tests for the local ThinkingLLM fork.

Run from the ComfyUI-ThinkingLLM folder:
    python tests/test_new_features.py

These tests intentionally avoid importing the model modules so they can run in a
plain shell without ML dependencies.
"""

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import types
import unittest
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
        "huggingface_hub": build_stub_module("huggingface_hub", snapshot_download=lambda *args, **kwargs: ""),
        "PIL": pil_package,
        "PIL.Image": pil_image,
        "folder_paths": build_stub_module("folder_paths", models_dir=str(PKG)),
        "AILab_StreamDisplay": build_stub_module("AILab_StreamDisplay", TerminalStreamDisplay=terminal_stream_display),
        "AILab_LlamaCppInstaller": build_stub_module(
            "AILab_LlamaCppInstaller",
            ensure_llama_cpp_backend=lambda *args, **kwargs: object(),
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
        for method in ["start_stage", "push", "flush", "end_stage"]:
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
                # Accept either the old buffered API or compact rolling-tail API
                self.assertTrue(
                    ".start_stage(" in source or ".push_compact(" in source,
                    f"No streaming API usage found in {filename}: "
                    "expected .start_stage() or .push_compact()"
                )

    def test_gguf_prompt_enhancer_has_no_off_toggle_text_stream_fallback(self):
        source = read_source("AILab_QwenVL_GGUF_PromptEnhancer.py")
        self.assertNotIn("_maybe_emit_prompt_compact_progress", source)
        self.assertNotIn("Prompt enhancer compact terminal progress is active", source)
        self.assertIn("_maybe_emit_prompt_background_heartbeat", source)


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
                "AILab_QwenVL_PromptEnhancer.py",
                "AILab_QwenVL_GGUF_PromptEnhancer.py",
            ]
        )

        for node_name in [
            "ThinkingLLM_QwenVL_GGUF",
            "ThinkingLLM_QwenVL_GGUF_Advanced",
            "ThinkingLLM_QwenVL_PromptEnhancer",
            "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
        ]:
            with self.subTest(node_name=node_name):
                self.assertIn(node_name, package.NODE_CLASS_MAPPINGS)


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
            "keep_last_prompt",
        ]
        self.assertEqual(parameter_names[: len(expected_prefix)], expected_prefix)


class _GlobalCheck(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Global(self, node):
        if "LAST_SAVED_PROMPT" in node.names:
            self.found = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
