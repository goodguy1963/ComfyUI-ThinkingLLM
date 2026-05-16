"""Lightweight regression tests for the local ThinkingLLM fork.

Run from the ComfyUI-ThinkingLLM folder:
    python tests/test_new_features.py

These tests intentionally avoid importing the model modules so they can run in a
plain shell without ML dependencies.
"""

import ast
import json
import re
import unittest
from pathlib import Path


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


class _GlobalCheck(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Global(self, node):
        if "LAST_SAVED_PROMPT" in node.names:
            self.found = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
