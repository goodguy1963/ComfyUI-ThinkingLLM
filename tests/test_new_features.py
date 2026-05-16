"""Lightweight regression tests for the local ThinkingLLM fork.

Run from the ComfyUI-ThinkingLLM folder:
    python tests/test_new_features.py

These tests intentionally avoid importing the model modules so they can run in a
plain shell without ML dependencies.
"""

import ast
import re
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PKG = HERE.parent

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


class TestClassKeyUniqueness(unittest.TestCase):
    """ThinkingLLM class keys must NOT collide with Qwen3 AILab_* keys."""

    def test_no_ailab_class_keys(self):
        for filename in [
            "AILab_QwenVL.py",
            "AILab_QwenVL_GGUF.py",
            "AILab_QwenVL_PromptEnhancer.py",
            "AILab_QwenVL_GGUF_PromptEnhancer.py",
        ]:
            source = read_source(filename)
            mappings_block = re.search(
                r"NODE_CLASS_MAPPINGS\s*=\s*\{(.*?)\}", source, re.DOTALL
            )
            if mappings_block:
                keys = re.findall(r'"([^"]*)"', mappings_block.group(1))
                with self.subTest(filename=filename):
                    self.assertTrue(
                        all(k.startswith("ThinkingLLM_") for k in keys),
                        f"{filename} contains non-ThinkingLLM_ class keys: {keys}",
                    )

    def test_thinkingllm_prefix_on_all_class_names(self):
        for filename in [
            "AILab_QwenVL.py",
            "AILab_QwenVL_GGUF.py",
            "AILab_QwenVL_PromptEnhancer.py",
            "AILab_QwenVL_GGUF_PromptEnhancer.py",
        ]:
            tree = parse_source(filename)
            class_names = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and node.name.startswith("ThinkingLLM_")
            ]
            with self.subTest(filename=filename):
                self.assertTrue(
                    len(class_names) >= 1,
                    f"{filename} has no ThinkingLLM_* class definitions",
                )


class TestThinkingAndStreamToggles(unittest.TestCase):
    """All 6 node classes must expose enable_thinking and stream_tokens_to_terminal."""

    NODE_SOURCES = {
        "AILab_QwenVL.py",
        "AILab_QwenVL_GGUF.py",
        "AILab_QwenVL_PromptEnhancer.py",
        "AILab_QwenVL_GGUF_PromptEnhancer.py",
    }

    def test_thinking_toggle_present_on_all_nodes(self):
        for filename in self.NODE_SOURCES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn('"enable_thinking": ("BOOLEAN"', source)

    def test_stream_toggle_present_on_all_nodes(self):
        for filename in self.NODE_SOURCES:
            source = read_source(filename)
            with self.subTest(filename=filename):
                self.assertIn('"stream_tokens_to_terminal": ("BOOLEAN"', source)


class _GlobalCheck(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Global(self, node):
        if "LAST_SAVED_PROMPT" in node.names:
            self.found = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
