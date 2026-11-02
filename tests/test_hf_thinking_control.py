import ast
import unittest
from pathlib import Path

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output


ROOT = Path(__file__).resolve().parents[1]


class HFThinkingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enhancer_source = (ROOT / "AILab_QwenVL_PromptEnhancer.py").read_text(encoding="utf-8")
        cls.enhancer_tree = ast.parse(cls.enhancer_source)
        cls.runtime_source = (ROOT / "thinkingllm_core" / "hf_runtime.py").read_text(encoding="utf-8")
        cls.runtime_tree = ast.parse(cls.runtime_source)

    def test_hf_text_enhancer_passes_thinking_as_template_variable(self):
        calls = [node for node in ast.walk(self.enhancer_tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template"]
        self.assertTrue(calls, "HF text enhancer must call apply_chat_template")
        self.assertIn('local_template_kwargs["enable_thinking"] = thinking_enabled', self.enhancer_source)
        self.assertNotIn('"chat_template_kwargs"', self.enhancer_source)

    def test_hf_text_enhancer_never_falls_back_to_raw_reasoning(self):
        self.assertNotIn("clean_model_output(raw_text, OutputCleanConfig(mode=\"prompt\")) or raw_text", self.enhancer_source)
        self.assertIn("Streaming produced no usable prompt after removing reasoning", self.enhancer_source)
        self.assertIn("Generation produced no usable prompt after removing reasoning", self.enhancer_source)
        self.assertIn("no usable prompt after removing reasoning", self.enhancer_source)

    def test_hf_multimodal_runtime_passes_thinking_as_template_variable(self):
        calls = [node for node in ast.walk(self.runtime_tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template"]
        self.assertTrue(calls, "HF runtime must call apply_chat_template")
        self.assertIn('chat_kwargs["enable_thinking"] = thinking_enabled', self.runtime_source)

    def test_hf_runtime_cleans_and_rejects_reasoning_only_before_cache(self):
        self.assertIn('mode="text",', self.runtime_source)
        self.assertIn('strip_code_fences=False,', self.runtime_source)
        self.assertIn('strip_role_prefixes=False,', self.runtime_source)
        self.assertIn('strip_json_wrappers=False,', self.runtime_source)
        self.assertIn('strip_leading_preamble=False,', self.runtime_source)
        self.assertIn('strip_planning=False,', self.runtime_source)
        self.assertIn("Generation returned no usable prompt after removing reasoning", self.runtime_source)
        text_config = OutputCleanConfig(
            mode="text",
            strip_think=True,
            strip_code_fences=False,
            strip_role_prefixes=False,
            strip_json_wrappers=False,
            strip_leading_preamble=False,
            strip_planning=False,
        )
        structured = '```json\n{"status":"ready","answer":42}\n```'
        with_reasoning = f"<think>private reasoning</think>\n{structured}"
        self.assertEqual(clean_model_output(with_reasoning, text_config), structured)
        self.assertEqual(clean_model_output("<think>reasoning only</think>", text_config), "")
        run_method = next(
            node for node in ast.walk(self.runtime_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        )
        cleaner_line = next(
            node.lineno for node in ast.walk(run_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "clean_model_output"
        )
        cache_line = next(
            node.lineno for node in ast.walk(run_method)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "PROMPT_CACHE"
        )
        self.assertLess(cleaner_line, cache_line, "empty cleaned output must be rejected before caching")


if __name__ == "__main__":
    unittest.main()
