import unittest

from thinkingllm_core.prompt_contracts import (
    QWEN_MAX_REASONING_BUDGET_TOKENS,
    resolve_qwen_output_token_plan,
    resolve_qwen_thinking_mode,
)


class QwenThinkingBudgetTests(unittest.TestCase):
    def test_default_plan_scales_answer_reserve_with_available_output(self):
        self.assertEqual(
            resolve_qwen_output_token_plan(3000),
            (3000, 2000, 1000),
        )

    def test_plan_accounts_for_prompt_and_context_overhead(self):
        self.assertEqual(
            resolve_qwen_output_token_plan(
                4096,
                prompt_tokens=3000,
                context_window=3500,
            ),
            (372, 512, 0),
        )

    def test_plan_caps_reasoning_only_when_requested(self):
        available, answer_reserve, reasoning_budget = resolve_qwen_output_token_plan(
            8192,
            max_reasoning_tokens=QWEN_MAX_REASONING_BUDGET_TOKENS,
        )
        self.assertEqual((available, answer_reserve), (8192, 5462))
        self.assertEqual(reasoning_budget, QWEN_MAX_REASONING_BUDGET_TOKENS)

    def test_default_thinking_gate_disables_at_3000_tokens(self):
        self.assertFalse(resolve_qwen_thinking_mode(True, 3000, quiet=True))

    def test_default_thinking_gate_allows_budget_at_3072_tokens(self):
        self.assertTrue(resolve_qwen_thinking_mode(True, 3072, quiet=True))

    def test_explicit_enforced_budget_can_use_shorter_reasoning(self):
        self.assertTrue(
            resolve_qwen_thinking_mode(
                True,
                3000,
                min_reasoning_tokens=512,
                answer_reserve_tokens=2000,
                quiet=True,
            )
        )

    def test_nonpositive_output_budget_disables_thinking(self):
        self.assertFalse(resolve_qwen_thinking_mode(True, 0, quiet=True))


if __name__ == "__main__":
    unittest.main()
