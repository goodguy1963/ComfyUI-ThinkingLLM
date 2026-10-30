from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinkingllm_core import decision_api
from thinkingllm_core.decision_fields import (
    MAX_BOOLEAN_FIELDS,
    MAX_DECISION_FIELDS,
    MAX_INTEGER_FIELDS,
    field_routes,
    normalize_field_result,
    normalize_fields,
)


def load_router_module():
    module_name = "thinkingllm_decision_fields_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "nodes" / "decision_router.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def field_values():
    values = [("", "BOOLEAN", "", 0.5, 0, 0) for _ in range(MAX_BOOLEAN_FIELDS)]
    values += [("", "INT", "", 0.5, 0, 10) for _ in range(MAX_INTEGER_FIELDS)]
    values[0] = ("is_fox", "BOOLEAN", "Is a fox visible?", 0.6, 0, 0)
    values[1] = ("is_dog", "BOOLEAN", "Is a dog visible?", 0.5, 0, 0)
    values[MAX_BOOLEAN_FIELDS] = ("people_count", "INT", "How many people are visible?", 0.5, 0, 2)
    return values


def normalized_fields():
    return normalize_fields(field_values())


class DecisionFieldsSchemaTests(unittest.TestCase):
    def test_fields_keep_stable_slots_and_build_bounded_options(self):
        fields = normalized_fields()
        self.assertEqual([field["id"] for field in fields], ["field_01", "field_02", "field_09"])
        self.assertEqual([field["slot"] for field in fields], [0, 1, 8])
        self.assertEqual([option["value"] for option in fields[0]["options"]], ["true", "false"])
        self.assertEqual([option["value"] for option in fields[2]["options"]], ["0", "1", "2", ">max"])
        self.assertEqual(len(fields), 3)

    def test_bool_threshold_is_applied_to_true_probability(self):
        field = normalized_fields()[0]
        result = normalize_field_result(field, "false", {"true": 0.7, "false": 0.3})
        self.assertIs(result["value"], True)
        self.assertEqual(result["choice"], "false")

    def test_integer_overflow_is_clamped_and_marked(self):
        field = normalized_fields()[2]
        probabilities = {option["value"]: 0.0 for option in field["options"]}
        probabilities[">max"] = 1.0
        result = normalize_field_result(field, ">max", probabilities)
        self.assertEqual(result["value"], 2)
        self.assertEqual(result["range_status"], "above_max")

    def test_integer_exact_values_are_native_ints(self):
        field = normalized_fields()[2]
        probabilities = {option["value"]: 0.0 for option in field["options"]}
        probabilities["1"] = 1.0
        result = normalize_field_result(field, "1", probabilities)
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["range_status"], "exact")
        self.assertIs(type(result["value"]), int)

    def test_invalid_fields_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "needs a question"):
            normalize_fields([("is_fox", "BOOLEAN", "", 0.5, 0, 0)])
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_fields([
                ("is_fox", "BOOLEAN", "Fox visible?", 0.5, 0, 0),
                ("IS_FOX", "BOOLEAN", "Another question?", 0.5, 0, 0),
            ])
        with self.assertRaisesRegex(ValueError, "supports at most 52 choices"):
            normalize_fields([("count", "INT", "Count?", 0.5, 0, 51)])
        with self.assertRaisesRegex(ValueError, "at most 8 BOOLEAN"):
            normalize_fields([
                (f"bool_{index}", "BOOLEAN", f"Condition {index}?", 0.5, 0, 0)
                for index in range(MAX_BOOLEAN_FIELDS + 1)
            ])


class DecisionFieldsApiTests(unittest.TestCase):
    def test_openai_request_batches_multiple_questions_and_image_once(self):
        fields = normalized_fields()
        payload = decision_api.build_fields_decision_request(
            "openai_decisions", "gpt-6-luna", "image", None,
            "data:image/png;base64,AA==", "Inspect this image.", fields,
        )
        self.assertEqual(len(payload["questions"]), 3)
        self.assertEqual(payload["questions"][0]["name"], "field_01")
        self.assertEqual(payload["questions"][0]["choices"][0]["value"], "true")
        self.assertEqual(payload["questions"][2]["choices"][-1]["value"], ">max")
        self.assertEqual(sum(part["type"] == "input_image" for part in payload["input"][0]["content"]), 1)

    def test_openrouter_request_uses_named_systemone_fields(self):
        fields = normalized_fields()
        payload = decision_api.build_fields_decision_request(
            "openrouter_decisions", "typesafe/jev", "text", "two people", None, "Shared instruction.", fields,
        )
        self.assertEqual(set(payload["questions"]), {"field_01", "field_02", "field_09"})
        self.assertEqual(payload["questions"]["field_01"]["type"], "choice")
        self.assertEqual(payload["questions"]["field_09"]["criteria"][">max"], "The integer is greater than 2.")

    def test_openai_and_openrouter_answer_schemas_normalize_by_field_id(self):
        fields = normalized_fields()
        openai_answers = []
        openrouter_answers = {}
        for field in fields:
            values = [str(option["value"]) for option in field["options"]]
            chosen = values[0]
            probs = {value: (1.0 if value == chosen else 0.0) for value in values}
            openai_answers.append({
                "name": field["id"], "type": "choice", "choice": chosen, "confidence": 0.9,
                "probabilities": [{"value": value, "probability": probability} for value, probability in probs.items()],
            })
            openrouter_answers[field["id"]] = {
                "type": "choice", "choice": chosen, "confidence": 0.9, "probabilities": probs,
            }
        openai = decision_api.normalize_fields_decision_response("openai_decisions", {"answers": openai_answers}, fields)
        openrouter = decision_api.normalize_fields_decision_response("openrouter_decisions", {"answers": openrouter_answers}, fields)
        self.assertEqual(openai, openrouter)
        self.assertEqual(openai["field_01"][0], "true")

    def test_provider_missing_or_refused_field_fails_closed(self):
        fields = normalized_fields()
        with self.assertRaisesRegex(RuntimeError, "did not return a choice"):
            decision_api.normalize_fields_decision_response("openai_decisions", {"answers": []}, fields)
        with self.assertRaisesRegex(RuntimeError, "did not return a choice"):
            decision_api.normalize_fields_decision_response(
                "openrouter_decisions",
                {"answers": {"field_01": {"type": "refusal"}}},
                fields,
            )


class DecisionFieldsNodeTests(unittest.TestCase):
    def test_node_contract_uses_dynamic_output_slots_and_list_batch_outputs(self):
        module = load_router_module()
        image_node = module.ThinkingLLMDecisionFieldsImage
        text_node = module.ThinkingLLMDecisionFieldsText
        expected_types = ("STRING", "STRING") + ("*",) * MAX_DECISION_FIELDS
        self.assertEqual(image_node.RETURN_TYPES, expected_types)
        self.assertEqual(text_node.RETURN_TYPES, expected_types)
        self.assertEqual(len(image_node.RETURN_NAMES), MAX_DECISION_FIELDS + 2)
        self.assertEqual(image_node.RETURN_NAMES[0], "decision_json")
        self.assertEqual(image_node.RETURN_NAMES[1], "decision")
        self.assertEqual(image_node.OUTPUT_IS_LIST, (False, False) + (True,) * MAX_DECISION_FIELDS)
        self.assertEqual(image_node.INPUT_TYPES()["required"]["image"][0], "IMAGE")
        self.assertEqual(text_node.INPUT_TYPES()["required"]["text"][0], "STRING")
        self.assertEqual(image_node.INPUT_TYPES()["required"]["decision_output_slots"][1]["default"], "[]")
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["ThinkingLLM_DecisionFieldsImage"],
            image_node,
        )
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["ThinkingLLM_DecisionFieldsText"],
            text_node,
        )

    def test_readable_decision_output_is_multiline_without_probability_noise(self):
        module = load_router_module()
        result = module._DecisionFieldsBase._format_decision_text("image", [{
            "image_index": 0,
            "fields": [
                {"name": "dog_visible", "value": True, "probabilities": {"true": 0.9670209}},
                {"name": "person_count", "value": 2, "range_status": "exact"},
                {"name": "upper_count", "value": 3, "range_status": "above_max"},
            ],
        }])
        self.assertEqual(result, "Decision results (image)\nImage 1\n  dog_visible: true\n  person_count: 2\n  upper_count: 3 (above maximum)")
        self.assertNotIn("0.967", result)

    def test_readable_decision_output_handles_text_batches_and_empty_batches(self):
        module = load_router_module()
        text = module._DecisionFieldsBase._format_decision_text("text", [
            {"text_index": 0, "fields": [{"name": "is_fox", "value": False}]},
            {"text_index": 1, "fields": [{"name": "is_fox", "value": True}]},
        ])
        self.assertEqual(text, "Decision results (text)\nText 1\n  is_fox: false\nText 2\n  is_fox: true")
        self.assertEqual(module._DecisionFieldsBase._format_decision_text("text", []), "No text items were evaluated.")

    def test_text_batch_returns_independent_boolean_and_integer_values(self):
        module = load_router_module()
        node = module.ThinkingLLMDecisionFieldsText()
        node._load_model = lambda *_args, **_kwargs: setattr(node, "llm", object())
        node.clear = mock.Mock()

        def fake_probabilities(_llm, text, instruction, routes):
            names = [route["name"] for route in routes]
            if names == ["true", "false"]:
                is_first = text == "first"
                if "fox" in instruction:
                    return [0.8, 0.2] if is_first else [0.3, 0.7]
                return [0.7, 0.3] if is_first else [0.2, 0.8]
            if ">max" in names:
                return [0.0, 0.0, 1.0, 0.0] if text == "first" else [0.0, 0.0, 0.0, 1.0]
            raise AssertionError(f"Unexpected choices: {names}")

        with mock.patch.object(module, "openjev_text_probabilities", side_effect=fake_probabilities):
            result = node.evaluate_text(
                ["first", "second"],
                "OpenJev 27B · Q4_K_M · non-commercial",
                "Evaluate each property.",
                keep_model_loaded=True,
                boolean_01_name="is_fox",
                boolean_01_question="Is a fox visible?",
                boolean_02_name="is_dog",
                boolean_02_question="Is a dog visible?",
                integer_01_name="people_count",
                integer_01_question="How many people are visible?",
                integer_01_minimum=0,
                integer_01_maximum=2,
                decision_output_slots='["integer_01", "boolean_02", "boolean_01"]',
            )

        payload = json.loads(result[0])
        self.assertEqual([item["text_index"] for item in payload["decisions"]], [0, 1])
        self.assertEqual([field["value"] for field in payload["decisions"][0]["fields"]], [True, True, 2])
        self.assertEqual(result[1], "Decision results (text)\nText 1\n  is_fox: true\n  is_dog: true\n  people_count: 2\nText 2\n  is_fox: false\n  is_dog: false\n  people_count: 2 (above maximum)")
        self.assertEqual(result[2], [2, 2])
        self.assertEqual(result[3], [True, False])
        self.assertEqual(result[4], [True, False])
        self.assertEqual(payload["decisions"][1]["fields"][2]["range_status"], "above_max")
        self.assertEqual(len(result), 5)
        node.clear.assert_not_called()

    def test_image_batch_returns_values_in_original_order(self):
        module = load_router_module()
        node = module.ThinkingLLMDecisionFieldsImage()
        node._load_model = lambda *_args, **_kwargs: setattr(node, "llm", object())
        node.clear = mock.Mock()
        images = torch.stack([torch.zeros((2, 2, 3)), torch.ones((2, 2, 3))])

        def fake_probabilities(_llm, _handler, image_data_url, _instruction, _routes):
            return [0.9, 0.1] if image_data_url == "first-image" else [0.1, 0.9]

        with (
            mock.patch.object(module, "_encode_image", side_effect=lambda image: "first-image" if float(image[0, 0, 0]) == 0 else "second-image"),
            mock.patch.object(module, "openjev_probabilities", side_effect=fake_probabilities),
        ):
            result = node.evaluate_images(
                images,
                "OpenJev 27B · Q4_K_M · non-commercial",
                "Check the visible subject.",
                keep_model_loaded=True,
                boolean_01_name="is_fox",
                boolean_01_question="Is a fox visible?",
                decision_output_slots='["boolean_01", "boolean_02"]',
            )

        self.assertEqual(result[1], "Decision results (image)\nImage 1\n  is_fox: true\nImage 2\n  is_fox: false")
        self.assertEqual(result[2], [True, False])
        payload = json.loads(result[0])
        self.assertEqual(payload["input_type"], "image")
        self.assertEqual([item["image_index"] for item in payload["decisions"]], [0, 1])
        self.assertIsInstance(result[3], module.ExecutionBlocker)
        self.assertEqual(len(result), 4)
        node.clear.assert_not_called()

    def test_empty_batch_blocks_typed_outputs_without_loading_model(self):
        module = load_router_module()
        node = module.ThinkingLLMDecisionFieldsText()
        node._load_model = mock.Mock(side_effect=AssertionError("empty batch must not load a model"))
        node.clear = mock.Mock()
        result = node.evaluate_text(
            [], "Clef-Flash 9B · Q4_K_M", "Check the input.",
            boolean_01_name="has_text", boolean_01_question="Does the input contain text?",
        )
        self.assertEqual(json.loads(result[0])["decisions"], [])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1], "No text items were evaluated.")
        self.assertIsInstance(result[2], module.ExecutionBlocker)
        node._load_model.assert_not_called()
        node.clear.assert_called_once_with()

    def test_api_node_uses_one_multifield_request_per_input(self):
        module = load_router_module()
        node = module.ThinkingLLMDecisionFieldsText()
        profile = {
            "name": "Test OpenAI", "provider": "OpenAI", "decision_protocol": "openai_decisions",
            "decision_endpoint": "https://api.openai.com/v1/decisions", "allow_images": True,
        }
        api_module = types.ModuleType("openai_compatible_api")
        api_module._load_api_profiles = lambda: {"Test OpenAI": profile}
        api_module._credential_value = lambda _profile: "test-secret"
        api_module._NO_REDIRECT_OPENER = object()
        api_module._image_to_data_url = lambda *_args: "data:image/png;base64,AA=="
        api_module._validate_input_text = lambda *_args: None
        previous_api_module = sys.modules.get("openai_compatible_api")
        try:
            with mock.patch.dict(sys.modules, {"openai_compatible_api": api_module}):
                with (
                    mock.patch.object(decision_api, "decision_models_for_profile", return_value=[{"id": "gpt-6-luna"}]),
                    mock.patch.object(
                        decision_api,
                        "request_decision_fields",
                        side_effect=[
                            {
                                "field_01": ("true", {"true": 0.9, "false": 0.1}, 0.9),
                                "field_09": ("2", {"0": 0.0, "1": 0.0, "2": 1.0, ">max": 0.0}, 0.9),
                            },
                            {
                                "field_01": ("false", {"true": 0.1, "false": 0.9}, 0.9),
                                "field_09": (">max", {"0": 0.0, "1": 0.0, "2": 0.0, ">max": 1.0}, 0.9),
                            },
                        ],
                    ) as request_mock,
                ):
                    result = node.evaluate_text(
                        ["first", "second"], "unused local", "Inspect each item.",
                        decision_backend="API", api_profile="Test OpenAI", api_model="gpt-6-luna",
                        boolean_01_name="is_fox", boolean_01_question="Is a fox visible?",
                        integer_01_name="people_count", integer_01_question="How many people?",
                        integer_01_minimum=0, integer_01_maximum=2,
                    )
        finally:
            if previous_api_module is None:
                sys.modules.pop("openai_compatible_api", None)
            else:
                sys.modules["openai_compatible_api"] = previous_api_module

        self.assertEqual(request_mock.call_count, 2)
        self.assertEqual(result[1], "Decision results (text)\nText 1\n  is_fox: true\n  people_count: 2\nText 2\n  is_fox: false\n  people_count: 2 (above maximum)")
        self.assertEqual(result[2], [True, False])
        self.assertEqual(result[3], [2, 2])
        self.assertEqual(json.loads(result[0])["provider"], "OpenAI")

    def test_stale_or_invalid_dynamic_output_schema_fails_before_inference(self):
        module = load_router_module()
        node = module.ThinkingLLMDecisionFieldsText()
        node._load_model = mock.Mock(side_effect=AssertionError("invalid output schema must fail before loading"))
        node.clear = mock.Mock()

        with self.assertRaisesRegex(ValueError, "out of sync"):
            node.evaluate_text(
                "A fox is visible.", "Clef-Flash 9B · Q4_K_M", "Check the text.",
                boolean_01_name="is_fox", boolean_01_question="Is a fox visible?",
                integer_01_name="count", integer_01_question="How many foxes?",
                decision_output_slots='["boolean_01"]',
            )
        node._load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
