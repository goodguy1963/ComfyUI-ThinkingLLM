from __future__ import annotations

import asyncio
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


ROUTES = [
    {"name": "fox", "description": "A fox."},
    {"name": "dog", "description": "A domestic dog."},
]


class DecisionApiAdapterTests(unittest.TestCase):
    def test_openai_request_uses_native_choice_and_input_image_schema(self):
        payload = decision_api.build_decision_request(
            "openai_decisions", "gpt-6-luna", "image", None,
            "data:image/png;base64,AA==", "Pick the best route.", ROUTES,
        )
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["questions"][0]["type"], "choice")
        self.assertEqual(payload["questions"][0]["choices"][0], {"value": "fox", "description": "A fox."})
        self.assertEqual(payload["input"][0]["content"][1], {"type": "input_image", "image_url": "data:image/png;base64,AA=="})

    def test_openrouter_request_uses_native_systemone_schema(self):
        payload = decision_api.build_decision_request(
            "openrouter_decisions", "cloudflare/clef-flash", "image", None,
            "data:image/png;base64,AA==", "Pick the best route.", ROUTES,
        )
        self.assertEqual(payload["state"], [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}])
        self.assertEqual(payload["questions"]["route"]["criteria"], {"fox": "A fox.", "dog": "A domestic dog."})

    def test_openai_and_openrouter_response_schemas_normalize(self):
        openai = {"answers": [{"name": "route", "type": "choice", "choice": "dog", "confidence": 0.8,
                               "probabilities": [{"value": "fox", "probability": 0.2}, {"value": "dog", "probability": 0.8}]}]}
        openrouter = {"answers": {"route": {"type": "choice", "choice": "dog", "confidence": 0.8,
                                             "probabilities": {"fox": 0.2, "dog": 0.8}}}}
        expected = ("dog", {"fox": 0.2, "dog": 0.8}, 0.8)
        self.assertEqual(decision_api.normalize_decision_response("openai_decisions", openai, ["fox", "dog"]), expected)
        self.assertEqual(decision_api.normalize_decision_response("openrouter_decisions", openrouter, ["fox", "dog"]), expected)

    def test_refusal_unknown_routes_and_malformed_probabilities_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "did not return a route"):
            decision_api.normalize_decision_response("openai_decisions", {"answers": [{"name": "route", "type": "refusal"}]}, ["fox", "dog"])
        with self.assertRaisesRegex(RuntimeError, "unknown route"):
            decision_api.normalize_decision_response("openrouter_decisions", {"answers": {"route": {"type": "choice", "choice": "cat", "probabilities": {"fox": 0.2, "dog": 0.8}}}}, ["fox", "dog"])
        with self.assertRaisesRegex(RuntimeError, "invalid route probability"):
            decision_api.normalize_decision_response("openrouter_decisions", {"answers": {"route": {"type": "choice", "choice": "fox", "probabilities": {"fox": float("nan"), "dog": 0.8}}}}, ["fox", "dog"])
        with self.assertRaisesRegex(RuntimeError, "duplicate route probabilities"):
            decision_api.normalize_decision_response(
                "openai_decisions",
                {"answers": [{"name": "route", "type": "choice", "choice": "fox", "probabilities": [
                    {"value": "fox", "probability": 0.5}, {"value": "fox", "probability": 0.4},
                    {"value": "dog", "probability": 0.1},
                ]}]},
                ["fox", "dog"],
            )

    def test_openrouter_model_catalog_filters_by_required_modality(self):
        response = {
            "data": [
                {"id": "decision/vision", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["decisions"]}},
                {"id": "decision/text", "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]}},
                {"id": "chat/vision", "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
            ]
        }
        opener = mock.Mock()
        http_response = mock.MagicMock()
        http_response.read.return_value = json.dumps(response).encode()
        http_response.__enter__.return_value = http_response
        opener.open.return_value = http_response
        decision_api._MODEL_CACHE.clear()
        with mock.patch.object(decision_api, "_request_json", return_value=response):
            image_models = decision_api.openrouter_decision_models("image", now=100)
            text_models = decision_api.openrouter_decision_models("text", now=100)
        self.assertEqual([model["id"] for model in image_models], ["decision/vision"])
        self.assertEqual([model["id"] for model in text_models], ["decision/vision", "decision/text"])

    def test_profile_allowlist_restricts_api_model_choices(self):
        with mock.patch.object(decision_api, "openrouter_decision_models", return_value=[{"id": "a"}, {"id": "b"}]):
            models = decision_api.decision_models_for_profile({"decision_protocol": "openrouter_decisions", "allowed_models": ["b"]}, "text")
        self.assertEqual(models, [{"id": "b"}])

    def test_profile_image_policy_hides_vision_models(self):
        with mock.patch.object(decision_api, "openrouter_decision_models") as catalog:
            models = decision_api.decision_models_for_profile(
                {"decision_protocol": "openrouter_decisions", "allow_images": False}, "image",
            )
        self.assertEqual(models, [])
        catalog.assert_not_called()


def load_router_module():
    module_name = "thinkingllm_decision_router_api_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "nodes" / "decision_router.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DecisionApiRouterTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "name": "Test OpenAI",
            "provider": "OpenAI",
            "decision_protocol": "openai_decisions",
            "decision_endpoint": "https://api.openai.com/v1/decisions",
            "allow_images": True,
            "max_image_pixels": 100,
            "max_image_bytes": 100000,
        }
        self.api_module = types.ModuleType("openai_compatible_api")
        self.api_module._load_api_profiles = lambda: {"Test OpenAI": self.profile}
        self.api_module._credential_value = lambda _profile: "test-secret"
        self.api_module._image_to_data_url = lambda _profile, _image: "data:image/png;base64,AA=="
        self.api_module._NO_REDIRECT_OPENER = object()
        def validate_input_text(profile, system_prompt, prompt):
            if system_prompt is not None and not isinstance(system_prompt, str):
                raise ValueError("system_prompt must be a string.")
            if prompt is not None and not isinstance(prompt, str):
                raise ValueError("prompt must be a string.")
            total = len(system_prompt or "") + len(prompt or "")
            limit = profile.get("max_input_chars", 1_000_000)
            if total > limit:
                raise ValueError(f"Combined input length exceeds server-side max_input_chars={limit}.")
        self.api_module._validate_input_text = validate_input_text
        self.previous_api_module = sys.modules.get("openai_compatible_api")
        sys.modules["openai_compatible_api"] = self.api_module

    def tearDown(self):
        if self.previous_api_module is None:
            sys.modules.pop("openai_compatible_api", None)
        else:
            sys.modules["openai_compatible_api"] = self.previous_api_module

    def test_registers_model_catalog_route_and_returns_filtered_models(self):
        registered = {}

        def route_decorator(path):
            return lambda handler: registered.setdefault(path, handler)

        fake_server = types.ModuleType("server")
        fake_server.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=types.SimpleNamespace(get=route_decorator)),
        )
        router_module = load_router_module()
        with mock.patch.dict(sys.modules, {"server": fake_server}):
            with mock.patch.object(
                decision_api,
                "decision_models_for_profile",
                return_value=[{"id": "gpt-6-luna", "vision": True}],
            ):
                router_module._register_decision_model_route()
                handler = registered["/thinkingllm/decision/models"]
                response = asyncio.run(handler(types.SimpleNamespace(
                    query={"profile": "Test OpenAI", "input_kind": "image"},
                )))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.text)["models"], [{"id": "gpt-6-luna", "vision": True}])

    def test_api_text_batch_routes_without_loading_local_model(self):
        router_module = load_router_module()
        node = router_module.ThinkingLLMDecisionRouterText()
        node._load_model = mock.Mock(side_effect=AssertionError("API route must not load a local model"))
        node.llm = mock.Mock()
        node.current_signature = ("previously-retained-local-model",)
        route_fields = {"route_01_name": "fox", "route_02_name": "dog"}
        api_args = {"decision_backend": "API", "api_profile": "Test OpenAI", "api_model": "gpt-6-luna"}
        with (
            mock.patch.object(decision_api, "decision_models_for_profile", return_value=[{"id": "gpt-6-luna", "vision": True}]),
            mock.patch.object(decision_api, "request_decision", side_effect=[("fox", {"fox": 0.9, "dog": 0.1}, 0.9), ("dog", {"fox": 0.1, "dog": 0.9}, 0.9)]),
        ):
            result = node.route_text(["fox text", "dog text"], "unused local", "Choose.", **api_args, **route_fields)
        payload = json.loads(result[0])
        self.assertEqual(payload["provider"], "OpenAI")
        self.assertEqual([item["route"] for item in payload["decisions"]], ["fox", "dog"])
        self.assertEqual(result[1], ["fox text"])
        self.assertEqual(result[2], ["dog text"])
        node._load_model.assert_not_called()
        self.assertIsNone(node.llm)
        self.assertIsNone(node.current_signature)

    def test_api_image_uses_profile_limited_data_url_and_blocks_empty_routes(self):
        router_module = load_router_module()
        node = router_module.ThinkingLLMDecisionRouter()
        image = torch.zeros((1, 2, 2, 3))
        route_fields = {"route_01_name": "fox", "route_02_name": "dog"}
        api_args = {"decision_backend": "API", "api_profile": "Test OpenAI", "api_model": "gpt-6-luna"}
        with (
            mock.patch.object(decision_api, "decision_models_for_profile", return_value=[{"id": "gpt-6-luna", "vision": True}]),
            mock.patch.object(decision_api, "request_decision", return_value=("dog", {"fox": 0.1, "dog": 0.9}, 0.9)) as request_mock,
        ):
            result = node.route_images(image, "unused local", "Choose.", **api_args, **route_fields)
        self.assertEqual(json.loads(result[0])["decisions"][0]["route"], "dog")
        self.assertEqual(request_mock.call_args.args[4], "data:image/png;base64,AA==")
        self.assertIsInstance(result[1], router_module.ExecutionBlocker)
        self.assertEqual(tuple(result[2].shape), (1, 2, 2, 3))

    def test_api_mode_rejects_unknown_image_model_before_request(self):
        router_module = load_router_module()
        node = router_module.ThinkingLLMDecisionRouter()
        with (
            mock.patch.object(decision_api, "decision_models_for_profile", return_value=[{"id": "text-only", "vision": False}]),
            mock.patch.object(decision_api, "request_decision") as request_mock,
        ):
            with self.assertRaisesRegex(ValueError, "image-capable"):
                node.route_images(
                    torch.zeros((1, 1, 1, 3)), "unused", "Choose.",
                    decision_backend="API", api_profile="Test OpenAI", api_model="text-only",
                    route_01_name="a", route_02_name="b",
                )
        request_mock.assert_not_called()

    def test_api_mode_enforces_profile_input_limit_before_request(self):
        router_module = load_router_module()
        node = router_module.ThinkingLLMDecisionRouterText()
        self.profile["max_input_chars"] = 20
        with (
            mock.patch.object(decision_api, "decision_models_for_profile", return_value=[{"id": "gpt-6-luna"}]),
            mock.patch.object(decision_api, "request_decision") as request_mock,
        ):
            with self.assertRaisesRegex(ValueError, "exceeds server-side max_input_chars"):
                node.route_text(
                    "cat", "unused", "Choose.",
                    decision_backend="API", api_profile="Test OpenAI", api_model="gpt-6-luna",
                    route_01_name="fox", route_01_description="long route description",
                    route_02_name="dog",
                )
        request_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
