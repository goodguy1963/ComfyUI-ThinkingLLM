"""Security-focused tests for the ThinkingLLM OpenAI-compatible API node.

Run from the repository root:
    python tests/test_openai_api_security.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from nodes import openai_compatible_api as api_node


class OpenAICompatibleAPISecurityTests(unittest.TestCase):
    def test_workflow_inputs_do_not_expose_credentials_or_endpoint(self):
        inputs = api_node.ThinkingLLMOpenAICompatibleAPI.INPUT_TYPES()
        names = set(inputs.get("required", {})) | set(inputs.get("optional", {}))

        self.assertIn("api_profile", names)
        self.assertNotIn("api_key", names)
        self.assertNotIn("api_key_env", names)
        self.assertNotIn("base_url", names)
        self.assertNotIn("headers", names)

    def test_authenticated_profiles_require_https(self):
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            api_node._normalize_profile(
                "Unsafe",
                {
                    "base_url": "http://example.test/v1",
                    "auth": "bearer_env",
                    "api_key_env": "TEST_KEY",
                },
            )

    def test_profile_rejects_literal_secret_and_header_fields(self):
        for forbidden in ("api_key", "token", "authorization", "headers", "secret"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    api_node._normalize_profile(
                        "Unsafe",
                        {
                            "base_url": "https://example.test/v1",
                            "auth": "bearer_env",
                            "api_key_env": "TEST_KEY",
                            forbidden: "do-not-store-this-here",
                        },
                    )

    def test_extra_body_cannot_override_core_fields(self):
        protected = (
            "model",
            "messages",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "seed",
            "enable_thinking",
            "thinking_budget",
        )
        for field in protected:
            with self.subTest(field=field):
                payload = json.dumps({field: "override"})
                with self.assertRaisesRegex(ValueError, "protected request fields"):
                    api_node._parse_extra_json(payload, allowed=True)

    def test_extra_body_is_disabled_by_default(self):
        profile = api_node._normalize_profile(
            "Locked",
            {
                "base_url": "https://example.test/v1",
                "auth": "bearer_env",
                "api_key_env": "TEST_KEY",
            },
        )
        self.assertFalse(profile["allow_extra_body"])
        with self.assertRaisesRegex(ValueError, "disabled"):
            api_node._parse_extra_json('{"top_k": 40}', allowed=profile["allow_extra_body"])

    def test_server_side_model_and_token_limits_are_enforced(self):
        profile = api_node._normalize_profile(
            "Production",
            {
                "base_url": "https://example.test/v1",
                "auth": "bearer_env",
                "api_key_env": "TEST_KEY",
                "allowed_models": ["approved/model"],
                "max_tokens_limit": 2048,
            },
        )

        self.assertEqual(api_node._validate_model(profile, "approved/model"), "approved/model")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            api_node._validate_model(profile, "other/model")

        self.assertEqual(api_node._validate_max_tokens(profile, 2048), 2048)
        with self.assertRaisesRegex(ValueError, "exceeds server-side limit"):
            api_node._validate_max_tokens(profile, 2049)

    def test_missing_key_does_not_expose_env_name_in_client_error(self):
        env_name = "THINKINGLLM_TEST_SECRET_KEY"
        profile = api_node._normalize_profile(
            "Production",
            {
                "base_url": "https://example.test/v1",
                "auth": "bearer_env",
                "api_key_env": env_name,
            },
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env_name, None)
            with self.assertRaises(RuntimeError) as ctx:
                api_node._authorization_headers(profile)
        self.assertNotIn(env_name, str(ctx.exception))

    def test_builtin_profiles_can_be_disabled_for_production(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_profile_file = str(Path(tmpdir) / "profiles.json")
            with mock.patch.dict(
                os.environ,
                {
                    api_node.DISABLE_BUILTINS_ENV: "1",
                    api_node.PROFILE_FILE_ENV: missing_profile_file,
                },
                clear=False,
            ):
                self.assertEqual(api_node._load_api_profiles(), {})

    def test_custom_profile_file_contains_configuration_not_secret(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "Production": {
                                "provider": "Test",
                                "base_url": "https://example.test/v1",
                                "auth": "bearer_env",
                                "api_key_env": "TEST_API_KEY",
                                "allowed_models": ["approved/model"],
                                "max_tokens_limit": 1024,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {
                    api_node.DISABLE_BUILTINS_ENV: "1",
                    api_node.PROFILE_FILE_ENV: str(profile_path),
                },
                clear=False,
            ):
                profiles = api_node._load_api_profiles()

            self.assertEqual(list(profiles), ["Production"])
            self.assertEqual(profiles["Production"]["api_key_env"], "TEST_API_KEY")
            self.assertNotIn("api_key", profiles["Production"])


if __name__ == "__main__":
    unittest.main()
