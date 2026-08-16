from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
MODULE_PATH = PKG / 'nodes' / 'openai_compatible_api.py'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

output_cleaner = types.ModuleType('AILab_OutputCleaner')
class OutputCleanConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
output_cleaner.OutputCleanConfig = OutputCleanConfig
output_cleaner.clean_model_output = lambda text, config=None: text.replace('<think>x</think>', '').strip()
sys.modules['AILab_OutputCleaner'] = output_cleaner

stream_display = types.ModuleType('AILab_StreamDisplay')
stream_display.compose_streamed_model_output = lambda reasoning, content: (
    f'<think>\n{reasoning}\n</think>\n\n{content}' if reasoning and content
    else f'<think>\n{reasoning}\n</think>' if reasoning else content
)
stream_display.extract_stream_token = lambda chunk: {
    'reasoning': ((chunk.get('choices') or [{}])[0].get('delta') or {}).get('reasoning_content', ''),
    'content': ((chunk.get('choices') or [{}])[0].get('delta') or {}).get('content', ''),
}
stream_display.get_thinking_stream_display = lambda: None
sys.modules['AILab_StreamDisplay'] = stream_display

SPEC = importlib.util.spec_from_file_location('thinkingllm_openai_compatible_api_test', MODULE_PATH)
assert SPEC and SPEC.loader
api_node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_node)

class SecurityTests(unittest.TestCase):
    def profile(self, **overrides):
        raw = {
            'provider': 'Test',
            'base_url': 'https://example.test/v1',
            'auth': 'bearer_env',
            'api_key_env': 'TEST_API_KEY',
        }
        raw.update(overrides)
        return api_node._normalize_profile('Production', raw)

    def test_no_workflow_secret_or_endpoint_inputs(self):
        inputs = api_node.ThinkingLLMOpenAICompatibleAPI.INPUT_TYPES()
        names = set(inputs.get('required', {})) | set(inputs.get('optional', {}))
        self.assertIn('api_profile', names)
        self.assertIn('image', names)
        self.assertIn('image_url', names)
        self.assertNotIn('video', names)
        self.assertNotIn('video_url', names)
        for forbidden in ('api_key', 'api_key_env', 'base_url', 'headers'):
            self.assertNotIn(forbidden, names)

    def test_authenticated_profiles_require_https(self):
        with self.assertRaisesRegex(ValueError, 'must use HTTPS'):
            api_node._normalize_profile('Unsafe', {
                'base_url': 'http://example.test/v1',
                'auth': 'bearer_env',
                'api_key_env': 'TEST_KEY',
            })

    def test_plain_http_non_loopback_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'Plain HTTP'):
            api_node._normalize_profile('LAN', {
                'base_url': 'http://192.168.1.10:8000/v1',
                'auth': 'none',
            })
        p = api_node._normalize_profile('LAN', {
            'base_url': 'http://192.168.1.10:8000/v1',
            'auth': 'none',
            'allow_insecure_http': True,
        })
        self.assertTrue(p['allow_insecure_http'])

    def test_loopback_http_allowed(self):
        p = api_node._normalize_profile('Local', {
            'base_url': 'http://127.0.0.1:8000/v1',
            'auth': 'none',
        })
        self.assertEqual(p['base_url'], 'http://127.0.0.1:8000/v1')

    def test_unknown_profile_fields_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            self.profile(max_token_limit=10)

    def test_deprecated_boolean_extra_body_switch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'deprecated allow_extra_body'):
            self.profile(allow_extra_body=True)

    def test_exact_extra_body_allowlist(self):
        p = self.profile(allowed_extra_body_fields=['reasoning'])
        parsed = api_node._parse_extra_json('{"reasoning":{"effort":"high"}}', allowed_fields=p['allowed_extra_body_fields'])
        self.assertIn('reasoning', parsed)
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            api_node._parse_extra_json('{"n":4}', allowed_fields=p['allowed_extra_body_fields'])
        with self.assertRaisesRegex(ValueError, 'protected request fields'):
            api_node._parse_extra_json('{"model":"evil"}', allowed_fields=['model'])

    def test_extra_json_size_limit(self):
        p = self.profile(allowed_extra_body_fields=['x'])
        payload = '{"x":"' + ('a' * api_node.MAX_EXTRA_BODY_JSON_CHARS) + '"}'
        with self.assertRaisesRegex(ValueError, 'safety limit'):
            api_node._parse_extra_json(payload, allowed_fields=p['allowed_extra_body_fields'])

    def test_nonfinite_extra_json_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            api_node._parse_extra_json('{"reasoning":NaN}', allowed_fields=['reasoning'])

    def test_model_and_limits(self):
        p = self.profile(allowed_models=['approved/model'], max_tokens_limit=2048, max_input_chars=20, max_timeout_seconds=30)
        self.assertEqual(api_node._validate_model(p, 'approved/model'), 'approved/model')
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            api_node._validate_model(p, 'other/model')
        with self.assertRaisesRegex(ValueError, 'control characters'):
            api_node._validate_model_identifier('bad\x1bmodel')
        with self.assertRaisesRegex(ValueError, 'exceeds server-side limit'):
            api_node._validate_max_tokens(p, 2049)
        with self.assertRaisesRegex(ValueError, 'max_input_chars'):
            api_node._validate_input_text(p, '1234567890', '12345678901')
        with self.assertRaisesRegex(ValueError, 'exceeds server-side limit'):
            api_node._validate_timeout(p, 31)

    def test_sampling_rejects_nan_and_wrong_types(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            api_node._validate_sampling(float('nan'), 0.9, 1, 0)
        with self.assertRaisesRegex(ValueError, 'seed must be an integer'):
            api_node._validate_sampling(0.5, 0.9, 1.2, 0)
        with self.assertRaisesRegex(ValueError, 'temperature must be a numeric'):
            api_node._validate_sampling('0.5', 0.9, 1, 0)

    def test_thinking_budget_has_server_side_cap(self):
        p = self.profile(max_thinking_budget=4096)
        self.assertEqual(api_node._validate_thinking_budget(p, 4096), 4096)
        with self.assertRaisesRegex(ValueError, 'exceeds server-side limit'):
            api_node._validate_thinking_budget(p, 4097)

    def test_allowed_models_requires_strings(self):
        with self.assertRaisesRegex(ValueError, 'list of strings'):
            self.profile(allowed_models=[123])

    def test_profile_booleans_are_strict(self):
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            self.profile(send_seed='false')
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            self.profile(allow_images='true')

    def test_environment_variable_name_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'environment-variable name'):
            self.profile(api_key_env='BAD-NAME')

    def test_missing_key_does_not_expose_env_name_in_client_error(self):
        p = self.profile(api_key_env='THINKINGLLM_TEST_SECRET_KEY')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('THINKINGLLM_TEST_SECRET_KEY', None)
            with self.assertRaises(RuntimeError) as ctx:
                api_node._authorization_headers(p)
        self.assertNotIn('THINKINGLLM_TEST_SECRET_KEY', str(ctx.exception))

    def test_redirects_are_rejected(self):
        handler = api_node._NoRedirectHandler()
        req = __import__('urllib.request').request.Request('https://example.test/v1/chat/completions')
        self.assertIsNone(handler.redirect_request(req, None, 302, 'Found', {}, 'https://evil.test/steal'))

    def test_terminal_copy_strips_controls(self):
        value = api_node._terminal_display_safe('hello\x1b[31mred\x07\rworld')
        self.assertNotIn('\x1b', value)
        self.assertNotIn('\x07', value)
        self.assertNotIn('\r', value)
        self.assertIn('[31mredworld', value)

    def test_openrouter_reasoning_details_are_captured(self):
        chunk = {
            'choices': [{
                'delta': {
                    'reasoning_details': [
                        {'type': 'reasoning.text', 'text': 'step one '},
                        {'type': 'reasoning.summary', 'summary': 'step two'},
                    ],
                    'content': 'answer',
                }
            }]
        }
        token = api_node._extract_stream_token_extended(chunk)
        self.assertEqual(token['reasoning'], 'step one step two')
        self.assertEqual(token['content'], 'answer')

    def test_stream_error_payload_detected(self):
        self.assertIn('429', api_node._provider_error_from_payload({'error': {'code': 429, 'message': 'rate limit'}}))

    def test_remote_node_always_reruns(self):
        self.assertTrue(math.isnan(api_node.ThinkingLLMOpenAICompatibleAPI.IS_CHANGED()))

    def test_image_support_is_fail_closed_for_custom_profiles(self):
        p = self.profile()
        self.assertFalse(p['allow_images'])
        self.assertFalse(p['allow_image_url'])
        with self.assertRaisesRegex(ValueError, 'Image input is disabled'):
            api_node._validate_image_url(p, 'https://example.com/image.png')

    def test_image_url_requires_image_capability_and_https(self):
        with self.assertRaisesRegex(ValueError, 'cannot enable allow_image_url'):
            self.profile(allow_image_url=True)
        p = self.profile(allow_images=True, allow_image_url=True)
        signed = 'https://example.com/image.png?signature=abc123'
        self.assertEqual(api_node._validate_image_url(p, signed), signed)
        with self.assertRaisesRegex(ValueError, 'public HTTPS'):
            api_node._validate_image_url(p, 'http://example.com/image.png')
        with self.assertRaisesRegex(ValueError, 'public HTTPS'):
            api_node._validate_image_url(p, 'data:image/png;base64,AAAA')

    def test_comfyui_image_is_encoded_as_png_base64(self):
        import numpy as np
        p = self.profile(allow_images=True, max_image_pixels=4, max_image_bytes=10000)
        image = np.zeros((1, 2, 2, 3), dtype=np.float32)
        image[..., 0] = 1.0
        data_url = api_node._image_to_data_url(p, image)
        self.assertTrue(data_url.startswith('data:image/png;base64,'))
        png = base64.b64decode(data_url.split(',', 1)[1])
        self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))

    def test_image_size_and_batch_limits_are_enforced(self):
        import numpy as np
        p = self.profile(allow_images=True, max_image_pixels=3, max_image_bytes=10000)
        with self.assertRaisesRegex(ValueError, 'max_image_pixels'):
            api_node._image_to_data_url(p, np.zeros((1, 2, 2, 3), dtype=np.float32))
        p2 = self.profile(allow_images=True)
        with self.assertRaisesRegex(ValueError, 'exactly one IMAGE'):
            api_node._image_to_data_url(p2, np.zeros((2, 1, 1, 3), dtype=np.float32))

    def test_image_and_image_url_are_alternatives(self):
        p = self.profile(allow_images=True, allow_image_url=True)
        with self.assertRaisesRegex(ValueError, 'either the IMAGE input or image_url'):
            api_node._build_user_content(p, 'describe', image=object(), image_url='https://example.com/x.png')

    def test_multimodal_content_uses_text_first_then_image_url(self):
        p = self.profile(allow_images=True, allow_image_url=True)
        content, supplied = api_node._build_user_content(p, 'describe', image_url='https://example.com/x.png')
        self.assertTrue(supplied)
        self.assertEqual(content[0], {'type': 'text', 'text': 'describe'})
        self.assertEqual(content[1]['type'], 'image_url')
        self.assertEqual(content[1]['image_url']['url'], 'https://example.com/x.png')

    def test_image_rejection_message_is_actionable(self):
        p = self.profile(allow_images=True)
        message = api_node._image_rejection_message(p, 'text-only/model', 400)
        self.assertIn('Image input was rejected', message)
        self.assertIn('vision-capable', message)

    def test_builtins_enable_standard_image_transport(self):
        profiles = api_node._load_api_profiles()
        self.assertTrue(profiles['OpenRouter']['allow_images'])
        self.assertTrue(profiles['OpenRouter']['allow_image_url'])

    def test_builtins_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {
                api_node.DISABLE_BUILTINS_ENV: '1',
                api_node.PROFILE_FILE_ENV: str(Path(tmp) / 'missing.json'),
            }, clear=False):
                self.assertEqual(api_node._load_api_profiles(), {})

    def test_invalid_custom_override_does_not_fall_back_to_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profiles.json'
            path.write_text(json.dumps({'profiles': {'OpenRouter': {
                'provider': 'OpenRouter',
                'base_url': 'https://openrouter.ai/api/v1',
                'auth': 'bearer_env',
                'api_key_env': 'OPENROUTER_API_KEY',
                'max_token_limit': 1
            }}}), encoding='utf-8')
            with mock.patch.dict(os.environ, {api_node.PROFILE_FILE_ENV: str(path)}, clear=False):
                profiles = api_node._load_api_profiles()
            self.assertNotIn('OpenRouter', profiles)

    def test_custom_profile_file_has_configuration_not_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profiles.json'
            path.write_text(json.dumps({'profiles': {'Production': {
                'provider': 'Test',
                'base_url': 'https://example.test/v1',
                'auth': 'bearer_env',
                'api_key_env': 'TEST_API_KEY',
                'allowed_models': ['approved/model'],
                'max_tokens_limit': 1024,
                'max_input_chars': 10000,
                'max_timeout_seconds': 60,
                'allow_images': True,
                'allow_image_url': False,
                'max_image_pixels': 16777216,
                'max_image_bytes': 20000000,
            }}}), encoding='utf-8')
            with mock.patch.dict(os.environ, {
                api_node.DISABLE_BUILTINS_ENV: '1',
                api_node.PROFILE_FILE_ENV: str(path),
            }, clear=False):
                profiles = api_node._load_api_profiles()
            self.assertEqual(list(profiles), ['Production'])
            self.assertEqual(profiles['Production']['api_key_env'], 'TEST_API_KEY')
            self.assertTrue(profiles['Production']['allow_images'])
            self.assertNotIn('api_key', profiles['Production'])

if __name__ == '__main__':
    unittest.main()
