"""Regression checks for SageAttention selection and model-local fallback."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parent.parent


def load_attention_selector(*, hip, sage_version="1.0.6", triton_kernel=None, cuda_kernels=False):
    tree = ast.parse((ROOT / "thinkingllm_core/hf_models.py").read_text(encoding="utf-8"))
    names = {"sage_attn_available", "get_sage_attention_config", "resolve_attention_mode"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=(
            (lambda: (_ for _ in ()).throw(AssertionError("HIP queried CUDA SM")))
            if hip else (lambda: (8, 6))
        ),
    )
    metadata = types.SimpleNamespace(
        version=lambda package: sage_version,
        PackageNotFoundError=Exception,
    )
    namespace = {
        "torch": types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(hip=hip)),
        "importlib": types.SimpleNamespace(metadata=metadata),
        "sageattn_triton": triton_kernel,
        "SAGE_ATTENTION_AVAILABLE": cuda_kernels,
        "sageattn_qk_int8_pv_fp16_cuda": triton_kernel if cuda_kernels else None,
        "sageattn_qk_int8_pv_fp8_cuda": None,
        "sageattn_qk_int8_pv_fp8_cuda_sm90": None,
        "flash_attn_available": lambda: False,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "hf_models.py", "exec"), namespace)
    return namespace


class TestSageBackendSelection(unittest.TestCase):
    def test_rocm_v1_selects_triton_without_nvidia_capability_query(self):
        kernel = lambda *args, **kwargs: None
        selector = load_attention_selector(hip="7.0", triton_kernel=kernel)
        self.assertIs(selector["get_sage_attention_config"]()[0], kernel)
        self.assertEqual("sage", selector["resolve_attention_mode"]("sage"))
        self.assertEqual("sage", selector["resolve_attention_mode"]("auto"))

    def test_rocm_unavailable_or_unsupported_version_falls_back(self):
        for version, kernel in (("2.2.0", lambda: None), ("1.0.6", None)):
            with self.subTest(version=version, installed=kernel is not None):
                selector = load_attention_selector(hip="7.0", sage_version=version, triton_kernel=kernel)
                self.assertEqual("sdpa", selector["resolve_attention_mode"]("sage"))

    def test_forced_sdpa_still_wins(self):
        selector = load_attention_selector(hip="7.0", triton_kernel=lambda: None)
        self.assertEqual("sdpa", selector["resolve_attention_mode"]("sage", force_sdpa=True))

    def test_nvidia_cuda_kernel_selection_is_preserved(self):
        kernel = lambda *args, **kwargs: None
        selector = load_attention_selector(hip=None, triton_kernel=kernel, cuda_kernels=True)
        self.assertEqual((kernel, "per_warp", "fp32"), selector["get_sage_attention_config"]())


class FakeAttention(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0
        self.scaling = head_dim ** -0.5
        self.q_proj = torch.nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = torch.nn.Linear(head_dim, head_dim, bias=False)
        self.v_proj = torch.nn.Linear(head_dim, head_dim, bias=False)
        self.o_proj = torch.nn.Linear(head_dim, head_dim, bias=False)
        self.original_calls = 0

    def forward(self, hidden_states, **kwargs):
        self.original_calls += 1
        return hidden_states, None


class FakeCache:
    def __init__(self):
        self.keys = None

    def update(self, keys, values, layer_idx, cache_kwargs):
        self.keys = keys
        return keys, values


class TestSageModelPatch(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def kernel(q, k, v, **kwargs):
            self.calls.append(kwargs)
            k.add_(1)  # SageAttention 1 smooths K in place.
            return q

        hf_models = types.ModuleType("thinkingllm_core.hf_models")
        hf_models.get_sage_attention_config = lambda: (kernel, None, None)
        qwen2 = types.ModuleType("transformers.models.qwen2.modeling_qwen2")
        qwen2.Qwen2Attention = FakeAttention
        qwen2.apply_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
        self.modules = mock.patch.dict(sys.modules, {
            "thinkingllm_core.hf_models": hf_models,
            "transformers.models.qwen2.modeling_qwen2": qwen2,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location("_test_sageattention_patch", ROOT / "sageattention_patch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.patch_model = module.set_sage_attention

    @staticmethod
    def embeddings():
        return torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)

    def test_unmasked_prefill_and_cached_decode_use_sage_without_mutating_cache(self):
        layer = FakeAttention().half().eval()
        self.patch_model(torch.nn.Sequential(layer))
        prefill = torch.randn(1, 2, 64, dtype=torch.float16)
        output, weights = layer(prefill, position_embeddings=self.embeddings(), use_cache=True, position_ids=None)
        self.assertEqual((1, 2, 64), tuple(output.shape))
        self.assertIsNone(weights)
        self.assertTrue(self.calls[-1]["is_causal"])
        self.assertEqual(0, layer.original_calls)

        cache = FakeCache()
        token = torch.randn(1, 1, 64, dtype=torch.float16)
        layer(token, position_embeddings=self.embeddings(), past_key_values=cache, use_cache=True)
        self.assertFalse(self.calls[-1]["is_causal"])
        self.assertEqual(2, len(self.calls))
        self.assertIsNotNone(cache.keys)
        expected_keys = layer.k_proj(token).view(1, 1, 1, 64).transpose(1, 2)
        torch.testing.assert_close(cache.keys, expected_keys)

    def test_mask_and_unsupported_head_size_call_original_forward(self):
        layer = FakeAttention().half().eval()
        self.patch_model(torch.nn.Sequential(layer))
        inputs = torch.randn(1, 2, 64, dtype=torch.float16)
        layer(inputs, position_embeddings=self.embeddings(), attention_mask=torch.zeros(1, 1, 2, 2))
        self.assertEqual(1, layer.original_calls)
        self.assertEqual([], self.calls)

        unsupported = FakeAttention(head_dim=80).half().eval()
        self.patch_model(torch.nn.Sequential(unsupported))
        unsupported(torch.randn(1, 2, 80, dtype=torch.float16), position_embeddings=self.embeddings())
        self.assertEqual(1, unsupported.original_calls)
        self.assertEqual([], self.calls)

    def test_no_supported_layers_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "No compatible attention layers"):
            self.patch_model(torch.nn.Sequential(torch.nn.Linear(2, 2)))


if __name__ == "__main__":
    unittest.main()
