"""Model-local SageAttention patching for supported Qwen attention layers."""

import torch


def set_sage_attention(model):
    """Patch supported Qwen attention layers, falling back per call when needed."""
    from thinkingllm_core.hf_models import get_sage_attention_config

    attn_func, qk_quant_gran, pv_accum_dtype = get_sage_attention_config()
    if attn_func is None:
        raise RuntimeError("No compatible SageAttention kernel found for this GPU")

    attention_classes = []
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb
        attention_classes.append((Qwen2Attention, apply_rotary_pos_emb))
    except ImportError:
        pass
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb
        attention_classes.append((Qwen3Attention, apply_rotary_pos_emb))
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, apply_rotary_pos_emb
        attention_classes.append((Qwen3VLTextAttention, apply_rotary_pos_emb))
    except ImportError:
        pass

    def make_sage_forward(original_forward, apply_rotary):
        def sage_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple = None,
            attention_mask: torch.Tensor = None,
            past_key_values=None,
            cache_position: torch.LongTensor = None,
            **kwargs,
        ):
            def fallback():
                return original_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            # The Triton v1 kernel supports these dimensions and causal
            # attention only when query and key sequence lengths agree.
            if (
                hidden_states.ndim != 3
                or hidden_states.shape[1] < 1
                or (self.head_dim not in (64, 96, 128) if qk_quant_gran is None
                    else not 0 < self.head_dim <= 128)
                or attention_mask is not None
                or position_embeddings is None
                or len(position_embeddings) != 2
                or getattr(self, "sliding_window", None) is not None
                or self.training
                or any(name not in ("position_ids", "use_cache") for name in kwargs)
                or (kwargs.get("use_cache") is False and past_key_values is not None)
                or (past_key_values is not None and hidden_states.shape[1] != 1)
                or hidden_states.dtype not in (torch.float16, torch.bfloat16)
            ):
                return fallback()

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)
            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states)
            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary(query_states, key_states, cos, sin)

            # This check precedes cache.update(), since calling the original
            # forward after an update would append the same token twice.
            if (query_states.dtype != key_states.dtype or key_states.dtype != value_states.dtype
                    or query_states.dtype not in (torch.float16, torch.bfloat16)):
                return fallback()

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            is_causal = input_shape[1] > 1
            if qk_quant_gran is None:
                # SageAttention 1 smooths K in place, including cached K.
                attn_output = attn_func(
                    query_states, key_states.clone(), value_states,
                    tensor_layout="HND", is_causal=is_causal,
                    sm_scale=getattr(self, "scaling", self.head_dim ** -0.5),
                )
            else:
                attn_output = attn_func(
                    query_states, key_states, value_states,
                    tensor_layout="HND", is_causal=is_causal,
                    qk_quant_gran=qk_quant_gran, pv_accum_dtype=pv_accum_dtype,
                )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(attn_output), None

        return sage_forward

    patched_count = 0
    for module in model.modules():
        for attention_class, apply_rotary in attention_classes:
            if isinstance(module, attention_class):
                sage_forward = make_sage_forward(module.forward, apply_rotary)
                module.forward = sage_forward.__get__(module, attention_class)
                patched_count += 1
                break

    if not patched_count:
        raise RuntimeError("SageAttention: No compatible attention layers found to patch")
    print(f"[QwenVL] SageAttention: Patched {patched_count} attention layers")
