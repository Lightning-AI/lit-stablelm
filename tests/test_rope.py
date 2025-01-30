# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass

import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as apply_rotary_pos_emb_gptneo
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama

from litgpt.model import apply_rope, build_rope_cache


@torch.inference_mode()
def test_rope_gptneox():
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head  # 4
    x = torch.randint(0, 10000, size=(bs, n_head, seq_len, head_size)).float()
    position_ids = torch.arange(seq_len).unsqueeze(0)

    @dataclass
    class RoPEConfig:
        dim: int
        max_position_embeddings: int
        rope_theta: int
        hidden_size: int
        num_attention_heads: int

    config = RoPEConfig(
        dim=head_size,
        max_position_embeddings=seq_len,
        rope_theta=10_000,
        hidden_size=head_size * n_head,
        num_attention_heads=n_head
    )

    theirs_rot_emb = GPTNeoXRotaryEmbedding(config)
    theirs_cos, theirs_sin = theirs_rot_emb(x, position_ids)

    ours_cos_cached, ours_sin_cached = build_rope_cache(seq_len, head_size, device=x.device)
    ours_cos_cached = ours_cos_cached.unsqueeze(0)
    ours_sin_cached = ours_sin_cached.unsqueeze(0)
    torch.testing.assert_close(ours_cos_cached, theirs_cos)
    torch.testing.assert_close(ours_sin_cached, theirs_sin)

    ours_x_rope = apply_rope(x, ours_cos_cached, ours_sin_cached)
    theirs_x_rope, _ = apply_rotary_pos_emb_gptneo(x, x, theirs_cos, theirs_sin, position_ids)
    torch.testing.assert_close(ours_x_rope, theirs_x_rope)


@torch.inference_mode()
def test_rope_llama_2():
    head_dim = 64
    rope_theta = 10_000
    num_heads = 4
    batch_size, seq_len = 1, 10

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope

    @dataclass
    class RoPEConfig:
        dim: int
        max_position_embeddings: int
        rope_theta: int
        hidden_size: int
        num_attention_heads: int

    config = RoPEConfig(
        dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=rope_theta,
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads
    )

    rot_emb = LlamaRotaryEmbedding(config)

    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta)
    ours_cos = ours_cos.unsqueeze(0)
    ours_sin = ours_sin.unsqueeze(0)
    torch.testing.assert_close(theirs_cos, ours_cos)
    torch.testing.assert_close(theirs_sin, ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3():
    head_dim = 64
    rope_theta = 50_000
    num_heads = 4
    batch_size, seq_len = 1, 10

    ##################################
    # Compare cos and sin
    ##################################

    @dataclass
    class RoPEConfig:
        dim: int
        max_position_embeddings: int
        rope_theta: int
        hidden_size: int
        num_attention_heads: int
        scaling_factor: float

    config = RoPEConfig(
        dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=rope_theta,
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        scaling_factor=None
    )

    # transformer rope
    rot_emb = LlamaRotaryEmbedding(config)
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta)
    ours_cos = ours_cos.unsqueeze(0)
    ours_sin = ours_sin.unsqueeze(0)
    torch.testing.assert_close(theirs_cos, ours_cos)
    torch.testing.assert_close(theirs_sin, ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3_1():
    head_dim = 32
    rope_theta = 50_000
    num_heads = 4
    batch_size, seq_len = 1, 131_072

    their_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
     }

    our_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
     }

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope

    @dataclass
    class RoPEConfig:
        dim: int
        max_position_embeddings: int
        rope_theta: int
        hidden_size: int
        num_attention_heads: int
        rope_type: str
        rope_scaling: dict

    config = RoPEConfig(
        dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=rope_theta,
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        rope_type="llama3",
        rope_scaling=their_rope_config
    )

    rot_emb = LlamaRotaryEmbedding(config=config)
    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta, extra_config=our_rope_config)
    ours_cos = ours_cos.unsqueeze(0)
    ours_sin = ours_sin.unsqueeze(0)
    torch.testing.assert_close(theirs_cos, ours_cos)
    torch.testing.assert_close(theirs_sin, ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)


# See https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json for settings
@torch.inference_mode()
def test_rope_llama_3_2():
    head_dim = 32
    rope_theta = 50_000
    batch_size, seq_len = 1, 131_072
    num_heads = 4

    their_rope_config = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
     }

    our_rope_config = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
     }

    ##################################
    # Compare cos and sin
    ##################################
    # transformer rope
    @dataclass
    class RoPEConfig:
        dim: int
        max_position_embeddings: int
        rope_theta: int
        hidden_size: int
        num_attention_heads: int
        rope_type: str
        rope_scaling: dict

    config = RoPEConfig(
        dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=rope_theta,
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        rope_type="llama3",
        rope_scaling=their_rope_config
    )

    rot_emb = LlamaRotaryEmbedding(config)

    qk_tensor = torch.randn(batch_size, seq_len, head_dim)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    theirs_cos, theirs_sin = rot_emb(qk_tensor, position_ids)

    # our rope
    ours_cos, ours_sin = build_rope_cache(seq_len, n_elem=head_dim, base=rope_theta, extra_config=our_rope_config)
    ours_cos = ours_cos.unsqueeze(0)
    ours_sin = ours_sin.unsqueeze(0)
    torch.testing.assert_close(theirs_cos, ours_cos)
    torch.testing.assert_close(theirs_sin, ours_sin)

    ##################################
    # Compare rotated tensors
    ##################################

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

    ours_q_rot = apply_rope(queries, ours_cos, ours_sin)
    ours_k_rot = apply_rope(keys, ours_cos, ours_sin)
    theirs_q_rot, theirs_k_rot = apply_rotary_pos_emb_llama(queries, keys, theirs_cos, theirs_sin)
    torch.testing.assert_close(theirs_q_rot, ours_q_rot)
    torch.testing.assert_close(theirs_k_rot, ours_k_rot)

@torch.inference_mode()
def test_rope_cos_sin_shapes_if_rope_n_elem_is_odd():
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head  # 4
    rotary_percentage = 0.75
    rope_n_elem = int(head_size * rotary_percentage)  # 3
    ours_cos, ours_sin = build_rope_cache(seq_len, rope_n_elem)
    required_shape = (seq_len, rope_n_elem)
    assert ours_cos.shape == required_shape
    assert ours_sin.shape == required_shape
    # Special case: If `rope_n_elem == 1`, the shape is extended. This is to
    # accommodate a current bug in Hugging Face, ensuring that other unit tests
    # pass.
    # https://github.com/huggingface/transformers/issues/35233
    rotary_percentage = 0.25
    rope_n_elem = int(head_size * rotary_percentage)  # 1
    ours_cos, ours_sin = build_rope_cache(seq_len, rope_n_elem)
    required_shape = (seq_len, rope_n_elem + 1)
    assert ours_cos.shape == required_shape
    assert ours_sin.shape == required_shape
