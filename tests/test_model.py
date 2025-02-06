# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy
from functools import partial
from unittest import mock
import math
import random

import pytest
import torch
from lightning import Fabric
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.init import _materialize_meta_tensors
from torch._dynamo.backends import debugging
from torch.backends.cuda import (
    SDPAParams,
    SDPBackend,
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
)
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.falcon import FalconConfig, FalconForCausalLM
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.olmo import OlmoConfig, OlmoForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

import litgpt.config as config_module
from litgpt import GPT, Config
from litgpt.kvcache.base import DefaultKeysAndValues
from litgpt.model import CausalSelfAttention, scaled_dot_product_attention
from litgpt.utils import batched_index_copy_
from litgpt.scripts.convert_hf_checkpoint import (
    copy_weights_falcon,
    copy_weights_gemma_2,
    copy_weights_gpt_neox,
    copy_weights_hf_llama,
    copy_weights_phi,
    copy_weights_qwen_2_5,
)
from litgpt.scripts.convert_lit_checkpoint import qkv_reassemble as make_qkv_interleaved
from tests.conftest import RunIf


@torch.inference_mode()
@pytest.mark.parametrize("rotary_pct", (0.25, 1))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("n_embd", (16, 32))
@pytest.mark.parametrize("parallel_residual", (False, True))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_gpt_neox_model(rotary_pct, batch_size, n_embd, parallel_residual, device, dtype) -> None:
    torch.set_default_dtype(dtype)

    ours_config = Config(
        block_size=64,
        vocab_size=100,
        n_layer=4,
        n_head=8,
        n_embd=n_embd,
        rotary_percentage=rotary_pct,
        parallel_residual=parallel_residual,
    )
    assert ours_config.padded_vocab_size == 512
    theirs_config = GPTNeoXConfig(
        hidden_act="gelu",
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        initializer_range=0.02,
        intermediate_size=ours_config.intermediate_size,
        layer_norm_eps=ours_config.norm_eps,
        max_position_embeddings=ours_config.block_size,
        rotary_emb_base=10000,
        rotary_pct=ours_config.rotary_percentage,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=ours_config.parallel_residual,
        attn_implementation="eager",
    )

    state_dict = {}
    theirs_model = GPTNeoXForCausalLM(theirs_config).to(device)
    # load the hf initialization into our model
    copy_weights_gpt_neox(ours_config, state_dict, theirs_model.state_dict())
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    token_sample = torch.randint(
        0, ours_config.padded_vocab_size, size=(batch_size, ours_config.block_size), dtype=torch.int64, device=device
    )

    theirs = theirs_model(token_sample)["logits"]
    ours = ours_model(token_sample)
    torch.testing.assert_close(ours, theirs)


@torch.inference_mode()
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="falcon-180B", n_layer=2, n_head=8, n_query_groups=4, n_embd=32),
        dict(name="falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32),
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_hf_falcon(kwargs, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(**kwargs)
    theirs_config = FalconConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_kv_heads=ours_config.n_query_groups,
        num_hidden_layers=ours_config.n_layer,
        parallel_attn=ours_config.parallel_residual,
        vocab_size=ours_config.padded_vocab_size,
        bias=ours_config.bias,
        new_decoder_architecture=True,
    )

    theirs_model = FalconForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_falcon(ours_config, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_open_llama_3b(device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name("open_llama_3b", n_layer=2, n_head=8, n_embd=32, intermediate_size=86)
    T = 5
    theirs_config = LlamaConfig(
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs",
    [
        {"name": "Llama-2-7b-hf"},
        {"name": "CodeLlama-7b-hf"},
        {"name": "Llama-2-70b-chat-hf", "n_query_groups": 1},
        {"name": "Llama-3-8B"},
        {"name": "Llama-3-8B-Instruct"},
        {"name": "Llama-3.1-405B", "n_query_groups": 4},
        {"name": "Llama-3.1-8B"},
        {"name": "Llama-3.1-8B-Instruct"},
        {"name": "Llama-3.2-1B"},
        {"name": "Llama-3.2-3B"},
        {"name": "Llama-3.3-70B-Instruct"},
        {"name": "R1-Distill-Llama-8B"},
        {"name": "R1-Distill-Llama-70B"},
    ],
)
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_hf_llama_2_and_3(ours_kwargs, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        padded_vocab_size=10000, n_layer=2, n_head=8, n_embd=32, intermediate_size=86, **ours_kwargs
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("phi-1_5", "phi-2"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[pytest.mark.xfail(raises=AssertionError, strict=False), RunIf(min_cuda_gpus=1)],
        ),
    ],
)
def test_against_hf_phi(model_name, device, dtype):
    from transformers.models.phi.configuration_phi import PhiConfig
    from transformers.models.phi.modeling_phi import PhiForCausalLM

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name, padded_vocab_size=10000, n_layer=2, n_head=4, n_embd=256, rotary_percentage=0.5
    )
    T = 5
    theirs_config = PhiConfig(
        vocab_size=ours_config.padded_vocab_size,
        max_position_embeddings=ours_config.block_size,
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        partial_rotary_factor=ours_config.rotary_percentage,
        torch_dtype=dtype,
    )

    theirs_model = PhiForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_phi(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct", "Phi-3.5-mini-instruct", "phi-4"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[pytest.mark.xfail(raises=AssertionError, strict=False), RunIf(min_cuda_gpus=1)],
        ),
    ],
)
def test_against_hf_phi_3(model_name, device, dtype):
    from transformers.models.phi3.configuration_phi3 import Phi3Config
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_head=4,
        n_query_groups=4,
        n_embd=256,
    )
    T = 5
    theirs_config = Phi3Config(
        attention_bias=ours_config.bias,
        head_dim=ours_config.head_size,
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        num_key_value_heads=ours_config.n_query_groups,
        pad_token_id=ours_config.padded_vocab_size - 1,
        partial_rotary_factor=ours_config.rotary_percentage,
        rms_norm_eps=ours_config.norm_eps,
        rope_theta=ours_config.rope_base,
        torch_dtype=dtype,
        vocab_size=ours_config.padded_vocab_size,
    )

    theirs_model = Phi3ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_phi(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
@pytest.mark.parametrize("model_name", ["Mistral-7B-Instruct-v0.1", "Mistral-7B-v0.1"])
def test_against_mistral_hf_models(device, dtype, model_name):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        block_size=T,
        sliding_window_size=T // 2,
        sliding_window_layer_placing="all",
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )

    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attn_implementation="eager",
        sliding_window=ours_config.sliding_window_size,
    )

    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_mathstral_hf_models(device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "Mathstral-7B-v0.1",
        padded_vocab_size=10000,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
    )

    T = 5
    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )

    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x22B-Instruct-v0.1"))
def test_against_hf_mixtral(model_name):
    device = torch.device("cpu")
    dtype = torch.float32
    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
        n_expert=4,
    )
    T = 5
    theirs_config = MixtralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        num_local_experts=ours_config.n_expert,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = MixtralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("OLMo-1B-hf", "OLMo-7B-hf"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_olmo(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_head=8,
        n_embd=32,
        intermediate_size=86,
    )
    T = 5
    theirs_config = OlmoConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        num_key_value_heads=ours_config.n_query_groups,
        max_positional_embeddings=T,
        attention_bias=ours_config.bias,
        rope_theta=ours_config.rope_base,
        tie_word_embeddings=(model_name == "OLMo-1B-hf"),
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = OlmoForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_stablelm_zephyr_3b(device, dtype):
    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name("stablelm-zephyr-3b", n_layer=2, n_head=16, n_embd=32, intermediate_size=86)
    theirs_config = AutoConfig.from_pretrained(
        "stabilityai/stablelm-zephyr-3b",
        trust_remote_code=True,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        num_key_value_heads=ours_config.n_head,
        hidden_size=ours_config.n_embd,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        torch_dtype=dtype,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = AutoModelForCausalLM.from_config(theirs_config, trust_remote_code=True).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ["gemma-2b", "gemma-7b"])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_gemma(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 5
    ours_config = Config.from_name(
        model_name,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = GemmaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = GemmaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-2-9b", "gemma-2-27b"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_gemma_2(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
        rotary_percentage=1.0,  # Gemma2 does not have this
    )
    theirs_config = Gemma2Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        sliding_window=ours_config.sliding_window_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=ours_config.attention_logit_softcapping,
        final_logit_softcapping=ours_config.final_logit_softcapping,
        initializer_range=1.0,  # to make the affect of attention_logit_softcapping more prominent
        attn_implementation="eager",
        query_pre_attn_scalar=ours_config.attention_scores_scalar,
    )

    theirs_model = Gemma2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_gemma_2({}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, rtol=3e-5, atol=3e-5)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Math-1.5B", "QwQ-32B-Preview"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_qwen_2_5(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = Qwen2Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.attn_bias,
        tie_word_embeddings=True,
    )

    theirs_model = Qwen2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_qwen_2_5(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("salamandra-2b", "salamandra-7b"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_salamandra(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_head=8,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_smollm2(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_head=8,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Falcon3-1B-Base", "Falcon3-7B-Base"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_hf_falcon3(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        model_name,
        padded_vocab_size=10000,
        n_layer=2,
        n_head=8,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=86,
    )
    T = 5
    theirs_config = LlamaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = LlamaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@RunIf(dynamo=True)
@torch.inference_mode()
def test_model_compile():
    model = GPT.from_name("pythia-14m", n_layer=3)
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = GPT(model.config)
    model.set_kv_cache(2)
    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@torch.inference_mode()
def test_model_kv_cache_amp():
    config = Config.from_name("pythia-14m", n_layer=2)
    model = GPT(config)
    encoded = torch.arange(45).view(1, -1)
    model.set_kv_cache(batch_size=1)
    with torch.autocast("cpu", torch.bfloat16):
        output = model(encoded, for_prefill=True)
    assert output.dtype is torch.bfloat16


# https://github.com/pytorch/pytorch/blob/ad3572a5d/torch/testing/_internal/common_cuda.py#L31-L34
SUPPORTS_FLASH_ATTENTION = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0) and not _IS_WINDOWS
)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("config", deepcopy(config_module.configs), ids=[c["name"] for c in config_module.configs])
@torch.inference_mode()
def test_sdpa_choice(config):
    if config["name"].startswith("Gemma-2-"):
        pytest.skip("Gemma 2 doesn't support SDPA")

    torch.set_default_dtype(torch.float16)

    def assert_sdpa_backend(original_fn, q, k, v, mask):
        # SDPAParams gained an additional argument in PyTorch 2.5
        args = []
        if hasattr(SDPAParams, "enable_gqa"):
            args.append(False)
        params = SDPAParams(q, k, v, mask, 0.0, True, *args)
        if expected is SDPBackend.FLASH_ATTENTION:
            assert flash_sdp_enabled(), "flash_sdp_enabled() is False"
            if config.sliding_window_size is None:
                assert can_use_flash_attention(params, True), "can_use_flash_attention(params, True) is False"
        elif expected is SDPBackend.EFFICIENT_ATTENTION:
            assert mem_efficient_sdp_enabled(), "mem_efficient_sdp_enabled() is False"
            assert can_use_efficient_attention(params, True), "can_use_efficient_attention(params, True) is False"
        elif expected is SDPBackend.MATH:
            assert math_sdp_enabled(), "math_sdp_enabled() is False"
        else:
            raise NotImplementedError
        return original_fn(q, k, v, mask)

    config["n_layer"] = 1
    config = config_module.Config(**config)

    try:
        with torch.device("cuda"):
            model = GPT(config)
            x = torch.randint(0, 10, (2, 16), dtype=torch.int32)
    except torch.cuda.OutOfMemoryError:
        # best effort, if the GPU can load it
        pytest.xfail()

    for h in model.transformer.h:
        h.attn.scaled_dot_product_attention = partial(assert_sdpa_backend, h.attn.scaled_dot_product_attention)

    if SUPPORTS_FLASH_ATTENTION:
        expected = SDPBackend.FLASH_ATTENTION
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
            model(x)

    expected = SDPBackend.EFFICIENT_ATTENTION if config.head_size % 8 == 0 else SDPBackend.MATH
    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        model(x)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("config", deepcopy(config_module.configs), ids=[c["name"] for c in config_module.configs])
@torch.inference_mode()
def test_sdpa_choice_kv_cache(config):
    torch.set_default_dtype(torch.float16)

    def assert_sdpa_backend(original_fn, q, k, v, mask):
        # SDPAParams gained an additional argument in PyTorch 2.5
        args = []
        if hasattr(SDPAParams, "enable_gqa"):
            args.append(False)
        params = SDPAParams(q, k, v, mask, 0.0, True, *args)
        if expected is SDPBackend.FLASH_ATTENTION:
            assert flash_sdp_enabled()
            assert can_use_flash_attention(params, True)
        elif expected is SDPBackend.EFFICIENT_ATTENTION:
            assert mem_efficient_sdp_enabled()
            assert can_use_efficient_attention(params, True)
        elif expected is SDPBackend.MATH:
            assert math_sdp_enabled()
        else:
            raise NotImplementedError
        return original_fn(q, k, v, mask)

    config["n_layer"] = 1
    config = config_module.Config(**config)

    try:
        with torch.device("cuda"):
            model = GPT(config)
            model.max_seq_length = 1
            model.set_kv_cache(2)
            x = torch.randint(0, 10, (2, 1), dtype=torch.int32)
    except torch.cuda.OutOfMemoryError:
        # best effort, if the GPU can load it
        pytest.xfail()

    for h in model.transformer.h:
        h.attn.scaled_dot_product_attention = partial(assert_sdpa_backend, h.attn.scaled_dot_product_attention)

    if SUPPORTS_FLASH_ATTENTION:
        # flash attention does not support an attention mask
        expected = SDPBackend.MATH
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
            model(x, for_prefill=True)

    expected = (
        SDPBackend.EFFICIENT_ATTENTION if config.head_size % 8 == 0 and config.n_query_groups != 1 else SDPBackend.MATH
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        model(x, for_prefill=True)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_rope_init_under_fsdp():
    """Check that the rope cache is properly initialized"""
    fabric = Fabric(devices=2, strategy="fsdp", accelerator="cuda")
    fabric.launch()

    with fabric.init_module(empty_init=True):
        model = GPT.from_name("pythia-14m", n_layer=1)
    assert model.cos.device.type == "meta"
    assert model.sin.device.type == "meta"

    model = fabric.setup(model)
    assert model.cos.device.type == "cuda"
    assert model.sin.device.type == "cuda"
    cos, sin = model.rope_cache(device=fabric.device)
    torch.testing.assert_close(model.cos, cos)
    torch.testing.assert_close(model.sin, sin)


@RunIf(min_cuda_gpus=1)
def test_reset_parameters_device():
    with torch.device("meta"):
        model = GPT.from_name("pythia-14m", n_layer=1)
    _materialize_meta_tensors(model, torch.device("cuda"))
    model.reset_parameters()
    assert model.cos.device.type == "cuda"


def test_batched_index_copy_modes():
    # Mock the torch.backends.mps.is_available() function to simulate MPS availability
    with mock.patch("torch.backends.mps.is_available", return_value=True):
        # Mock the device type to simulate the "mps" device
        with mock.patch("torch.Tensor.device", new_callable=mock.PropertyMock) as mock_device:
            mock_device.return_value = torch.device("mps")

            # Test case when idx.dim() == 1
            t_original_1 = torch.randn(3, 5)
            dim_1 = 0
            idx_1 = torch.tensor([0, 2])
            val_1 = torch.randn(2, 5)

            t1_cpu = t_original_1.clone()
            t1_mps = t_original_1.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t1_cpu, dim_1, idx_1, val_1)

            # Simulate the MPS index copy
            idx_1_mps = idx_1
            val_1_mps = val_1
            batched_index_copy_(t1_mps, dim_1, idx_1_mps, val_1_mps)
            assert torch.allclose(t1_cpu, t1_mps), "Mismatch with idx.dim() == 1 on mocked MPS"

            # Test case when idx.dim() == 2
            t_original_2 = torch.randn(2, 5, 4)
            dim_2 = 1
            idx_2 = torch.tensor([[0, 2], [1, 3]])
            val_2 = torch.randn(2, 2, 4)

            t2_cpu = t_original_2.clone()
            t2_mps = t_original_2.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t2_cpu, dim_2, idx_2, val_2)

            # Simulate the MPS index copy
            idx_2_mps = idx_2
            val_2_mps = val_2
            batched_index_copy_(t2_mps, dim_2, idx_2_mps, val_2_mps)
            assert torch.allclose(t2_cpu, t2_mps), "Mismatch with idx.dim() == 2 on mocked MPS"

            # Additional test with negative dimension
            t_original_3 = torch.randn(2, 3, 4)
            dim_3 = -2
            idx_3 = torch.tensor([[0, 1], [1, 2]])
            val_3 = torch.randn(2, 2, 4)

            t3_cpu = t_original_3.clone()
            t3_mps = t_original_3.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t3_cpu, dim_3, idx_3, val_3)

            # Simulate the MPS index copy
            idx_3_mps = idx_3
            val_3_mps = val_3
            batched_index_copy_(t3_mps, dim_3, idx_3_mps, val_3_mps)
            assert torch.allclose(t3_cpu, t3_mps), "Mismatch with negative dimension on mocked MPS"

def test_load_legacy_state_dict():
    """Check that a legacy state dict (with an interleaved placement in QKV matrix) can be loaded into a model with CausalSelfAttention layers."""
    config = Config(
        n_embd=32,
        n_head=4,
        head_size=8,
        n_query_groups=4,
        bias=True,
    )

    attention_1 = CausalSelfAttention(config=config, block_idx=0)

    # make weights to be as-like in a legacy checkpoint, with `attn.attn.weight` instead of `attn.qkv.weight`
    # and make them interleaved
    state_dict = deepcopy(attention_1.state_dict())
    state_dict["attn.weight"] = make_qkv_interleaved(state_dict.pop("qkv.weight"), config)
    state_dict["attn.bias"] = make_qkv_interleaved(state_dict.pop("qkv.bias"), config)

    attention_2 = CausalSelfAttention(config=config, block_idx=0)
    attention_2.load_state_dict(state_dict)

@pytest.mark.parametrize("n_query_groups", (1, 2, 4, 8))
@torch.inference_mode()
def test_kv_cache_buffer_shape(n_query_groups):
    batch_size = 3
    max_seq_length = 23
    config = Config(
        block_size=25,
        padded_vocab_size=5,
        n_layer=2,
        n_head=8,
        n_embd=16,
        n_query_groups=n_query_groups,
    )
    model = GPT(config)
    model.max_seq_length = max_seq_length
    model.set_kv_cache(batch_size)
    required_shape = (batch_size, n_query_groups, max_seq_length, config.head_size)
    for block in model.transformer.h:
        kv_cache = block.attn.kv_cache
        assert kv_cache is not None
        assert kv_cache.k.shape == required_shape
        assert kv_cache.v.shape == required_shape


@pytest.mark.parametrize(
    ("rotary_percentage", "final_dim"),
    ((0.75, 3), (0.25, 2))
)
@torch.inference_mode()
def test_rope_cos_sin_shapes_if_rope_n_elem_is_odd(rotary_percentage, final_dim):
    batch_size = 3
    config = Config(
        block_size=25,
        padded_vocab_size=5,
        n_layer=2,
        n_head=4,
        n_embd=16,
        rotary_percentage=rotary_percentage,
    )
    model = GPT(config)
    required_shape = (config.block_size, final_dim)
    assert model.cos.shape == required_shape
    assert model.sin.shape == required_shape


@pytest.mark.parametrize(
    ("n_head", "n_query_groups"),
    (
        (2, 1),
        (4, 1),
        (8, 4),
        (12, 4),
        (24, 8),
        (9, 3),
    ),
)
@torch.inference_mode()
def test_scaled_dot_product_attention(n_head, n_query_groups):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 5
    dtype = torch.bfloat16

    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        len_key = random.randint(16, 128)
        is_causal = repeat % 2 == 0
        if is_causal:
            len_query = len_key
        elif repeat % 4 == 1:
            len_query = random.randint(1, len_key // 2)
        else:
            len_query = 1
        shape = (batch_size, n_head, len_query, head_size)
        query = torch.randn(shape, dtype=dtype)
        shape = (batch_size, n_query_groups, len_key, head_size)
        key = torch.randn(shape, dtype=dtype)
        value = torch.randn(shape, dtype=dtype)
        k_and_v = DefaultKeysAndValues(key, value)
        scale = 1.0 / math.sqrt(head_size)

        result, scores = scaled_dot_product_attention(
            query,
            k_and_v,
            scale=scale,
            is_causal=is_causal,
        )
        q_per_kv = n_head // n_query_groups
        key_bc = key.repeat_interleave(q_per_kv, dim=1)
        value_bc = value.repeat_interleave(q_per_kv, dim=1)
        k_and_v_bc = DefaultKeysAndValues(key_bc, value_bc)
        result_cmp, scores_cmp = scaled_dot_product_attention(
            query,
            k_and_v_bc,
            scale=scale,
            is_causal=is_causal,
        )
        msg = f"bs={batch_size}, hs={head_size}, nh_q={n_head}, nh_k={n_query_groups}, len_q={len_query}, len_k={len_key}"
        kwargs = dict(atol=0.0005, rtol=0.05)
        torch.testing.assert_close(result, result_cmp, **kwargs), msg
        torch.testing.assert_close(scores, scores_cmp, **kwargs), msg
