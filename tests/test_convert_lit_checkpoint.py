# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from dataclasses import asdict
from unittest.mock import ANY

import pytest
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.falcon import FalconConfig, FalconForCausalLM
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.olmo import OlmoConfig, OlmoForCausalLM
from transformers.models.phi.configuration_phi import PhiConfig
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from litgpt import GPT, Config
from litgpt.scripts.convert_lit_checkpoint import (
    check_conversion_supported,
    convert_lit_checkpoint,
    copy_weights_falcon,
    copy_weights_gemma_2,
    copy_weights_gpt_neox,
    copy_weights_llama,
    copy_weights_phi,
    copy_weights_qwen_2_5,
    qkv_reassemble,
)
from tests.conftest import RunIf


@pytest.mark.parametrize("model_name", ("pythia-14m", "falcon-7b", "Llama-2-7b-hf", "phi-2"))
def test_convert_lit_checkpoint(tmp_path, model_name):
    ours_config = Config.from_name(model_name, block_size=8, n_layer=2, n_embd=32, n_head=2, padding_multiple=128)
    ours_model = GPT(ours_config)
    checkpoint_path = tmp_path / "lit_model.pth"
    config_path = tmp_path / "model_config.yaml"
    torch.save(ours_model.state_dict(), checkpoint_path)
    with open(config_path, "w", encoding="utf-8") as fp:
        yaml.dump(asdict(ours_config), fp)
    output_dir = tmp_path / "out_dir"

    convert_lit_checkpoint(checkpoint_path.parent, output_dir)
    assert set(os.listdir(tmp_path)) == {"lit_model.pth", "model_config.yaml", "out_dir"}
    assert os.path.isfile(output_dir / "model.pth")

    # check checkpoint is unwrapped
    torch.save({"model": ours_model.state_dict()}, checkpoint_path)
    convert_lit_checkpoint(checkpoint_path.parent, output_dir)
    converted_sd = torch.load(output_dir / "model.pth")
    assert "model" not in converted_sd


@torch.inference_mode()
def test_against_falcon_40b():
    ours_config = Config.from_name("falcon-40b", n_layer=2, n_head=8, n_query_groups=4, n_embd=32)
    theirs_config = FalconConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_hidden_layers=ours_config.n_layer,
        num_attention_heads=ours_config.n_head,
        num_kv_heads=ours_config.n_query_groups,
        new_decoder_architecture=True,
        parallel_attn=ours_config.parallel_residual,
        bias=ours_config.bias,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_falcon(ours_config, theirs_state_dict, ours_state_dict)

    theirs_model = FalconForCausalLM(theirs_config)
    # assign must be set to True for torch.testing.assert_close to pass
    theirs_model.load_state_dict(theirs_state_dict, assign=True)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_gpt_neox():
    ours_config = Config(block_size=64, vocab_size=100, n_layer=4, n_head=8, n_embd=16)
    assert ours_config.padded_vocab_size == 512
    theirs_config = GPTNeoXConfig(
        hidden_act="gelu",
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        initializer_range=0.02,
        intermediate_size=ours_config.intermediate_size,
        layer_norm_eps=1e-05,
        max_position_embeddings=ours_config.block_size,
        rotary_emb_base=10000,
        rotary_pct=ours_config.rotary_percentage,
        vocab_size=ours_config.padded_vocab_size,
        use_parallel_residual=ours_config.parallel_residual,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_gpt_neox(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = GPTNeoXForCausalLM(theirs_config)
    # strict=False because we don't save the rotary embeddings inv frequency
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys
    assert all("inv_freq" in k for k in keys.missing_keys)

    # test end to end
    x = torch.randint(0, ours_config.padded_vocab_size, size=(2, ours_config.block_size), dtype=torch.int64)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize(
    "ours_kwargs", [{"name": "Llama-2-7b-hf"}, {"name": "CodeLlama-7b-hf"}, {"name": "Llama-2-70b-chat-hf"}]
)
def test_against_hf_llama2(ours_kwargs):
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
        num_query_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x22B-Instruct-v0.1"))
def test_against_mixtral(model_name):
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

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = MixtralForCausalLM(theirs_config)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32)
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("OLMo-1B-hf", "OLMo-7B-hf"))
def test_against_olmo(model_name):
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

    ours_model = GPT(ours_config)
    # tie weights
    ours_model.lm_head.weight = ours_model.transformer.wte.weight
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict, untie_weights=(model_name == "OLMo-1B-hf"))
    theirs_model = OlmoForCausalLM(theirs_config)
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_open_llama_3b():
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

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = LlamaForCausalLM(theirs_config)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("phi-1_5", "phi-2"))
def test_against_hf_phi(model_name):
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
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_phi(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = PhiForCausalLM(theirs_config)
    # strict=False because we don't save the rotary embeddings inv frequency
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys
    assert all("inv_freq" in k for k in keys.missing_keys)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Phi-3-mini-4k-instruct",))
def test_against_hf_phi_3(model_name):
    ours_config = Config.from_name(model_name, padded_vocab_size=10000, n_layer=2, n_head=4, n_embd=256)
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
        vocab_size=ours_config.padded_vocab_size,
    )

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_phi(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = Phi3ForCausalLM(theirs_config)
    # strict=False because we don't save the rotary embeddings inv frequency
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys
    assert all("inv_freq" in k for k in keys.missing_keys)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
def test_against_original_stablelm_zephyr_3b():
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
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config)
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict)
    theirs_model = AutoModelForCausalLM.from_config(theirs_config, trust_remote_code=True)
    theirs_model.load_state_dict(theirs_state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"]
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
    ours_config = Config.from_name(model_name, n_layer=2, n_head=16, n_embd=32, intermediate_size=86)
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

    ours_model = GPT(ours_config).to(device)
    # tie weights
    ours_model.lm_head.weight = ours_model.transformer.wte.weight
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_llama(ours_config, theirs_state_dict, ours_state_dict, untie_weights=True)
    theirs_model = GemmaForCausalLM(theirs_config).to(device)
    theirs_model.load_state_dict(
        theirs_state_dict,
        strict=False,
    )

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-2-2b", "gemma-2-9b", "gemma-2-27b"))
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

    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config).to(device)
    # tie weights
    ours_model.lm_head.weight = ours_model.transformer.wte.weight
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_gemma_2(ours_config, theirs_state_dict, ours_state_dict, untie_weights=True)
    theirs_model = Gemma2ForCausalLM(theirs_config).to(device)
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, rtol=3e-5, atol=3e-5)


def test_check_conversion_supported_adapter():
    lit_weights = {"some.key.name": ANY, "error.key.gating_factor": ANY}
    with pytest.raises(NotImplementedError, match="Converting adapter"):
        check_conversion_supported(lit_weights=lit_weights)

    lit_weights = {"some.key.name": ANY, "error.key.adapter_bias": ANY}
    with pytest.raises(NotImplementedError, match="Converting adapter"):
        check_conversion_supported(lit_weights=lit_weights)


def test_check_conversion_supported_lora():
    lit_weights = {"some.key.name": ANY, "error.key.lora": ANY}
    with pytest.raises(ValueError, match=r"LoRA.*cannot be converted"):
        check_conversion_supported(lit_weights=lit_weights)

@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Math-1.5B", "QwQ-32B-Preview", "Qwen2.5-14B-Instruct-1M"))
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

    assert ours_config.intermediate_size == theirs_config.intermediate_size

    ours_model = GPT(ours_config).to(device)
    # tie weights
    ours_model.lm_head.weight = ours_model.transformer.wte.weight
    ours_state_dict = ours_model.state_dict()
    theirs_state_dict = {}
    copy_weights_qwen_2_5(ours_config, theirs_state_dict, ours_state_dict, untie_weights=True)
    theirs_model = Qwen2ForCausalLM(theirs_config).to(device)
    keys = theirs_model.load_state_dict(theirs_state_dict, strict=False)
    assert not keys.unexpected_keys

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)

def test_qkv_reassemble():
    # MHA
    config = Config(n_embd=4, n_head=4)
    qkv = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [4, 5, 6, 7],  # query
            [8, 9, 10, 11],  # query
            [12, 13, 14, 15],  # query
            [16, 17, 18, 19],  # key
            [20, 21, 22, 23],  # key
            [24, 25, 26, 27],  # key
            [28, 29, 30, 31],  # key
            [32, 33, 34, 35],  # value
            [36, 37, 38, 39],  # value
            [40, 41, 42, 43],  # value
            [44, 45, 46, 47],  # value
        ]
    )
    qkv_interleaved = qkv_reassemble(qkv, config)
    torch.testing.assert_close(
        qkv_interleaved,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [16, 17, 18, 19],  # key
                [32, 33, 34, 35],  # value
                [4, 5, 6, 7],  # query
                [20, 21, 22, 23],  # key
                [36, 37, 38, 39],  # value
                [8, 9, 10, 11],  # query
                [24, 25, 26, 27],  # key
                [40, 41, 42, 43],  # value
                [12, 13, 14, 15],  # query
                [28, 29, 30, 31],  # key
                [44, 45, 46, 47],  # value
            ]
        ),
    )

    # GQA
    config = Config(n_embd=4, n_head=4, n_query_groups=2)
    qkv = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [4, 5, 6, 7],  # query
            [8, 9, 10, 11],  # query
            [12, 13, 14, 15],  # query
            [16, 17, 18, 19],  # key
            [20, 21, 22, 23],  # key
            [24, 25, 26, 27],  # value
            [28, 29, 30, 31],  # value
        ]
    )
    qkv_interleaved = qkv_reassemble(qkv, config)
    torch.testing.assert_close(
        qkv_interleaved,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [4, 5, 6, 7],  # query
                [16, 17, 18, 19],  # key
                [24, 25, 26, 27],  # value
                [8, 9, 10, 11],  # query
                [12, 13, 14, 15],  # query
                [20, 21, 22, 23],  # key
                [28, 29, 30, 31],  # value
            ]
        ),
    )

    # MQA
    config = Config(n_embd=4, n_head=4, n_query_groups=1)
    qkv = torch.tensor(
        [
            [0, 1, 2, 3],  # query
            [4, 5, 6, 7],  # query
            [8, 9, 10, 11],  # query
            [12, 13, 14, 15],  # query
            [16, 17, 18, 19],  # key
            [20, 21, 22, 23],  # value
        ]
    )
    qkv_interleaved = qkv_reassemble(qkv, config)
    torch.testing.assert_close(
        qkv_interleaved,
        torch.tensor(
            [
                [0, 1, 2, 3],  # query
                [4, 5, 6, 7],  # query
                [8, 9, 10, 11],  # query
                [12, 13, 14, 15],  # query
                [16, 17, 18, 19],  # key
                [20, 21, 22, 23],  # value
            ]
        ),
    )
