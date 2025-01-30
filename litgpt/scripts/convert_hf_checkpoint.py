# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import json
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor
from tqdm import tqdm

from litgpt.config import Config
from litgpt.utils import extend_checkpoint_dir, incremental_save, lazy_load, save_config
from safetensors.torch import load_file as load_safetensors


def copy_weights_gpt_neox(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "gpt_neox.embed_in.weight": "transformer.wte.weight",
        "gpt_neox.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "gpt_neox.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "gpt_neox.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.qkv.bias",
        "gpt_neox.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.qkv.weight",
        "gpt_neox.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
        "gpt_neox.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "gpt_neox.layers.{}.attention.rotary_emb.inv_freq": None,
        "gpt_neox.layers.{}.attention.bias": None,
        "gpt_neox.layers.{}.attention.masked_bias": None,
        "gpt_neox.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "gpt_neox.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": "transformer.h.{}.mlp.fc.bias",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": "transformer.h.{}.mlp.proj.bias",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "gpt_neox.final_layer_norm.bias": "transformer.ln_f.bias",
        "gpt_neox.final_layer_norm.weight": "transformer.ln_f.weight",
        "embed_out.weight": "lm_head.weight",
    }

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

    for from_name, param in hf_weights.items():
        name_template, layer_idx = layer_template(from_name)
        to_name = weight_map[name_template]
        if to_name is None:
            continue
        to_name = to_name.format(layer_idx)
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if from_name.endswith((".query_key_value.weight", ".query_key_value.bias")):
            # Reassemble [q, k, v, q, k, v, ...] --> [q, q, ..., k, k, ..., v, v, ...]
            param = qkv_reassemble(param, config)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_falcon(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "transformer.word_embeddings.weight": "transformer.wte.weight",
        "transformer.h.{}.self_attention.query_key_value.weight": "transformer.h.{}.attn.qkv.weight",
        "transformer.h.{}.self_attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "transformer.h.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "transformer.h.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if "7b" in config.name:
        weight_map.update(
            {
                "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            }
        )
    elif "40b" in config.name or "180B" in config.name:
        weight_map.update(
            {
                "transformer.h.{}.ln_attn.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.ln_attn.weight": "transformer.h.{}.norm_1.weight",
                "transformer.h.{}.ln_mlp.bias": "transformer.h.{}.norm_2.bias",
                "transformer.h.{}.ln_mlp.weight": "transformer.h.{}.norm_2.weight",
            }
        )
    else:
        raise NotImplementedError

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

    for from_name, param in hf_weights.items():
        name_template, layer_idx = layer_template(from_name)
        to_name = weight_map[name_template].format(layer_idx)
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if from_name.endswith((".query_key_value.weight", ".query_key_value.bias")):
            # Reassemble [q, k, v, q, k, v, ...] --> [q, q, ..., k, k, ..., v, v, ...]
            param = qkv_reassemble(param, config)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "transformer.h.{}.mlp.experts.{}.fc_1.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "transformer.h.{}.mlp.experts.{}.fc_2.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "transformer.h.{}.mlp.experts.{}.proj.weight",
            }
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for from_name, param in hf_weights.items():
        name_template, *ids = layer_template(from_name, num_matches=2)
        to_name = weight_map[name_template]
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
            qkv = qkv_weights.setdefault(ids[0], defaultdict(dict))
            weight_name, weight_type = from_name.split(".")[-2:]
            qkv[weight_type][weight_name] = param
        if to_name is None:
            continue
        to_name = to_name.format(*ids)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is splitted across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            qkv = torch.cat((q, k, v))
            state_dict[f"transformer.h.{i}.attn.qkv.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_gemma_2(
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.post_attention_norm.weight",
        "model.layers.{}.pre_feedforward_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.post_feedforward_layernorm.weight": "transformer.h.{}.post_mlp_norm.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for from_name, param in hf_weights.items():
        name_template, *ids = layer_template(from_name, num_matches=2)
        to_name = weight_map[name_template]
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
            qkv = qkv_weights.setdefault(ids[0], defaultdict(dict))
            weight_name, weight_type = from_name.split(".")[-2:]
            qkv[weight_type][weight_name] = param
        if to_name is None:
            continue
        to_name = to_name.format(*ids)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is splitted across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            qkv = torch.cat((q, k, v))
            state_dict[f"transformer.h.{i}.attn.qkv.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_phi(
    config: Config,
    qkv_weights: dict,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    if any(layer_name.startswith(("layers.", "transformer.")) for layer_name in hf_weights):
        raise ValueError(
            "You are using an outdated Phi checkpoint. Please reload it as described in 'tutorials/download_phi.md'"
        )

    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "model.layers.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "model.layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "model.final_layernorm.weight": "transformer.ln_f.weight",
        "model.final_layernorm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }

    if config.name.startswith(("Phi-3", "phi-4")):
        weight_map.update(
            {
                "model.layers.{}.self_attn.qkv_proj.weight": "transformer.h.{}.attn.qkv.weight",
                "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
                "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
                "model.norm.weight": "transformer.ln_f.weight",
            }
        )

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for from_name, param in hf_weights.items():
        name_template, layer_idx = layer_template(from_name)
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
            qkv = qkv_weights.setdefault(layer_idx, defaultdict(dict))
            weight_name, weight_type = from_name.split(".")[-2:]
            qkv[weight_type][weight_name] = param
        elif from_name.endswith("gate_up_proj.weight"):
            weight = load_param(param, f"layer {layer_idx} gate_up_proj", dtype, verbose=debug_mode)
            fc_1, fc_2 = weight.chunk(2, dim=0)
            state_dict[f"transformer.h.{layer_idx}.mlp.fc_1.weight"] = fc_1
            state_dict[f"transformer.h.{layer_idx}.mlp.fc_2.weight"] = fc_2
            continue
        to_name = weight_map[name_template]
        if to_name is None:
            continue
        to_name = to_name.format(layer_idx)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is splitted across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            qkv = torch.cat((q, k, v))
            state_dict[f"transformer.h.{i}.attn.qkv.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

            if progress_per_file is not None:
                pbar.update(progress_per_file)

def copy_weights_qwen_2_5(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for from_name, param in hf_weights.items():
        name_template, *ids = layer_template(from_name, num_matches=2)
        to_name = weight_map[name_template]
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
            qkv = qkv_weights.setdefault(ids[0], defaultdict(dict))
            weight_name, weight_type = from_name.split(".")[-2:]
            qkv[weight_type][weight_name] = param
        if to_name is None:
            continue
        to_name = to_name.format(*ids)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is splitted across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            qkv = torch.cat((q, k, v))
            state_dict[f"transformer.h.{i}.attn.qkv.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_olmo2(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.self_attn.q_norm.weight": "transformer.h.{}.attn.q_norm.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_norm.weight": "transformer.h.{}.attn.k_norm.weight",
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "model.layers.{}.post_feedforward_layernorm.weight": "transformer.h.{}.post_mlp_norm.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights) + len(qkv_weights))

    for from_name, param in hf_weights.items():
        name_template, *ids = layer_template(from_name, num_matches=2)
        to_name = weight_map[name_template]
        param = load_param(param, from_name, dtype, verbose=debug_mode)
        if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
            qkv = qkv_weights.setdefault(ids[0], defaultdict(dict))
            weight_name, weight_type = from_name.split(".")[-2:]
            qkv[weight_type][weight_name] = param
        if to_name is None:
            continue
        to_name = to_name.format(*ids)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is splitted across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype, verbose=debug_mode)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype, verbose=debug_mode)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype, verbose=debug_mode)
            qkv = torch.cat((q, k, v))
            state_dict[f"transformer.h.{i}.attn.qkv.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

            if progress_per_file is not None:
                pbar.update(progress_per_file)


def qkv_reassemble(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reassemble from a normal to an interleaved placement in a QKV matrix.
    [Q, K, V, Q, K, V, ...] --> [Q, Q, ..., K, K, ..., V, V, ...]
    """
    q_per_kv = config.n_head // config.n_query_groups
    qs = []
    ks = []
    vs = []
    for chunk in torch.chunk(param, config.n_query_groups):
        split = torch.split(chunk, [config.head_size * q_per_kv, config.head_size, config.head_size])
        qs.append(split[0])
        ks.append(split[1])
        vs.append(split[2])
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)
    return torch.cat((q, k, v))



def layer_template(layer_name: str, num_matches: int = 1) -> Tuple[str, int]:
    pattern = r"\.(\d+)\."
    if not (search_res := re.findall(pattern, layer_name)):
        return layer_name, -1
    layer_name_template = re.sub(pattern, ".{}.", layer_name, count=num_matches)
    return layer_name_template, *(int(x) for x in search_res[:num_matches])


def load_param(
    param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype], verbose: bool =False
) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        if verbose:
            print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        if verbose:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param



@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    """
    Convert a Hugging Face Transformers checkpoint into a LitGPT compatible checkpoint.

    Arguments:
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to load. This is useful to download alternative weights of existing
            architectures.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        debug_mode: Prints the individual layers being loaded instead of a progress bar, which can be useful when
            developing and adding new models to LitGPT.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    save_config(config, checkpoint_dir)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, config)
    elif model_name.lower().startswith("gemma-2"):
        qkv_weights = {}
        copy_fn = partial(copy_weights_gemma_2, qkv_weights)
    elif model_name.lower().startswith("phi"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    elif model_name.lower().startswith(("qwen2.5","qwq")):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_qwen_2_5, config, qkv_weights)
    elif model_name.lower().startswith(("olmo-2-")):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_olmo2, config, qkv_weights)
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    else:
        copy_fn = partial(copy_weights_gpt_neox, config)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    model_safetensor_map_json_path = checkpoint_dir / "model.safetensors.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif model_safetensor_map_json_path.is_file():
        with open(model_safetensor_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_dir.glob("*.bin")) | set(checkpoint_dir.glob("*.safetensors"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin or .safetensors files")

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end

        if not debug_mode:
            # Using tqdm progress bar when not in debug mode

            total_size = max(1, sum(os.path.getsize(bin_file) for bin_file in bin_files))
            total_progress = 100

            with tqdm(
                total=total_progress,
                desc="Initializing",
                bar_format="{desc}{percentage:3.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}",
            ) as pbar:
                for bin_file in sorted(bin_files):
                    pbar.set_description(f"Loading weights: {bin_file.name}")
                    current_file_size = os.path.getsize(bin_file)
                    progress_per_file = (current_file_size / total_size) * total_progress

                    hf_weights = load_safetensors(bin_file) if bin_file.suffix == ".safetensors" else lazy_load(bin_file)
                    copy_fn(sd, hf_weights, saver=saver, dtype=dtype, pbar=pbar, progress_per_file=progress_per_file, debug_mode=debug_mode)
                gc.collect()

                if pbar.n < total_progress:
                    pbar.update(total_progress - pbar.n)
                pbar.close()
        else:
            # Handling files without progress bar in debug mode
            for bin_file in sorted(bin_files):
                hf_weights = load_safetensors(bin_file) if bin_file.suffix == ".safetensors" else lazy_load(bin_file)
                copy_fn(sd, hf_weights, saver=saver, dtype=dtype, debug_mode=debug_mode)

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        saver.save(sd)
