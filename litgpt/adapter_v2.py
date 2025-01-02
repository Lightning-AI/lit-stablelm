# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation of the paper:

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010

Port for LitGPT
"""

from dataclasses import dataclass
from typing import Any, Dict, Type, Optional, List

import torch
import torch.nn as nn
from typing_extensions import Self

import litgpt
from litgpt.adapter import GPT as BaseModel
from litgpt.model import Block as BaseBlock
from litgpt.adapter import CausalSelfAttention as BaseCausalSelfAttention
from litgpt.adapter import Config as BaseConfig
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights
from litgpt.kvcache.base import KVCache


@dataclass
class Config(BaseConfig):
    @property
    def mlp_class(self) -> Type:
        return getattr(litgpt.adapter_v2, self.mlp_class_name)


def adapter_filter(key: str, value: Any) -> bool:
    adapter_substrings = (
        # regular adapter v1 parameters
        "adapter_wte",
        "gating_factor",
        # adapter v2: new bias and scale used in Linear
        "adapter_scale",
        "adapter_bias",
        # adapter v2: Norm parameters are now trainable
        "norm_1",
        "norm_2",
        "ln_f",
    )
    return any(s in key for s in adapter_substrings)


class AdapterV2Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.adapter_bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.adapter_scale = torch.nn.Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter_scale * (self.linear(x) + self.adapter_bias)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.adapter_bias)
        nn.init.ones_(self.adapter_scale)


class GPT(BaseModel):
    # Copy & paste from :class:`model.GPT`. Note that :class:`Block` is new here.
    def __init__(
        self,
        config: Config,
        kv_cache: Optional[List[KVCache]] = None
    ) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        if kv_cache is not None:
            if len(kv_cache) != config.n_layer:
                raise ValueError(f"kv_cache length {len(kv_cache)} != {config.n_layer} = config.n_layer")
            for kvc in kv_cache:
                self._check_kv_cache(config, kvc)
            self._default_kv_cache = False
        else:
            kv_cache = [None] * config.n_layer
            self._default_kv_cache = True
        self.lm_head = AdapterV2Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, block_idx, kv_cache=kvc)
                    for block_idx, kvc in enumerate(kv_cache)
                ),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.max_seq_length = self.config.block_size

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, AdapterV2Linear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight", "lm_head.bias": "lm_head.linear.bias"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)
        self.mlp = config.mlp_class(config)


class CausalSelfAttention(BaseCausalSelfAttention):
    """A modification of `litgpt.adapter.CausalSelfAttention` that uses the Adapter V2 Linear class"""

    # Copy&paste from :class:`model.CausalSelfAttention`
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        # key, query, value projections for all heads, but in a batch
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.qkv = AdapterV2Linear(
            in_features=config.n_embd,
            out_features=shape,
            bias=config.bias or config.attn_bias
        )
        # output projection
        self.proj = AdapterV2Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base and/or legacy checkpoints."""
        mapping = {
            "qkv.weight": "qkv.linear.weight",
            "qkv.bias": "qkv.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        # For compatibility with older checkpoints
        if (key := prefix + "gating_factor") in state_dict and state_dict[key].size(1) == self.config.n_head:
            state_dict[key] = state_dict[key].permute(0, 2, 1, 3)

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.linear.{attr}"
            current_key = f"{prefix}qkv.linear.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(litgpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = AdapterV2Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = AdapterV2Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(litgpt.model.LLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = AdapterV2Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.fc_2 = AdapterV2Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = AdapterV2Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(litgpt.model.LLaMAMoE):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.gate = AdapterV2Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"gate.weight": "gate.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def mark_only_adapter_v2_as_trainable(model: GPT) -> None:
    """Sets requires_grad=False for all non-adapter weights"""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)
