# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for LitGPT
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from typing_extensions import Self

from litgpt.config import Config as BaseConfig
from litgpt.model import GPT as BaseModel
from litgpt.model import Block as BaseBlock
from litgpt.model import CausalSelfAttention as BaseCausalSelfAttention
from litgpt.kvcache.base import KVCache, KeysAndValues, DefaultKeysAndValues


@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


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
        self.lm_head = nn.Linear(
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
        if isinstance(module, CausalSelfAttention):
            module.reset_parameters()


class Block(BaseBlock):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)


class CausalSelfAttention(BaseCausalSelfAttention):
    """A modification of `litgpt.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, 1, config.n_head, 1))
            # kv cache for inference
            self.adapter_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k_and_v: KeysAndValues,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y, scores = super().scaled_dot_product_attention(q, k_and_v, mask, is_causal, return_scores)
        if self.block_idx < self.config.adapter_start_layer:
            return y, scores

        aT = self.config.adapter_prompt_length
        if self.adapter_kv_cache is not None:
            # since this uses the wte weights as the prefix and the kv cache is only used during inference, ak and av
            # are the same every call
            ak, av = self.adapter_kv_cache
        else:
            prefix = self.adapter_wte.weight.reshape(1, aT, self.config.n_embd)
            aqkv = self.qkv(prefix)
            q_per_kv = self.config.n_head // self.config.n_query_groups
            aqkv = aqkv.view(1, aT, self.config.n_query_groups, q_per_kv + 2, self.config.head_size)
            aqkv = aqkv.permute(0, 2, 3, 1, 4)
            _, ak, av = aqkv.split((q_per_kv, 1, 1), dim=2)
            if self.config.n_query_groups != 1:
                # for MHA this is a no-op
                ak = ak.repeat_interleave(q_per_kv, dim=2)
                av = av.repeat_interleave(q_per_kv, dim=2)
            ak = ak.view(1, -1, aT, self.config.head_size)  # (1, nh_ak, aT, hs)
            av = av.view(1, -1, aT, self.config.head_size)  # (1, nh_av, aT, hs)
            self.adapter_kv_cache = (ak, av)

        T = q.size(2)
        amask = torch.ones(T, aT, dtype=torch.bool, device=q.device)
        a_k_and_v = DefaultKeysAndValues(keys=ak, values=av)
        ay, _ = super().scaled_dot_product_attention(
            q=q,
            k_and_v=a_k_and_v,
            mask=amask,
            is_causal=False,
        )
        return y + self.gating_factor * ay, scores

    def reset_parameters(self) -> None:
        if hasattr(self, "gating_factor"):
            torch.nn.init.zeros_(self.gating_factor)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with older checkpoints."""
        if (key := prefix + "gating_factor") in state_dict and state_dict[key].size(1) == self.config.n_head:
            state_dict[key] = state_dict[key].permute(0, 2, 1, 3)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def mark_only_adapter_as_trainable(model: GPT) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)


def adapter_filter(key: str, value: Any) -> bool:
    return "adapter_wte" in key or "gating_factor" in key
