# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type, Union, Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import litgpt
from litgpt.config import Config as BaseConfig
from litgpt.model import GPT as BaseModel
from litgpt.model import Block as BaseBlock
from litgpt.model import CausalSelfAttention as BaseCausalSelfAttention
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights
from litgpt.kvcache.base import KVCache


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(weight_data, requires_grad=False, **weight.__dict__)
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(device=self.linear.weight.data.device)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora


class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        head_size: int,
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            head_size: size of a single attention head
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `litgpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.head_size = head_size
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(torch.empty((r * sum(enable_lora), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
                # might not be equal to `head_size * n_head`, thus we use it directly here
                head_size * n_head * enable_q,
                head_size * n_query_groups * enable_k,
                head_size * n_query_groups * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(torch.empty(sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            self.reset_parameters()

    @property
    def lora_ind(self) -> torch.Tensor:
        """Lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used."""
        # Indices are needed to properly pad weight updates with zeros.
        if not hasattr(self, "_lora_ind"):
            enable_q, enable_k, enable_v = self.enable_lora
            kv_embd_size = self.linear.in_features // (self.n_head // self.n_query_groups)
            lora_ind = []
            if enable_q:
                lora_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                lora_ind.extend(range(self.linear.in_features, self.linear.in_features + kv_embd_size))
            if enable_v:
                lora_ind.extend(range(self.linear.in_features + kv_embd_size, self.linear.out_features))
            self.register_buffer(
                "_lora_ind", torch.tensor(lora_ind, device=self.linear.weight.device), persistent=False
            )

        return self._lora_ind

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad the last dimension of weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)

        result = x.new_zeros(*x.shape[:-1], self.linear.out_features)  # (64, 64, 384)
        if result.device.type == "mps":
            result[..., self.lora_ind] = x
            return result
        else:
            return result.index_copy_(dim=-1, index=self.lora_ind, source=x)  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `litgpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            return F.conv1d(input, weight, groups=sum(self.enable_lora))  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)],
            dim=1,  # (B, C_output', T)
        )  # (B, C_output, T)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        lora = self.conv1d(
            self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
            self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).squeeze(
            0
        )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
        return self.zero_pad(lora.T * self.scaling).T  # (256, 128) after zero_pad (384, 128)

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            super().merge()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(
            -2, -1
        )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        lora = self.zero_pad(after_B) * self.scaling  # (64, 64, 256) after zero_pad (64, 64, 384)
        return pretrained + lora


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


@dataclass
class Config(BaseConfig):
    """
    Args:
        lora_r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        lora_*: whether to apply LoRA to the specified weights or not
    """

    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_query: bool = False
    lora_key: bool = False
    lora_value: bool = False
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False

    @property
    def mlp_class(self) -> Type:
        return getattr(litgpt.lora, self.mlp_class_name)


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
        self.lm_head = create_lora_linear(
            config,
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            use_r=config.lora_head,
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
        if isinstance(module, LoRALinear):
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
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        # key, query, value projections for all heads, but in a batch
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.qkv = LoRAQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias or config.attn_bias,
            # for MQA/GQA support
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.proj = create_lora_linear(
            config,
            config.head_size * config.n_head,
            config.n_embd,
            use_r=config.lora_projection,
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

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.linear.{attr}"
            current_key = f"{prefix}qkv.linear.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def create_lora_linear(
    config: Config,
    in_size: int,
    out_size: int,
    bias: Optional[Union[float, bool]] = None,
    use_r: Optional[bool] = None,
) -> LoRALinear:
    if bias is None:
        bias = config.bias
    if use_r is None:
        use_r = config.lora_mlp
    return LoRALinear(
        in_size,
        out_size,
        bias=bias,
        r=(config.lora_r if use_r else 0),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )


class GptNeoxMLP(litgpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = create_lora_linear(
            config, config.n_embd, config.intermediate_size
        )
        self.proj = create_lora_linear(
            config, config.intermediate_size, config.n_embd
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
        self.fc_1 = create_lora_linear(
            config, config.n_embd, config.intermediate_size
        )
        self.fc_2 = create_lora_linear(
            config, config.n_embd, config.intermediate_size
        )
        self.proj = create_lora_linear(
            config, config.intermediate_size, config.n_embd
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
        self.gate = create_lora_linear(
            config, config.n_embd, config.n_expert, bias=False
        )
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))
        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"gate.weight": "gate.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_lora_weights(model: GPT) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
