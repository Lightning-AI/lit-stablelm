# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from multiprocessing.managers import Value
from typing import Any, Optional, Tuple, Union, List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from litgpt.config import Config
from litgpt.kvcache import (
    DefaultKeysAndValues,
    DenseKVCache,
    KeysAndValues,
    KVCache,
    KVCacheParams,
)
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble


class GPT(nn.Module):
    def __init__(
        self,
        config: Config,
        kv_cache: Optional[List[KVCache]] = None
    ) -> None:
        """
        Args:
            config: Configuration parameters
            kv_cache: KV caches to be used for inference, one for each layer.
                If not given, a dense KV cache of :class:`DenseKVCache` is
                created with the first inference :meth:`forward` call, using
                `max_seq_length` as size. This could result in an out of memory
                error. For models to be used for inference, we recommend
                passing KV caches here.
        """
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        if kv_cache is not None:
            if len(kv_cache) != config.n_layer:
                raise ValueError(f"kv_cache length {len(kv_cache)} != {config.n_layer} = config.n_layer")
            for kvc in kv_cache:
                self._check_kv_cache(config, kvc)
            self._default_kv_cache = False
        else:
            # Default KV caches will be created once first required, or
            # if `set_kv_cache` is called.
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

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory.

        If KV caches are of type `DenseKVCache`, they are resized here if too
        small.
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        # RoPE cache
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif self.cos.size(0) < value:
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # KV caches
        # Only need to do something for :class:`DenseKVCache` caches, all others
        # are supposed to be independent of `max_seq_length`.
        for l_ix, block in enumerate(self.transformer.h):
            attn = block.attn
            kv_cache = attn.kv_cache
            if kv_cache is not None and isinstance(kv_cache, DenseKVCache) and kv_cache.cache_length < value:
                print(f"KV cache for layer {l_ix} too small: Reallocating")
                attn.create_default_kv_cache(
                    batch_size=kv_cache.batch_size,
                    device=kv_cache.device,
                    dtype=kv_cache.dtype,
                    max_sequence_length=value
                )
        # Mask cache
        if self.mask_cache is None or self.mask_cache.shape[-1] < value:
            self.mask_cache = build_mask_cache(
                value,
                sliding_window_size=self.config.sliding_window_size,
                device=self.cos.device,
            )

    def set_kv_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        This method can be called only if KV caches have not been passed at
        construction. It creates default (dense) KV caches for every layer.

        Calling this method is optional, default KV caches are otherwise
        created when :meth:`forward` is called for inference for the first
        time (i.e., with `for_prefill=True`).

        Args:
            batch_size: Inference batch size
            device: Device for buffers
            dtype: Data type for buffers
            max_sequence_length: Cache length. If not given, we use
                `self.max_seq_length`
        """
        if not self._default_kv_cache:
            raise ValueError("KV caches have already been passed at construction")
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        for block in self.transformer.h:
            attn = block.attn
            kv_cache = attn.kv_cache
            if (
                kv_cache is None or
                kv_cache.batch_size != batch_size or
                kv_cache.cache_length != max_seq_length or
                kv_cache.device != device or
                kv_cache.dtype != dtype
            ):
                if kv_cache is not None:
                    device = kv_cache.device if device is None else device
                    dtype = kv_cache.dtype if dtype is None else dtype
                attn.create_default_kv_cache(
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_seq_length,
                )

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)
        self.mask_cache = build_mask_cache(
            self.max_seq_length,
            sliding_window_size=self.config.sliding_window_size,
            device=self.cos.device,
        )

    @staticmethod
    def _check_kv_cache(
        config: Config,
        kv_cache: KVCache,
    ):
        params = kv_cache.get_params()
        if config.n_query_groups != params.n_query_groups:
            raise ValueError(f"config and kv_cache not compatible: config.n_query_groups = {config.n_query_groups} != {params.n_query_groups} = kv_cache.n_query_groups")
        if config.n_head != params.n_head:
            raise ValueError(f"config and kv_cache not compatible: config.n_head = {config.n_head} != {params.n_head} = kv_cache.n_head")
        head_size = config.n_embd // config.n_head
        if head_size != params.head_size:
            raise ValueError(f"config and kv_cache not compatible: config.head_size = {head_size} != {params.head_size} = kv_cache.head_size")

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[int] = None,
        for_prefill: bool = False,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        There are three different contexts in which this method is called:
        - Training: `input_pos=None`, `for_prefill=False`. KV cache not needed.
        - Inference, prefill: `input_pos=None`, `for_prefill=True`. Same as
          training, but KV cache is initialized with K and V vectors. If KV
          caches are not present, they are created here (using default dense KV
          caches which store all K, V information).
        - Inference, token generation: `input_pos` given, `for_prefill`
          is ignored. KV caches must be given, they are used and updated. We
          check that`input_pos == kv_cache.next_token_pos`. Note that `T > 1`
          is permitted here as well.

        Token generation and `T > 1`:

        This situation is non-standard, since `idx` needs to provide tokens at
        positions `input_pos:(input_pos + T)`, whereas the logits are for
        generating tokens at `(input_pos + 1):(input_pos + T + 1)`, so only the
        last position is needed to generate a new token. Use cases:
        - Updating KV caches sequentially if prompt size is larger than max
          prefill length of cache
        - Speculative decoding. Here, `idx` comes from the cheaper proposal
          model, and the logits are needed for the accept/reject probabilities.

        Args:
            idx: Token indices of input sequences, shape `(B, T)`, where `B`
                is batch size.
            input_pos: See above. Defaults to `None`
            for_prefill: See above. Defaults to `False`
            lm_head_chunk_size: Optional. If `lm_head_chunk_size > 0`, the final
                `lm_head` computation is done in chunks of this size.

        Returns:
            Logit outputs, shape `(B, T, config.padded_vocab_size)`. If
            `lm_head_chunk_size > 0`, this is a list of chunks of shape
            `(B, lm_head_chunk_size, config.padded_vocab_size)`, the final
            entry can be shorter.

        """
        if idx.ndim == 1:
            idx = idx.unsqueeze(0)
        elif idx.ndim != 2:
            raise ValueError(f"idx must be 1D or 2D tensor, but idx.shape = {idx.shape}")
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        if input_pos is not None:
            # Few tokens generation. This needs a KV cache. If none is passed
            # at construction, the caches are created with the first call with
            # `for_prefill=True`.
            for l_ix, block in enumerate(self.transformer.h):
                kv_cache = block.attn.kv_cache
                if kv_cache is None or kv_cache.next_token_pos is None:
                    raise ValueError(f"KV cache for layer {l_ix} is missing or not initialized. You need to start with pre-filling (for_prefill=True)")
                if kv_cache.next_token_pos != input_pos:
                    raise ValueError(f"KV cache for layer {l_ix}: input_pos = {input_pos} != {self.kv_cache.next_token_pos} = kv_cache.next_token_pos")
                if kv_cache.max_tokens_forward < T:
                    raise ValueError(f"KV cache for layer {l_ix}: T = {T}, must be <= max_tokens_forward = {kv_cache.max_tokens_forward}")
            for_prefill = False

            if self.config.rope_n_elem > 0:
                input_pos_array = torch.arange(input_pos, input_pos + T, device=self.cos.device, dtype=torch.int64)
                cos = batched_index_select(self.cos, 0, input_pos_array).unsqueeze(0)
                sin = batched_index_select(self.sin, 0, input_pos_array).unsqueeze(0)
            else:
                cos = sin = None
        else:
            # Unsqueeze to have a batch dimension
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
            # `cos`, `sin` have shape (1, T, config.rope_n_elem)

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd ** 0.5, dtype=x.dtype)

        for l_ix, block in enumerate(self.transformer.h):
            if for_prefill:
                # Create default KV caches if not present, or if batch size of
                # cache is too small (latter only if default cache)
                batch_size = x.shape[0]
                create_def_cache = False
                attn = block.attn
                if attn.kv_cache is None:
                    print(f"Allocating KV cache for layer {l_ix}")
                    create_def_cache = True
                elif attn.kv_cache.batch_size < batch_size:
                    if self._default_kv_cache:
                        print(f"Re-allocating KV cache for layer {l_ix} (batch size was too small)")
                        create_def_cache = True
                    else:
                        raise ValueError(f"Batch size {batch_size} is too large for KV cache layer {l_ix} (batch size {attn.kv_cache.batch_size})")
                if create_def_cache:
                    # Same device and dtype as input `x`
                    attn.create_default_kv_cache(
                        batch_size=batch_size,
                        device=x.device,
                        dtype=x.dtype,
                        max_sequence_length=self.max_seq_length,
                    )
            x = block(x, cos, sin, input_pos, for_prefill, self.mask_cache)

        x = self.transformer.ln_f(x)
        clamp_head = partial(
            do_softcapping, thresh=self.config.final_logit_softcapping
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [
                clamp_head(self.lm_head(x_i))
                for x_i in x.split(lm_head_chunk_size, dim=1)
            ]
        else:
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_adjustments is None:
            extra_config = None
        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param for param, present in zip(adjusted_params_required, params_present) if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def clear_kv_cache(self) -> None:
        """
        Note that KV cache objects are removed only if none have been passed at
        construction.
        """
        self.mask_cache = None
        if self._default_kv_cache:
            for block in self.transformer.h:
                block.attn.kv_cache = None

    def get_kv_cache_params(self) -> Optional[KVCacheParams]:
        kv_cache = self.transformer.h[0].attn.kv_cache
        return None if kv_cache is None else kv_cache.get_params()

    def kv_cache_max_tokens_forward(self) -> Optional[int]:
        caches = [layer.attn.kv_cache for layer in self.transformer.h]
        if any(cache is None for cache in caches):
            return None
        else:
            return min(cache.max_tokens_forward for cache in caches)

    def kv_cache_max_prefill_length(self) -> Optional[int]:
        caches = [layer.attn.kv_cache for layer in self.transformer.h]
        if any(cache is None for cache in caches):
            return None
        else:
            mlps = [kvc.max_prefill_length for kvc in caches]
            if all(mlp is None for mlp in mlps):
                return None
            else:
                return min(mlp for mlp in mlps if mlp is not None)


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[int] = None,
        for_prefill: bool = False,
        mask_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(
            x_normed,
            cos=cos,
            sin=sin,
            input_pos=input_pos,
            for_prefill=for_prefill,
            mask_cache=mask_cache,
        )
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)
        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # KV cache (needed for inference)
        self.kv_cache = kv_cache
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_stride == 0
        )

        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size * config.n_head, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size * config.n_query_groups, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        input_pos: Optional[int] = None,
        for_prefill: bool = False,
        mask_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            cos: RoPE parameters
            sin: RoPE parameters
            input_pos: See :meth:`GPT.forward`
            for_prefill: See :meth:`GPT.forward`
            mask_cache: Used for building mask in special case

        Returns:
            Output tensor
        """
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - hs         | head size
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        if input_pos is not None:
            if self.kv_cache is None or self.kv_cache.next_token_pos is None:
                raise ValueError("If input_pos is given, KV cache must exist and be initialized (call with for_prefill=True)")
            if self.kv_cache.next_token_pos != input_pos:
                raise ValueError(f"input_pos = {input_pos} != {self.kv_cache.next_token_pos} = kv_cache.next_token_pos")
            if self.kv_cache.max_tokens_forward < T:
                raise ValueError(
                    f"T = {T}, must be <= max_tokens_forward = {self.kv_cache.max_tokens_forward}")
            for_prefill = False
        elif for_prefill:
            # Sanity check
            if self.kv_cache is None:
                raise ValueError("If for_prefill is True, KV cache must exist")

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram in `lit_gpt/config.py::Config`).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        if self.config.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh_q), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        # Note that `nh_k` can be smaller than `nh_q` (but the latter must be a
        # multiple of the former). This works with the
        # `scaled_dot_product_attention` implementations below.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_k, T, hs)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        if rope_n_elem > 0:
            q_roped = apply_rope(q[..., : rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., : rope_n_elem], cos, sin)
            q = torch.cat((q_roped, q[..., rope_n_elem :]), dim=-1)  # (B, nh_q, T, hs)
            k = torch.cat((k_roped, k[..., rope_n_elem :]), dim=-1)  # (B, nh_k, T, hs)

        if input_pos is not None:
            # Extend KV cache and retrieve key, value tensors to be used.
            # Instead of asking for the key and value tensors as such,
            # `k_and_v` allows access to them. Since they are never needed at
            # the same time, this can save memory.
            k_and_v = self.kv_cache(k, v)
            # k, v: (B, nh_k, cache_length, hs)
        else:
            if for_prefill:
                # Prefill KV cache
                self.kv_cache.prefill(key=k, value=v)
            # In this case, `k_and_v` can vend both keys and values at the same
            # time.
            k_and_v = DefaultKeysAndValues(k, v)

        # We need the attention mask if there is sliding window attention,
        # or if `input_pos` is given and T > 1.
        is_causal = input_pos is None
        use_mask = self.apply_sliding_window_attention or (not is_causal and T > 1)
        mask = None
        if use_mask:
            # Special case requires building a mask. `mask_cache` is only needed
            # then.
            assert mask_cache is not None, "mask_cache must be given if sliding window attention is used, or if input_pos given and T > 1"
            if is_causal:
                mask = mask_cache[:T, :T].view(1, 1, T, T)
                is_causal = False
            else:
                # We need a mask if T > 1, since inference needs to be causal
                # for the new tokens
                mask = batched_index_select(
                    mask_cache[:, input_pos:(input_pos + T)],
                    dim=0,
                    idx=self.kv_cache.token_positions(),
                )

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        return_scores = (not is_causal) and self.kv_cache.update_requires_attn_weights()
        y, scores = self.scaled_dot_product_attention(
            q, k_and_v, mask, is_causal, return_scores
        )

        if return_scores:
            # Pass attention weights to KV cache
            self.kv_cache.update(attn_weights=scores)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k_and_v: KeysAndValues,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert mask is None or not is_causal, "Cannot have mask and is_causal=True"
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # We cannot call PyTorch scaled_dot_product_attention if:
        # - Attention scores need to be returned; or
        # - Logit softcapping is required; or
        # - We cannot access keys and values from `k_and_v` in parallel (this
        #   never happens if `is_causal == True`)
        if return_scores or self.config.attention_logit_softcapping is not None or not k_and_v.both_in_parallel():
            y, scores = scaled_dot_product_attention(
                query=q,
                k_and_v=k_and_v,
                scale=scale,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
                is_causal=is_causal,
            )
            if not return_scores:
                scores = None
        else:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            key = k_and_v.keys()
            value = k_and_v.values()
            for retry in range(2):
                try:
                    y = F.scaled_dot_product_attention(
                        query=q,
                        key=key,
                        value=value,
                        attn_mask=mask,
                        dropout_p=0.0,
                        scale=scale,
                        is_causal=is_causal,
                    )
                    break
                except RuntimeError as ex:
                    if retry == 1 or self.config.n_query_groups == self.config.n_head:
                        raise ex  # Re-throw
                    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
                    # `scaled_dot_product_attention` is supposed to support
                    # `query.shape = (bs, nh_q, ...), key.shape = (bs, nh_k, ...)`
                    # and `nh_k < nh_q` if `nh_q` is a multiple of `nh_k`. But
                    # this seems not yet supported (in 2.5.1), so have to lift
                    # K, V here. This is annoying, as it wastes memory.
                    q_per_kv = self.config.n_head // self.config.n_query_groups
                    key = key.repeat_interleave(q_per_kv, dim=1)
                    value = value.repeat_interleave(q_per_kv, dim=1)
            scores = None
        return y.transpose(1, 2), scores

    def create_default_kv_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ):
        self.kv_cache = DenseKVCache(
            config=self.config,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.fc_2 = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def _attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    assert query.ndim == key.ndim == 4
    assert query.shape[0] == key.shape[0] and query.shape[3] == key.shape[3]
    nh_q = query.shape[1]
    nh_k = key.shape[1]
    assert nh_q % nh_k == 0
    # - query: (bs, nh_q, T_q, hs)
    # - key: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    key_transposed = key.mT  # (bs, nh_k, hs, T_k)
    if q_per_kv == 1:
        return query @ key_transposed
    else:
        assert q_per_kv > 1
        q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
        _query = query.view(*q_shape)
        key_transposed = key_transposed.unsqueeze(2)
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, T_q, hs)
        # - key_transposed: (bs, nh_k, 1, hs, T_k)
        # - scores: (bs, nh_k, q_per_kv, T_q, T_k)
        scores = torch.matmul(_query, key_transposed)
        s_shape = query.shape[:-1] + (key.shape[2],)
        return scores.view(*s_shape)


def _attention_compute_weighted_values(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert scores.ndim == value.ndim == 4
    assert scores.shape[0] == scores.shape[0] and scores.shape[3] == value.shape[2]
    nh_q = scores.shape[1]
    nh_k = value.shape[1]
    assert nh_q % nh_k == 0
    # - scores: (bs, nh_q, T_q, T_k)
    # - value: (bs, nh_k, T_k, hs)
    q_per_kv = nh_q // nh_k
    if q_per_kv == 1:
        return scores @ value
    else:
        s_shape = scores.shape[:1] + (nh_k, q_per_kv) + scores.shape[2:]
        _scores = scores.view(*s_shape)
        _value = value.unsqueeze(2)
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, T_q, T_k)
        # - _value: (bs, nh_k, 1, T_k, hs)
        # - result: (bs, nh_k, q_per_kv, T_q, hs)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (value.shape[-1],)
        return result.view(*r_shape)


def scaled_dot_product_attention(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    attention_logit_softcapping: Optional[float] = None,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = query.dtype
    key = k_and_v.keys()
    scores = _attention_compute_scores(query, key) * scale
    scores = do_softcapping(scores, attention_logit_softcapping)
    if mask is None and is_causal:
        T = query.shape[2]
        assert key.size(2) == T, "is_causal=True only if query, key have same size"
        mask = torch.ones(T, T, dtype=dtype, device=query.device).triu(diagonal=1)
        mask.masked_fill_(mask.bool(), torch.finfo(query.dtype).min)
        mask = mask.view(1, 1, T, T)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=dtype)
    value = k_and_v.values()
    return _attention_compute_weighted_values(scores, value), scores


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.

    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    # TODO: Remove `> 1` once HF bug is fixed!
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(
    t: torch.Tensor,
    dim: int,
    idx: torch.Tensor
) -> torch.Tensor:
    """index_select for batched index and unbatched t"""
    if idx.ndim == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    if dim > 0:
        # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
        dims = [dim] + list(range(res.ndim))
        del dims[dim + 1]
        res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE transform to `x`. Note that `cos`, `sin` need to have a batch
    dimension.

    Args:
        x: Input tensor, `(B, ..., T, head_size)`
        cos: Cached cosines, `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, `(B, T, head_size)` or `(1, T, head_size)`

    Returns:
        Encoded tensor, `(B, ..., T, head_size)`
    """
    if cos.ndim != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., : head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half :]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.ndim - cos.ndim
    if dims_diff > 0:
        # Ensure that shapes of `x`, `cos`, `sin` align
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def do_softcapping(x: torch.Tensor, thresh: Optional[float]) -> torch.Tensor:
    if thresh is not None:
        return torch.tanh(x / thresh) * thresh
    else:
        return x


def build_mask_cache(
    max_seq_length: int,
    sliding_window_size: Optional[int],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
          Global Window              Sliding window             Sliding window
          attention mask      +            bias          =      attention mask
    ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
    │ True False False False │  │ True  True  True True │  │ True  False False False │
    │ True True  False False │  │ True  True  True True │  │ True  True  False False │
    │ True True  True  False │  │ False True  True True │  │ False True  True  False │
    │ True True  True  True  │  │ False False True True │  │ False False True  True  │
    └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
    """
    # Usual causal mask:
    mask = torch.ones(max_seq_length, max_seq_length, device=device).triu(diagonal=1)
    mask.masked_fill_(mask.bool(), float("-inf"))
    if sliding_window_size is not None:
        sliding_window_bias = torch.ones_like(mask).tril(diagonal=-sliding_window_size)
        sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
        mask += sliding_window_bias
    return mask


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
