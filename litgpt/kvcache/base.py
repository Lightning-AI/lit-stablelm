from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import torch

from litgpt.config import Config
from litgpt.kvcache.utils import bitsize_of, bits_for_torch_dtype


@dataclass(frozen=True)
class KVCacheParams:
    batch_size: int
    n_query_groups: int
    cache_length: int
    head_size: int
    n_head: int
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    @staticmethod
    def from_config(
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ) -> "KVCacheParams":
        if head_size is None:
            head_size = config.n_embd // config.n_head,
        return KVCacheParams(
            batch_size=batch_size,
            n_query_groups=config.n_query_groups,
            cache_length=cache_length,
            head_size=head_size,
            n_head=config.n_head,
            device=device,
            dtype=dtype,
        )


class KeysAndValues:
    """
    Object returned by :meth:`KVCache.forward`. Allows to access cached
    keys or values, but (in general) not both at the same time. Implementations
    may use the same buffer to return them in the methods below.

    However, if :meth:`both_in_parallel` returns `True`, the tensors returned
    by :meth:`keys` and :meth:`values` may be used in parallel, since they are
    supported by separate buffers.

    """
    def keys(self) -> torch.Tensor:
        """
        Returns:
            keys tensor from :meth:`KVCache.forward`, shape
            `(eff_batch_size, n_query_groups, T, head_size)`, where
            `T <= cache_length` is the current cache length)

        """
        raise NotImplementedError()

    def values(self) -> torch.Tensor:
        """
         Returns:
             values tensor from :meth:`KVCache.forward`, shape
             `(eff_batch_size, n_query_groups, T, head_size)`, where
             `T <= cache_length` is the current cache length)

         """
        raise NotImplementedError()

    @staticmethod
    def both_in_parallel() -> bool:
        """
        Returns:
            Can use both `keys` and `values` in parallel? Otherwise, can only
            use one of them at the same time
        """
        return False


class KVCache(torch.nn.Module):
    """
    Base class for key-value (KV) caches.

    Buffers have shapes
    `(batch_size, config.n_query_groups, cache_length, head_size)`, where
    `head_size` is a parameter. Caching can be used for
    batch size `1 <= eff_batch_size <= batch_size`, which is determined in
    the :meth:`prefill` call.

    A KV cache is used as follows. First, the buffer is prefilled by calling
    :meth:`prefill`, which also determines the effective batch size
    `eff_batch_size`. Then, new KV information is added sequentially by calling
    :meth:`forward`. The information can be for one or more tokens, up to
    `max_tokens_forward`. For some caches, :meth:`update` needs to be called
    after each :meth:`forward`, passing extra information. See for example
    :meth:`update_requires_attn_weights`.

    Note: In general, key tensors need to be position-encoded (e.g., RoPE).
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ):
        """
        Note that `batch_size` is the maximum batch size the cache can be used
        with. The effective batch size is determined when calling
        :meth:`prefill` and can change with any such call.

        Args:
            config: Model config
            batch_size: Inference batch size (maximum)
            cache_length: Number of slots in cache
            device: Device for buffers
            dtype: Data type for buffers
            head_size: Size of final dimension of buffers. Defaults to head
                size of model
        """
        super().__init__()
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")
        self.batch_size = batch_size
        self.n_query_groups = config.n_query_groups
        self.cache_length = cache_length
        if head_size is None:
            head_size = config.head_size
        self.head_size = head_size
        self.n_head = config.n_head
        self.device = device
        self.dtype = dtype
        # TODO: Remove once HuggingFace bug is fixed
        # https://github.com/huggingface/transformers/issues/35233
        self._work_around_hf_bug = config.rope_n_elem == 1

    @property
    def next_token_pos(self) -> Optional[int]:
        """
        Returns:
            Input position for next token to be generated, or `None` if cache
            has not been initialized yet (call of :meth:`prefill`).
        """
        raise NotImplementedError()

    @property
    def max_tokens_forward(self) -> int:
        """
        Returns:
            Maximum number of token positions which can be treated in
            :meth:`forward`. Depends on cache, but is `<= cache_length`

        """
        raise NotImplementedError()

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> KeysAndValues:
        """
        Accepts key and value tensors for `1 <= num <= max_tokens_forward`
        new token positions. These are written into the cache. If the cache
        is full, they overwrite slots for tokens to be evicted. In general,
        the eviction decision is taken in the last recent call of :meth:`update`.

        If a sequence is generated token by token, this method is always called
        with `num=1`. The case `num > 1` arises if large prompts are to be
        ingested with more than `max_prefill_length` tokens. Note that if the
        cache makes eviction decisions by scoring in :meth:`update`, then large
        `num` may lead to worse decisions. On the other hand, ingesting the
        prompt with larger `num` is faster.

        Args:
            key: New keys, `(eff_batch_size, n_query_groups, num, head_size)`,
                where `1 <= num <= max_tokens_forward`
            value: New values, `(eff_batch_size, n_query_groups, num, head_size)`

        Returns:
            key_cached, value_cached, `(eff_batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length

        """
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        """
        Some caches require this method to be called after each `forward`,
        passing extra information depending on the subclass. In general,
        this method updates internal scores and takes a decision which slot
        is evicted upon the next :meth:`forward` call (if the cache is full).

        One important example are KV caches based on the Heavy Hitter Oracle
        (H2O) proposal. These require the attention weights from the current
        MLA computation to be passed, and :meth:`update_requires_attn_weights`
        has to return `True`.

        Note: The extra information typically scales with `num`, the number of
        tokens :meth:`forward` was called for.

        Args:
            *args: Depends on subclass
            **kwargs: Depends on subclass
        """
        raise NotImplementedError()

    def update_requires_attn_weights(self) -> bool:
        """
        Returns:
            If `True`, :meth:`update` requires argument `attn_weights`, which
            passes current attention weights as
            `(eff_batch_size, n_query_groups, num, T)` tensor, where
            `T <= cache_length` is the current cache length, and `num` is the
            number of tokens in the last recent :meth:`forward` call.

        """
        return False

    @property
    def max_prefill_length(self) -> Optional[int]:
        """
        Returns:
            Maximum sequence length for `key`, `value` tensors passed to
            :meth:`prefill`. If there is no such maximum length, `None` is returned.

        """
        raise NotImplementedError()

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length must be
        `T <= max_prefill_length`. The effective batch size must be
        `eff_batch_size <= batch_size`. This batch size is then fixed for
        subsequent calls of :meth:`forward` and :meth:`update`.

        Args:
            key: Prefill keys, `(eff_batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(eff_batch_size, n_query_groups, T, head_size)`
        """
        raise NotImplementedError()

    def get_params(self) -> KVCacheParams:
        return KVCacheParams(
            batch_size=self.batch_size,
            n_query_groups=self.n_query_groups,
            cache_length=self.cache_length,
            head_size=self.head_size,
            n_head=self.n_head,
            device=self.device,
            dtype=self.dtype,
        )

    def token_positions(self) -> torch.Tensor:
        """
        Returns:
            Token positions in slots of the cache, shape
            `(eff_batch_size, n_query_groups, T)`.where `T <= cache_length`
            is the current cache length.
        """
        raise NotImplementedError()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        This is a theoretical estimate of the main buffers (which should all
        be allocated up front), it does not cover temporary storage used in
        the methods (make sure these are small compared to the main buffers).
        Also, real memory usage may be larger due to alignment issues.

        Returns:
            num_bits_total, bits_by_part (unit is bit)
        """
        raise NotImplementedError()

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        Same semantics as :meth:`size_estimate`, but can be called without a
        cache being created. Results may not be exactly the same, but should
        be very close.

        Args:
            params: KV cache parameters
            **kwargs: Extra arguments (optional)

        Returns:
            num_bits_total, bits_by_part (unit is bit)
        """
        raise NotImplementedError()

    def reset_parameters(self) -> None:
        pass


class DefaultKeysAndValues(KeysAndValues):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        # The final dimension of K and V can be different (in general)
        assert keys.shape[:-1] == values.shape[:-1] and keys.ndim == 4, (keys.shape, values.shape)
        self._keys = keys
        self._values = values

    def keys(self) -> torch.Tensor:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    @staticmethod
    def both_in_parallel() -> bool:
        """
        Keys and values are supported by different buffers, so they can be
        used at the same time.

        """
        return True


class DenseKVCache(KVCache):
    """
    Key-value cache for dense attention. Key and value tensors for all
    past tokens are maintained. The cache length is the maximum sequence
    length. This cache requires a lot of memory, it can only be used for
    moderate cache lengths.

    Note: If the cache is full, :meth:`forward` raises an exception. The cache
    buffers are allocated up front and are not enlarged later on.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
        head_size: Optional[int] = None,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            device: Device for buffers
            dtype: Data type for buffers
            max_sequence_length: Cache length. If not given, we use
            `config.block_size`
            max_tokens_forward: See parent class
            head_size: Size of final dimension of buffers. Defaults to head
                size of model

        """
        if max_sequence_length is None:
            max_sequence_length = config.block_size

        super().__init__(
            config, batch_size, max_sequence_length, device, dtype, head_size,
        )
        shape = (batch_size, self.n_query_groups, max_sequence_length, self.head_size)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            shape = shape[:-1] + (self.head_size + 1,)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.next_position = None
        self.eff_batch_size = None

    @property
    def next_token_pos(self) -> Optional[int]:
        return self.next_position

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    @property
    def max_prefill_length(self) -> Optional[int]:
        return self.cache_length

    @property
    def current_length(self) -> int:
        return self.next_position

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        num = key.shape[2]
        if not 1 <= num <= self.max_tokens_forward:
            raise ValueError(f"key.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        np = self.next_position
        if np + num > self.cache_length:
            raise IndexError(f"Cache has at most {self.cache_length - np} free slots, cannot add {num} entries")
        shape = (self.eff_batch_size, self.n_query_groups, num, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        if key.dtype != value.dtype:
            raise ValueError(f"key.dtype = {key.dtype} != {value.dtype} = value.dtype")
        # Move the buffer to the activation dtype for when AMP is used
        if key.dtype != self.dtype:
            self.dtype = key.dtype
            self.k = self.k.to(self.dtype)
            self.v = self.v.to(self.dtype)
        # Append new content to cache
        self.k[:self.eff_batch_size, :, np:(np + num), :] = key
        self.v[:self.eff_batch_size, :, np:(np + num), :] = value
        self.next_position += num
        return DefaultKeysAndValues(
            self.k[:self.eff_batch_size, :, :self.next_position, :],
            self.v[:self.eff_batch_size, :, :self.next_position, :],
        )

    def update(self, *args, **kwargs):
        pass

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        if init_length > self.cache_length:
            raise ValueError(f"key.shape[2] = {init_length}, must be at most {self.cache_length}")
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :init_length, :] = key
        self.v[:eff_batch_size, :, :init_length, :] = value
        self.next_position = init_length
        self.eff_batch_size = eff_batch_size

    def token_positions(self) -> torch.Tensor:
        return torch.arange(self.next_position, device=self.device).reshape(
            1, 1, -1
        ).expand(self.eff_batch_size, self.n_query_groups, -1)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        return sz_buffs, dict(buffers=sz_buffs)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        max_sequence_length = kwargs.get("max_sequence_length")
        if max_sequence_length is None:
            raise IndexError("Argument 'max_sequence_length' is missing")
        else:
            max_sequence_length = int(max_sequence_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * max_sequence_length * params.head_size
        sz_buffs = 2 * numel * bits_for_torch_dtype(dtype)
        return sz_buffs, dict(buffers=sz_buffs)


class MostRecentKVCache(KVCache):
    """
    Baseline key-value cache which stores the most recent `cache_length` key,
    value tensors.
    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
    ):
        """
        Args:
            config: Model config
            batch_size: Inference batch size
            cache_length: Number of slots of cache
            device: Device for buffers
            dtype: Data type for buffers
            head_size: Size of final dimension of buffers. Defaults to head
                size of model

        """
        super().__init__(
            config, batch_size, cache_length, device, dtype, head_size,
        )
        shape = (batch_size, self.n_query_groups, cache_length, self.head_size)
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            shape = shape[:-1] + (self.head_size + 1,)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("token_pos", torch.zeros(cache_length, device=device, dtype=torch.int), persistent=False)
        self.next_position = None
        self.eff_batch_size = None
        self.current_length = None
        self._next_token_pos = None

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    @property
    def max_prefill_length(self) -> Optional[int]:
        return None

    def forward(self, key: torch.Tensor, value: torch.Tensor) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        if key.ndim != 4:
            raise ValueError(f"key must be a 4D tensor, but has shape {key.shape}")
        num = key.shape[2]
        if not 1 <= num <= self.max_tokens_forward:
            raise ValueError(f"key.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        shape = (self.eff_batch_size, self.n_query_groups, num, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        # Append new content to cache
        np = self.next_position
        num1 = min(num, self.cache_length - np)
        self.k[:self.eff_batch_size, :, np:(np + num1), :] = key[:, :, :num1, :]
        self.v[:self.eff_batch_size, :, np:(np + num1), :] = value[:, :, :num1, :]
        ntp = self._next_token_pos
        self.token_pos[np:(np + num1)] = torch.arange(
            ntp, ntp + num1, device=self.device, dtype=torch.int
        )
        if num1 < num:
            diff = num - num1
            self.k[:self.eff_batch_size, :, :diff, :] = key[:, :, num1:, :]
            self.v[:self.eff_batch_size, :, :diff, :] = value[:, :, num1:, :]
            self.token_pos[:diff] = torch.arange(
                ntp + num1, ntp + num, device=self.device, dtype=torch.int
            )
        self.next_position = (np + num) % self.cache_length
        self.current_length = min(self.current_length + num, self.cache_length)
        self._next_token_pos += num
        return DefaultKeysAndValues(
            self.k[:self.eff_batch_size, :, :self.current_length, :],
            self.v[:self.eff_batch_size, :, :self.current_length, :],
        )

    def update(self, *args, **kwargs):
        pass

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        init_length = key.shape[2]
        eff_init_length = min(init_length, self.cache_length)
        eff_batch_size = key.shape[0]
        if eff_batch_size > self.batch_size:
            raise ValueError(f"key.shape[0] = {eff_batch_size} must be at most batch_size = {self.batch_size}")
        shape = (eff_batch_size, self.n_query_groups, init_length, self.head_size)
        # TODO: Remove once HF bug fixed
        if self._work_around_hf_bug:
            assert value.shape == shape
            shape = shape[:-1] + (self.head_size + 1,)
            assert key.shape == shape
        elif key.shape != shape or value.shape != shape:
            raise ValueError(f"Shapes of key, value must be {shape}, but key.shape = {key.shape}, value.shape = {value.shape}")
        # Initialize cache content
        self.k = self.k.to(key.dtype)
        self.v = self.v.to(value.dtype)
        self.k[:eff_batch_size, :, :eff_init_length, :] = key[:, :, -eff_init_length:, :]
        self.v[:eff_batch_size, :, :eff_init_length, :] = value[:, :, -eff_init_length:, :]
        self.token_pos[:eff_init_length] = torch.arange(
            init_length - eff_init_length,
            init_length,
            dtype=self.token_pos.dtype,
            device=self.token_pos.device,
        )
        self.current_length = eff_init_length
        self._next_token_pos = init_length
        self.next_position = eff_init_length % self.cache_length
        self.eff_batch_size = eff_batch_size

    def token_positions(self) -> torch.Tensor:
        return self.token_pos[:self.current_length].reshape(1, 1, -1).expand(
            self.eff_batch_size, self.n_query_groups, -1
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        sz_pos = bitsize_of(self.token_pos)
        return sz_buffs + sz_pos, dict(buffers=sz_buffs, token_pos=sz_pos)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.batch_size * params.n_query_groups * cache_length * params.head_size
        k_and_v = 2 * numel * bits_for_torch_dtype(dtype)
        tk_p = cache_length * bits_for_torch_dtype(torch.int)
        return k_and_v + tk_p, dict(buffers=k_and_v, token_pos=tk_p)
