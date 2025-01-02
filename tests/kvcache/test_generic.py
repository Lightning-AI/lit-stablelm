import random

import torch
import pytest

from litgpt.kvcache.base import KVCacheParams
from litgpt.kvcache.test_utils import (
    KV_CACHE_NAMES,
    create_kv_cache,
    tensor_is_simple,
    random_attn_weights,
    random_keys_values,
)


@pytest.mark.parametrize("name", KV_CACHE_NAMES)
def test_store_retrieve(name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    params = KVCacheParams(
        batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=8,
        n_head=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    if name.startswith("dense"):
        num_insert = random.randint(cache_length // 2, cache_length)
    else:
        num_insert = random.randint(cache_length, 3 * cache_length)
    max_prefill_length = kv_cache.max_prefill_length
    num_prefill = random.randint(num_insert // 3, int(num_insert * 0.75))
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length

    keys, values = random_keys_values(params, num=num_insert)
    kv_cache.prefill(keys[:, :, :num_prefill, :], values[:, :, :num_prefill, :])
    keys_and_values = None
    for pos in range(num_prefill, num_insert):
        keys_and_values = kv_cache(
            keys[:, :, pos:(pos + 1), :], values[:, :, pos:(pos + 1), :]
        )
        if kv_cache.update_requires_attn_weights():
            attn_weights = random_attn_weights(params, num=kv_cache.current_length)
            kv_cache.update(attn_weights=attn_weights)

    current_length = min(cache_length, num_insert)
    assert kv_cache.current_length == current_length
    token_positions = kv_cache.token_positions().to(dtype=torch.int64)
    assert token_positions.shape == (params.batch_size, params.n_query_groups, current_length)
    assert tensor_is_simple(token_positions)
    # Positions for every (b, h) must be different
    for b, h in zip(range(params.batch_size), range(params.n_query_groups)):
        token_pos = token_positions[b, h, :].tolist()
        assert all(0 <= x < num_insert for x in token_pos)
        err_msg = f"num_insert = {num_insert}, b = {b}, h = {h}, current_length = {current_length}, num_prefill = {num_prefill}"
        assert len(set(token_pos)) == current_length, err_msg
    # Test cache content slice by slice
    for pos in range(current_length):
        index = token_positions[:, :, pos][:, :, None, None].expand(-1, -1, 1, params.head_size)
        # `index[i, j, 0, k] = next_position[i, j]`
        k_expected = keys.gather(-2, index).squeeze(-2)
        v_expected = values.gather(-2, index).squeeze(-2)
        torch.testing.assert_close(
            k_expected, keys_and_values.keys()[:, :, pos, :]
        )
        torch.testing.assert_close(
            v_expected, keys_and_values.values()[:, :, pos, :]
        )


@pytest.mark.parametrize("name", KV_CACHE_NAMES)
def test_prefill(name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_compares = 3

    params = KVCacheParams(
        batch_size=2,
        n_query_groups=2,
        cache_length=32,
        head_size=64,
        n_head=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)

    keys, values = random_keys_values(params, num=cache_length)
    keys_cached = []
    values_cached = []
    max_prefill_length = kv_cache.max_prefill_length
    for _ in range(num_compares):
        num_prefill = random.randint(cache_length // 8, cache_length)
        if max_prefill_length is not None and num_prefill > max_prefill_length:
            num_prefill = max_prefill_length
        kv_cache.prefill(keys[:, :, :num_prefill, :], values[:, :, :num_prefill, :])
        keys_and_values = None
        for pos in range(num_prefill, cache_length):
            keys_and_values = kv_cache(
                keys[:, :, pos:(pos + 1), :], values[:, :, pos:(pos + 1), :]
            )
            if kv_cache.update_requires_attn_weights():
                attn_weights = random_attn_weights(params, num=kv_cache.current_length)
                kv_cache.update(attn_weights=attn_weights)
        if keys_and_values is not None:
            keys_cached.append(keys_and_values.keys().clone())
            values_cached.append(keys_and_values.values().clone())
        else:
            keys_cached.append(None)
            values_cached.append(None)

    num_none = 0
    for k, v in zip(keys_cached[1:], values_cached[1:]):
        if k is not None:
            torch.testing.assert_close(k, keys_cached[0])
            torch.testing.assert_close(v, values_cached[0])
        else:
            num_none += 1
    assert num_none < num_compares - 1
