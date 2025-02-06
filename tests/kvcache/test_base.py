import random

import torch

from litgpt.kvcache.base import KVCacheParams
from litgpt.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_keys_values,
    random_attn_weights,
)


def test_most_recent():
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
    kv_cache = create_kv_cache("mostrec-default", params)
    num_insert = random.randint(cache_length, 3 * cache_length)
    max_prefill_length = kv_cache.max_prefill_length
    num_prefill = random.randint(num_insert // 3, int(num_insert * 0.75))
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length

    keys, values = random_keys_values(params, num=num_insert)
    kv_cache.prefill(keys[:, :, :num_prefill, :], values[:, :, :num_prefill, :])
    for pos in range(num_prefill, num_insert):
        kv_cache(keys[:, :, pos:(pos + 1), :], values[:, :, pos:(pos + 1), :])
        if kv_cache.update_requires_attn_weights():
            attn_weights = random_attn_weights(params, num=kv_cache.current_length)
            kv_cache.update(attn_weights=attn_weights)

    current_length = min(cache_length, num_insert)
    assert kv_cache.current_length == current_length
    token_positions = kv_cache.token_positions().to(dtype=torch.int64)
    assert token_positions.shape == (params.batch_size, params.n_query_groups, current_length)
    assert tensor_is_simple(token_positions)
    positions = token_positions[0, 0, :].tolist()
    assert len(set(positions)) == current_length
    assert all(num_insert - current_length <= x < num_insert for x in positions)
