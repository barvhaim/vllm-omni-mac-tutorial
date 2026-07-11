import pytest
from omni_tutorial.kv_cache import PagedKVCache, kv_bytes_per_token


def test_bytes_per_token_matches_hand_calc():
    # 2 (k+v) * 4 layers * 8 heads * 64 dim * 2 bytes (fp16) = 8192
    assert kv_bytes_per_token(num_layers=4, num_kv_heads=8, head_dim=64, dtype="fp16") == 8192
    assert kv_bytes_per_token(4, 8, 64, "fp32") == 16384


def test_unknown_dtype_rejected():
    with pytest.raises(ValueError):
        kv_bytes_per_token(1, 1, 1, "int4")


def test_block_allocation_is_lazy_and_ceils():
    cache = PagedKVCache(total_blocks=10, block_size=16)
    assert cache.append("a", 1)          # one token -> one block
    assert cache.blocks_held("a") == 1
    assert cache.used_blocks == 1
    cache.append("a", 15)                # fills the block, still one block
    assert cache.blocks_held("a") == 1
    cache.append("a", 1)                 # 17th token -> second block
    assert cache.blocks_held("a") == 2


def test_fragmentation_accounting():
    cache = PagedKVCache(total_blocks=10, block_size=16)
    cache.append("a", 17)                # 2 blocks = 32 slots, 17 used
    assert cache.used_slots == 17
    assert cache.wasted_slots == 32 - 17


def test_out_of_memory_leaves_state_unchanged():
    cache = PagedKVCache(total_blocks=2, block_size=16)
    assert cache.append("a", 32)         # exactly fills the pool
    assert cache.free_blocks == 0
    assert not cache.append("b", 1)      # no room
    assert "b" not in cache._seqs
    assert cache.free_blocks == 0


def test_free_returns_blocks_to_pool():
    cache = PagedKVCache(total_blocks=4, block_size=16)
    cache.append("a", 20)                # 2 blocks
    assert cache.free_blocks == 2
    cache.free("a")
    assert cache.free_blocks == 4
    assert cache.utilization == 0.0
