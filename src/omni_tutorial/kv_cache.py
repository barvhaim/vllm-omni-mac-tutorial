"""Block-paged KV cache simulation.

A CPU-only, torch-free model of the accounting vLLM does for PagedAttention: the
KV cache is carved into fixed-size *blocks*, each holding `block_size` token slots.
A sequence grows one token at a time and allocates a new block only when its last
block fills up. This is what lets many sequences share memory without
pre-reserving each one's maximum length, at the cost of up to `block_size - 1`
wasted slots in every sequence's final block (internal fragmentation).

Nothing here computes attention; it models *memory*, which is what governs how
many requests fit on an accelerator and therefore throughput.
"""
from dataclasses import dataclass, field

_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype: str = "fp16") -> int:
    """Bytes of KV cache one token occupies across all layers.

    Two tensors (key and value) per layer, each `num_kv_heads * head_dim` elements.
    """
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; pick one of {sorted(_DTYPE_BYTES)}")
    return 2 * num_layers * num_kv_heads * head_dim * _DTYPE_BYTES[dtype]


@dataclass
class PagedKVCache:
    """Fixed pool of blocks shared by all live sequences.

    Parameters
    ----------
    total_blocks: size of the pool.
    block_size: token slots per block.
    """
    total_blocks: int
    block_size: int = 16
    _seqs: dict[str, int] = field(default_factory=dict)   # seq_id -> tokens held
    free_blocks: int = field(init=False)

    def __post_init__(self):
        if self.total_blocks < 1 or self.block_size < 1:
            raise ValueError("total_blocks and block_size must be >= 1")
        self.free_blocks = self.total_blocks

    @staticmethod
    def _blocks_for(tokens: int, block_size: int) -> int:
        return (tokens + block_size - 1) // block_size  # ceil

    def blocks_held(self, seq_id: str) -> int:
        return self._blocks_for(self._seqs.get(seq_id, 0), self.block_size)

    def append(self, seq_id: str, tokens: int = 1) -> bool:
        """Add `tokens` to a sequence, allocating blocks as needed.

        Returns False and changes nothing if there is not enough free space
        (an out-of-memory event the scheduler would have to handle by preempting).
        """
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        held = self._seqs.get(seq_id, 0)
        before = self._blocks_for(held, self.block_size)
        after = self._blocks_for(held + tokens, self.block_size)
        need = after - before
        if need > self.free_blocks:
            return False
        self.free_blocks -= need
        self._seqs[seq_id] = held + tokens
        return True

    def free(self, seq_id: str) -> None:
        """Release every block held by a finished (or preempted) sequence."""
        held = self._seqs.pop(seq_id, 0)
        self.free_blocks += self._blocks_for(held, self.block_size)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def used_slots(self) -> int:
        return sum(self._seqs.values())

    @property
    def wasted_slots(self) -> int:
        """Allocated-but-unused token slots — internal fragmentation."""
        return self.used_blocks * self.block_size - self.used_slots

    @property
    def utilization(self) -> float:
        """Fraction of the pool's slots actually holding tokens."""
        capacity = self.total_blocks * self.block_size
        return self.used_slots / capacity if capacity else 0.0
