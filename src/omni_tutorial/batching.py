"""Static vs. continuous batching, as a deterministic step simulation.

Autoregressive decoding runs in *iterations*: every step advances each live
sequence by one token. The question a serving runtime answers is *which*
sequences share a step.

- **Static batching** locks a batch until its slowest member finishes. Short
  requests sit idle in the batch, wasting slots, until the longest one is done.
- **Continuous batching** (what vLLM does) fills a freed slot with a waiting
  request at the very next iteration, so the batch stays as full as demand allows.

Both simulators below return the batch occupancy at each iteration, which is the
throughput signal notebooks plot. No tensors, no timing — just the scheduling.
"""
from dataclasses import dataclass


@dataclass
class Request:
    id: str
    prompt_len: int      # tokens already present (prefill); not decoded here
    output_len: int      # decode iterations this request needs


def simulate_static_batching(requests: list[Request], max_batch: int) -> list[int]:
    """Occupancy per iteration when batches are locked until fully drained.

    Requests are admitted in arrival order, `max_batch` at a time; the next
    group waits until every member of the current group has finished.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    occupancy: list[int] = []
    for start in range(0, len(requests), max_batch):
        group = requests[start:start + max_batch]
        longest = max(r.output_len for r in group)
        for step in range(longest):
            # a request occupies a slot until its own output_len is reached
            live = sum(1 for r in group if step < r.output_len)
            occupancy.append(live)
    return occupancy


def simulate_continuous_batching(requests: list[Request], max_batch: int) -> list[int]:
    """Occupancy per iteration when freed slots are refilled immediately.

    A finished request's slot is handed to the next waiting request on the same
    iteration, so the batch stays full while work remains.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    waiting = list(requests)
    running: list[int] = []          # remaining decode steps per running slot
    occupancy: list[int] = []
    while waiting or running:
        while len(running) < max_batch and waiting:
            running.append(waiting.pop(0).output_len)
        occupancy.append(len(running))
        running = [r - 1 for r in running]
        running = [r for r in running if r > 0]
    return occupancy


def throughput(occupancy: list[int]) -> float:
    """Mean sequences advanced per iteration — higher is better utilization."""
    return sum(occupancy) / len(occupancy) if occupancy else 0.0
