import pytest
from omni_tutorial.batching import (
    Request,
    simulate_static_batching,
    simulate_continuous_batching,
    throughput,
)

# One long request and three short ones — the classic case where a locked
# static batch strands slots behind the straggler.
REQUESTS = [
    Request("long", prompt_len=8, output_len=10),
    Request("s1", prompt_len=8, output_len=2),
    Request("s2", prompt_len=8, output_len=2),
    Request("s3", prompt_len=8, output_len=2),
]
TOTAL_WORK = sum(r.output_len for r in REQUESTS)


def test_both_schedulers_conserve_total_work():
    static = simulate_static_batching(REQUESTS, max_batch=2)
    cont = simulate_continuous_batching(REQUESTS, max_batch=2)
    assert sum(static) == TOTAL_WORK
    assert sum(cont) == TOTAL_WORK


def test_continuous_finishes_sooner_and_uses_slots_better():
    static = simulate_static_batching(REQUESTS, max_batch=2)
    cont = simulate_continuous_batching(REQUESTS, max_batch=2)
    assert len(cont) < len(static)              # fewer iterations
    assert throughput(cont) > throughput(static)  # fuller batches


def test_occupancy_never_exceeds_max_batch():
    for sched in (simulate_static_batching, simulate_continuous_batching):
        occ = sched(REQUESTS, max_batch=2)
        assert max(occ) <= 2


def test_bad_batch_size_rejected():
    with pytest.raises(ValueError):
        simulate_continuous_batching(REQUESTS, max_batch=0)


def test_throughput_empty_is_zero():
    assert throughput([]) == 0.0
