# vLLM-Omni from scratch: a Mac-executable course

A complete beginner-to-practical notebook course for understanding **vLLM-Omni**, its architecture, runtime, APIs, performance model, and production deployment.

> [!IMPORTANT]
> Regular vLLM can run on Apple Silicon through **vLLM-Metal**. The complete vLLM-Omni runtime is not generally available as an equivalent Mac/Metal stack. Every core lesson here executes on a Mac through faithful lightweight simulations, while clearly marked Linux GPU labs map those ideas to the official runtime.

## What you will learn

- LLM inference and serving fundamentals
- multimodal input versus any-to-any generation
- prefill, decode, KV cache, PagedAttention, and continuous batching
- Thinker/Talker autoregressive pipelines
- diffusion, latents, CFG, VAE, image/video/audio generation
- stage graphs, stage processors, and `OmniConnector`
- orchestration, request lifecycle, streaming, and outputs
- disaggregation versus within-stage parallelism
- per-stage scheduling, batching, scaling, and observability
- OpenAI-style media APIs
- model integration and production deployment

## Curriculum

| # | Notebook | Outcome |
|---:|---|---|
| 00 | Course orientation | Understand model/runtime/server/application boundaries |
| 01 | From LLM to Omni | Recognize any-to-any architecture shapes |
| 02 | vLLM foundations | Understand what AR stages inherit from vLLM |
| 03 | AR stages | Trace Thinker -> Talker |
| 04 | Diffusion stages | Understand DiT, CFG, latents, and VAE |
| 05 | Stage graphs | Read the core vLLM-Omni abstraction |
| 06 | Request lifecycle | Follow a request through the orchestrator |
| 07 | Connectors | Understand disaggregation and transport |
| 08 | Scheduling/scaling | Find and scale stage bottlenecks |
| 09 | APIs/streaming | Use offline and OpenAI-style interfaces conceptually |
| 10 | Benchmarking | Measure performance without misleading yourself |
| 11 | Adding a model | Understand the complete integration surface |
| 12 | Deployment | Move from a Mac client to Linux serving |

Also see [`docs/GLOSSARY.md`](docs/GLOSSARY.md).

## Requirements

- macOS 13+
- Apple Silicon or Intel Mac
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- no GPU required for the course notebooks

## Start

```bash
git clone https://github.com/barvhaim/vllm-omni-mac-tutorial.git
cd vllm-omni-mac-tutorial
./scripts/bootstrap.sh   # runs `uv sync`
uv run jupyter lab
```

Run all tests and notebooks:

```bash
make test
```

## Learning method

Every notebook follows the same spine:

1. Read the mental model (prose + a diagram grounded in the real architecture).
2. Execute the simulation — it computes real numbers from the `omni_tutorial` primitives.
3. Read the plot (KV fragmentation, batching occupancy, queueing curves, denoising, connector overhead).
4. Do the exercise, then check the worked solution cell.
5. Read the **Linux GPU lab** cell — the verbatim official command, shown as text and never run on Mac.
6. Follow the source-lab pointer into the official repository.

Executed cells are pure-Python/numpy simulations (no `torch`, no GPU, no `vllm` import), so the
course runs on any Mac. The real `vllm serve --omni` runtime is CUDA-only and appears only in the
non-executed Linux GPU lab cells.

The notebooks are generated from [`scripts/build_notebooks.py`](scripts/build_notebooks.py) — edit
that source and re-run `uv run python scripts/build_notebooks.py` to regenerate them.

## Primary sources

- [Official documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [Official repository](https://github.com/vllm-project/vllm-omni)
- [Architecture paper](https://arxiv.org/abs/2602.02204)
- [vLLM-Metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/)

Apache-2.0 licensed.
