# vLLM-Omni on a Mac: an executable tutorial

A beginner-friendly, notebook-based course for learning the architecture behind **vLLM-Omni** on Apple Silicon or Intel Macs.

> [!IMPORTANT]
> The official vLLM-Omni runtime targets Linux accelerator environments and does not natively run on macOS. This repository teaches its core ideas with small, executable simulations on macOS: autoregressive decoding, KV caches, diffusion, stage graphs, disaggregation, connectors, batching, and serving. The final notebook generates version-matched Linux deployment commands for the real runtime, but does not pretend that CUDA-only code runs on a Mac.

## What you will build

1. A mental model of multimodal and any-to-any inference.
2. A toy autoregressive decoder with a visible KV-cache analogue.
3. A 2D diffusion/denoising process.
4. A configurable Thinker -> Talker -> Vocoder stage graph.
5. A discrete-event serving simulator showing bottlenecks and independent stage scaling.
6. A deployment planner for moving from a Mac prototype to real Linux GPUs.

## Mac requirements

- macOS 13+
- Python 3.11 or 3.12
- Apple Silicon or Intel Mac
- No GPU is required

## Quickstart

```bash
git clone https://github.com/njs2017/vllm-omni-mac-tutorial.git
cd vllm-omni-mac-tutorial
./scripts/bootstrap.sh
source .venv/bin/activate
jupyter lab
```

Open notebooks in numeric order. To execute the complete course headlessly:

```bash
make test
```

## Course map

| Notebook | Topic | Artifact |
|---|---|---|
| `00` | Why vLLM-Omni exists | Architecture vocabulary |
| `01` | AR decoding and cache reuse | Tiny token generator |
| `02` | Diffusion versus AR | Animated denoising trajectory |
| `03` | Stage graphs and connectors | Thinker/Talker/Vocoder pipeline |
| `04` | Serving and scaling | Queueing and utilization experiment |
| `05` | Mac-to-Linux path | Version-matched deployment plan |

## What this repository is not

- It is not a macOS port of vLLM or vLLM-Omni.
- Toy models do not reproduce production model quality or kernels.
- Simulated timings are explanatory, not official benchmark results.

## References

- [vLLM-Omni documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [Official repository](https://github.com/vllm-project/vllm-omni)
- [Architecture paper](https://arxiv.org/abs/2602.02204)
- [Architecture overview](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/)

Apache-2.0 licensed.
