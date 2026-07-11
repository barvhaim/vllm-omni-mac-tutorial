"""Small educational primitives used by the notebooks.

Everything here is pure-Python / numpy — no torch, no GPU, no vllm import — so the
course runs on any Mac. The official vLLM-Omni runtime these simulate is CUDA-only.
"""
from .pipeline import Connector, Stage, StageGraph, build_voice_pipeline
from .serving import PipelineSimulator, StageSpec, sweep_replicas
from .kv_cache import PagedKVCache, kv_bytes_per_token
from .batching import (
    Request,
    simulate_static_batching,
    simulate_continuous_batching,
    throughput,
)
from .diffusion import denoise, distance_to_target, vae_decode_shape

__all__ = [
    "Connector", "Stage", "StageGraph", "build_voice_pipeline",
    "PipelineSimulator", "StageSpec", "sweep_replicas",
    "PagedKVCache", "kv_bytes_per_token",
    "Request", "simulate_static_batching", "simulate_continuous_batching", "throughput",
    "denoise", "distance_to_target", "vae_decode_shape",
]
