"""A toy diffusion stage: latents, classifier-free guidance, and VAE decode.

Diffusion image/video stages (DiT) do not decode one token at a time like an AR
stage. They start from pure noise in a compressed *latent* space and iteratively
*denoise* toward a sample, then a VAE decoder expands the final latent into pixels.

This module models the numerics of that loop on small numpy arrays so the
behaviour is real (guidance strength changes the trajectory; more steps converge
closer to the target) without needing a trained network or a GPU. The "model"
here is a fixed linear pull toward a target latent — enough to show the shapes,
the guidance formula, and the convergence curve.
"""
from __future__ import annotations

import numpy as np

# Compression factor of a typical latent VAE: 8x per spatial axis, so a
# H x W x 3 image comes from an (H/8) x (W/8) x C latent.
VAE_SPATIAL_FACTOR = 8


def _step_toward(latent: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    """One denoising step: predict the clean direction and move along it.

    Stands in for a single DiT forward pass. `strength` in (0, 1] is how far the
    step moves toward the predicted clean latent.
    """
    return latent + strength * (target - latent)


def denoise(
    latent: np.ndarray,
    target: np.ndarray,
    steps: int,
    cfg_scale: float = 1.0,
    step_strength: float = 0.3,
) -> list[np.ndarray]:
    """Run the denoising loop, returning the latent after every step.

    Classifier-free guidance combines a conditional and an unconditional
    prediction: ``guided = uncond + cfg_scale * (cond - uncond)``. Here the
    conditional prediction pulls toward `target`, the unconditional toward the
    origin (zeros); `cfg_scale > 1` extrapolates past the conditional prediction,
    sharpening adherence to the prompt.

    The returned list starts with the input `latent` and has ``steps + 1`` entries.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if latent.shape != target.shape:
        raise ValueError("latent and target must share a shape")
    uncond_target = np.zeros_like(target)
    trajectory = [latent.copy()]
    x = latent.copy()
    for _ in range(steps):
        cond = _step_toward(x, target, step_strength)
        uncond = _step_toward(x, uncond_target, step_strength)
        x = uncond + cfg_scale * (cond - uncond)
        trajectory.append(x.copy())
    return trajectory


def distance_to_target(trajectory: list[np.ndarray], target: np.ndarray) -> list[float]:
    """L2 distance from the target at each step — the convergence curve."""
    return [float(np.linalg.norm(x - target)) for x in trajectory]


def vae_decode_shape(latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Map a latent (h, w, c) to the decoded pixel shape (H, W, 3).

    Models only the shape transform a VAE decoder performs, not the convolution.
    """
    h, w, _ = latent_shape
    return (h * VAE_SPATIAL_FACTOR, w * VAE_SPATIAL_FACTOR, 3)
