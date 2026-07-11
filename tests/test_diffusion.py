import numpy as np
import pytest
from omni_tutorial.diffusion import (
    denoise,
    distance_to_target,
    vae_decode_shape,
    VAE_SPATIAL_FACTOR,
)


def _fixtures():
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((4, 4, 3))
    target = rng.standard_normal((4, 4, 3))
    return latent, target


def test_trajectory_length_and_shape():
    latent, target = _fixtures()
    traj = denoise(latent, target, steps=5)
    assert len(traj) == 6                      # input + 5 steps
    assert all(x.shape == latent.shape for x in traj)
    assert np.allclose(traj[0], latent)        # first entry is the untouched input


def test_denoising_converges_toward_target():
    latent, target = _fixtures()
    dist = distance_to_target(denoise(latent, target, steps=10), target)
    assert dist[-1] < dist[0]                   # ends closer than it began
    assert all(later <= earlier + 1e-9 for earlier, later in zip(dist, dist[1:]))


def test_higher_cfg_pulls_harder_in_first_step():
    latent, target = _fixtures()
    weak = denoise(latent, target, steps=1, cfg_scale=1.0)
    strong = denoise(latent, target, steps=1, cfg_scale=3.0)
    assert distance_to_target(strong, target)[1] < distance_to_target(weak, target)[1]


def test_vae_decode_expands_by_spatial_factor():
    assert vae_decode_shape((16, 16, 4)) == (16 * VAE_SPATIAL_FACTOR, 16 * VAE_SPATIAL_FACTOR, 3)


def test_shape_mismatch_and_bad_steps_rejected():
    latent, target = _fixtures()
    with pytest.raises(ValueError):
        denoise(latent, target, steps=0)
    with pytest.raises(ValueError):
        denoise(latent, np.zeros((2, 2, 3)), steps=1)
