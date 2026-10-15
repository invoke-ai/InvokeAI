"""Reference-latent noising schedules for Krea-2 style reference.

The styled attention needs the reference image at the *same* noise level as the target at every step,
so the reference latent has to be walked along a trajectory that matches the sampler's sigma schedule.

``linear`` is the rectified-flow forward process, ``z(sigma) = (1 - sigma) * ref + sigma * eps``. Note
that upstream draws ``eps`` **once** and reuses it for every sigma -- the reference travels a single
straight line rather than being re-noised independently at each step. Re-sampling per sigma would make
the reference features jitter between steps and defeat the point.

Upstream's default is instead ``flowturbo_pc``, a predictor-corrector that integrates the model's own
velocity field and blends the result back toward the linear prior by ``gamma``. That costs roughly two
extra transformer forwards per schedule point up front, and with ``gamma=0.5`` it stays half-anchored to
the linear prior anyway, so it is deferred until the cheap path has been measured.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Upstream fixes the reference noise seed so a given reference image always produces the same trajectory,
# independent of the generation seed. Keeping that makes style reference reproducible on its own terms.
KREA2_STYLE_REFERENCE_NOISE_SEED = 42


def build_linear_reference_latents(
    reference_latents: torch.Tensor,
    sigmas: Sequence[float],
    seed: int = KREA2_STYLE_REFERENCE_NOISE_SEED,
) -> list[torch.Tensor]:
    """Noise ``reference_latents`` to each sigma along one straight rectified-flow trajectory.

    Returns one latent per entry in ``sigmas``, in the same order. ``reference_latents`` may be packed or
    unpacked; the noise simply matches its shape.

    The noise is drawn on the CPU, matching ``Krea2DenoiseInvocation._get_noise``, so the trajectory does
    not change between CPU and CUDA runs.
    """
    if len(sigmas) == 0:
        raise ValueError("Krea-2 style reference: the sigma schedule is empty.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        reference_latents.shape,
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    ).to(device=reference_latents.device, dtype=reference_latents.dtype)

    latents: list[torch.Tensor] = []
    for sigma in sigmas:
        value = max(0.0, min(1.0, float(sigma)))
        latents.append((1.0 - value) * reference_latents + value * noise)
    return latents
