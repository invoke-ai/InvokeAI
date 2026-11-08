"""Ideogram 4's denoise previews go through the shared step callback.

That is what gives them the throttle, the cancel check and the FLUX.2 Tiny AutoEncoder: Ideogram 4 denoises in
the FLUX.2 VAE's latent space, so the shared path needs the unpacked 32-channel latent, filed under Ideogram 4 --
still in the normalized denoise space, which is what TAEF2 and the FLUX.2 linear factors read.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import Ideogram4ConditioningField
from invokeai.app.invocations.ideogram4 import ideogram4_denoise
from invokeai.app.invocations.ideogram4.ideogram4_denoise import Ideogram4DenoiseInvocation
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.sampling_utils import LATENT_DIM
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Ideogram4ConditioningInfo,
)

TINY = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=16,
    in_channels=LATENT_DIM,
    llm_features_dim=32,
    mrope_section=(4, 2, 2),
)


def test_each_step_previews_the_flux2_space_latent_through_the_shared_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ideogram4_denoise.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    # float64: CPU bf16 matmul can raise an illegal instruction on Windows runners.
    branches = (Ideogram4Transformer(TINY).double().eval(), Ideogram4Transformer(TINY).double().eval())
    monkeypatch.setattr(Ideogram4DenoiseInvocation, "_load_branches", lambda self, context, stack, mem, **_: branches)

    # Keep a copy of each packed latent the loop hands over, before the node's callback replaces it.
    packed: list[torch.Tensor] = []
    run = ideogram4_denoise.run_ideogram4_denoise

    def recording_run(*, step_callback, **kwargs):
        def record(state):
            packed.append(state.latents.clone())
            step_callback(state)

        return run(step_callback=record, **kwargs)

    monkeypatch.setattr(ideogram4_denoise, "run_ideogram4_denoise", recording_run)

    seen: list[tuple[BaseModelType, int, int, torch.Tensor]] = []
    conditioning = ConditioningFieldData(
        conditionings=[Ideogram4ConditioningInfo(prompt_embeds=torch.zeros(4, TINY.llm_features_dim))]
    )
    context = SimpleNamespace(
        conditioning=SimpleNamespace(load=lambda name: conditioning),
        tensors=SimpleNamespace(save=lambda tensor: "latents"),
        util=SimpleNamespace(
            sd_step_callback=lambda state, base: seen.append((base, state.step, state.total_steps, state.latents))
        ),
    )
    invocation = Ideogram4DenoiseInvocation.model_construct(
        positive_conditioning=Ideogram4ConditioningField(conditioning_name="c"), width=64, height=32, steps=2
    )

    invocation.invoke(context)  # type: ignore[arg-type]

    # The percentage is step / total, as it was before.
    assert [(base, step, total) for base, step, total, _ in seen] == [
        (BaseModelType.Ideogram4, 1, 2),
        (BaseModelType.Ideogram4, 2, 2),
    ]
    for (*_, latent), grid in zip(seen, packed, strict=True):
        # Packed channel (py * 2 + px) * 32 + c holds VAE channel c at pixel (2 * gy + py, 2 * gx + px). A 64x32 image
        # is a 4x2 grid and an 8x4 latent. Values unchanged: no denormalize.
        expected = grid.float().view(1, 2, 2, 32, 2, 4).permute(0, 3, 4, 1, 5, 2).reshape(1, 32, 4, 8)
        assert torch.equal(latent, expected)
