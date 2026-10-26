import torch

from invokeai.backend.flux.controlnet.instantx_controlnet_flux import (
    InstantXControlNetFlux,
    InstantXControlNetFluxOutput,
)
from invokeai.backend.flux.model import FluxParams

TINY_PARAMS = FluxParams(
    in_channels=4,
    vec_in_dim=4,
    context_in_dim=4,
    hidden_size=16,
    mlp_ratio=2.0,
    num_heads=2,
    depth=1,
    depth_single_blocks=1,
    axes_dim=[2, 2, 4],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=False,
)


def _tiny_controlnet(num_control_modes: int | None) -> InstantXControlNetFlux:
    torch.manual_seed(0)
    model = InstantXControlNetFlux(TINY_PARAMS, num_control_modes=num_control_modes)
    # The ControlNet output projections start at zero, which would make every output identical. Randomize all weights
    # so that a change in conditioning shows up in the outputs.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.5)
    return model.eval()


def _forward(model: InstantXControlNetFlux, controlnet_mode: torch.Tensor | None) -> InstantXControlNetFluxOutput:
    generator = torch.Generator().manual_seed(1)
    img_tokens, txt_tokens = 6, 3
    img = torch.randn(1, img_tokens, TINY_PARAMS.in_channels, generator=generator)
    with torch.no_grad():
        return model(
            controlnet_cond=torch.randn(1, img_tokens, TINY_PARAMS.in_channels, generator=generator),
            controlnet_mode=controlnet_mode,
            img=img,
            img_ids=torch.randn(1, img_tokens, 3, generator=generator),
            txt=torch.randn(1, txt_tokens, TINY_PARAMS.context_in_dim, generator=generator),
            txt_ids=torch.zeros(1, txt_tokens, 3),
            timesteps=torch.tensor([0.5]),
            y=torch.randn(1, TINY_PARAMS.vec_in_dim, generator=generator),
        )


def _samples(output: InstantXControlNetFluxOutput) -> list[torch.Tensor]:
    assert output.controlnet_block_samples is not None
    assert output.controlnet_single_block_samples is not None
    return output.controlnet_block_samples + output.controlnet_single_block_samples


def test_non_union_controlnet_ignores_a_supplied_control_mode():
    model = _tiny_controlnet(num_control_modes=None)

    without_mode = _samples(_forward(model, controlnet_mode=None))
    with_mode = _samples(_forward(model, controlnet_mode=torch.tensor([[2]])))

    assert len(with_mode) == len(without_mode) == 2
    for actual, expected in zip(with_mode, without_mode, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_union_controlnet_conditions_on_the_control_mode():
    model = _tiny_controlnet(num_control_modes=5)

    without_mode = _samples(_forward(model, controlnet_mode=None))
    mode_0 = _samples(_forward(model, controlnet_mode=torch.tensor([[0]])))
    mode_2 = _samples(_forward(model, controlnet_mode=torch.tensor([[2]])))

    for with_mode in (mode_0, mode_2):
        assert any(not torch.allclose(a, b) for a, b in zip(with_mode, without_mode, strict=True))
    assert any(not torch.allclose(a, b) for a, b in zip(mode_0, mode_2, strict=True))
