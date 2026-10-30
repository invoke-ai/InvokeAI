"""How ``Flux3DenoiseInvocation._run_diffusion`` wires schedule, start latents, inpainting and CFG together.

The backend loop and the schedule helpers have tests of their own; what those cannot see is the node that calls
them. Each of these one-line breakages passed the suite before (the same ones FLUX.2 had):

* normalizing the noise, or the noise-and-init mixture, instead of only the init latents (#8964);
* handing the inpaint extension normalized noise, which rescales the noise outside the mask;
* dropping the ``1 - mask`` inversion, which regenerates the region that should be kept;
* skipping the respacing of a clipped schedule, or taking ``t_0`` from the unclipped one;
* returning the early (nothing-to-step) result without denormalizing it.

The tests run the real ``_run_diffusion`` with the backend ``denoise`` replaced by a recorder.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import invokeai.app.invocations.flux3.flux3_denoise as flux3_denoise
from invokeai.app.invocations.flux3.flux3_denoise import Flux3DenoiseInvocation
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux2.latent_norm import bn_denormalize, bn_normalize
from invokeai.backend.flux2.sampling_utils import pack_flux2, unpack_flux2
from invokeai.backend.flux3.sampling import get_schedule_flux3, redensify_schedule_flux3
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Flux3ConditioningInfo,
)

# Far enough from (0, 1) that a stray or missing normalization is obvious.
BN_MEAN = torch.full((128,), 0.05)
BN_STD = torch.full((128,), 1.7676)
SIZE = 128
LATENT = SIZE // 8
STEPS = 4


def _tensor(seed: int, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _drive(
    *,
    variant: Flux3VariantType = Flux3VariantType.Klein,
    denoising_start: float = 0.0,
    denoising_end: float = 1.0,
    with_init: bool = False,
    with_mask: bool = False,
    cfg_scale: float = 1.0,
    captured: dict | None = None,
):
    noise = _tensor(7, 1, 32, LATENT, LATENT)
    tensors = {"init": _tensor(8, 1, 32, LATENT, LATENT), "mask": torch.rand(1, 1, LATENT, LATENT)}
    captured = {} if captured is None else captured
    captured.update({"denoise": [], "inpaint": [], "noise": noise, **tensors})

    def model_info(model):
        info = MagicMock()
        info.model = model
        info.model_on_device.return_value.__enter__.return_value = (None, model)
        return info

    transformer = SimpleNamespace(config=SimpleNamespace(flow_map=variant is Flux3VariantType.Klein))
    vae_info, transformer_info = model_info(MagicMock()), model_info(transformer)

    context = MagicMock()
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux3, type=ModelType.Main, name="flux3", variant=variant
    )
    context.models.load.side_effect = lambda key: vae_info if key == "vae" else transformer_info
    context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
        conditionings=[Flux3ConditioningInfo(prompt_embeds=torch.full((1, 80, 8), 1.0 if name == "pos" else -1.0))]
    )
    context.tensors.load.side_effect = lambda name: tensors[name].clone()
    captured["context"] = context

    invocation = Flux3DenoiseInvocation.model_construct(
        latents=SimpleNamespace(latents_name="init") if with_init else None,
        denoise_mask=SimpleNamespace(mask_name="mask") if with_mask else None,
        denoising_start=denoising_start,
        denoising_end=denoising_end,
        add_noise=True,
        transformer=SimpleNamespace(transformer="transformer", loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=SimpleNamespace(conditioning_name="neg"),
        cfg_scale=cfg_scale,
        width=SIZE,
        height=SIZE,
        steps=STEPS,
        seed=0,
        vae=SimpleNamespace(vae="vae"),
        kontext_conditioning=None,
    )

    def recording_denoise(**kwargs):
        captured["denoise"].append(kwargs)
        return kwargs["img"]

    class _RecordingInpaintExtension(RectifiedFlowInpaintExtension):
        def __init__(self, *, init_latents, inpaint_mask, noise):
            captured["inpaint"].append({"init_latents": init_latents, "inpaint_mask": inpaint_mask, "noise": noise})
            super().__init__(init_latents, inpaint_mask, noise)

    with (
        patch.object(flux3_denoise.TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
        patch.object(flux3_denoise, "get_flux2_bn_stats", return_value=(BN_MEAN, BN_STD)),
        patch.object(flux3_denoise, "get_noise_flux3", return_value=noise),
        patch.object(flux3_denoise, "peak_dequant_transient_bytes", return_value=0),
        patch.object(flux3_denoise, "denoise", recording_denoise),
        patch.object(flux3_denoise, "RectifiedFlowInpaintExtension", _RecordingInpaintExtension),
    ):
        captured["result"] = invocation._run_diffusion(context)
    captured["transformer_info"] = transformer_info
    return captured


def _expected_schedule(denoising_start: float, denoising_end: float = 1.0) -> list[float]:
    clipped = clip_timestep_schedule_fractional(get_schedule_flux3(STEPS), denoising_start, denoising_end)
    return redensify_schedule_flux3(clipped, STEPS)


def _normalized_init(captured) -> torch.Tensor:
    return bn_normalize(pack_flux2(captured["init"]), BN_MEAN, BN_STD)


def test_txt2img_starts_from_unit_noise_and_returns_denormalized_latents() -> None:
    captured = _drive()

    (call,) = captured["denoise"]
    assert torch.equal(call["img"], pack_flux2(captured["noise"]))
    assert call["timesteps"] == pytest.approx(get_schedule_flux3(STEPS))
    assert call["flow_map"] is True and call["neg_txt"] is None and call["inpaint_extension"] is None
    # The recorder hands the start back unchanged, so the output is the noise taken out of the normalized space.
    expected = unpack_flux2(bn_denormalize(pack_flux2(captured["noise"]), BN_MEAN, BN_STD), SIZE, SIZE)
    assert torch.allclose(captured["result"], expected)


def test_img2img_blends_unit_noise_with_normalized_init_on_the_respaced_schedule() -> None:
    captured = _drive(with_init=True, denoising_start=0.5)

    (call,) = captured["denoise"]
    expected_schedule = _expected_schedule(0.5)
    assert len(expected_schedule) == STEPS + 1, "the clipped window is respaced to the requested steps"
    assert call["timesteps"] == pytest.approx(expected_schedule)

    t_0 = expected_schedule[0]
    assert t_0 != pytest.approx(1.0), "precondition: the clip moved t_0"
    noise_packed = pack_flux2(captured["noise"])
    expected = t_0 * noise_packed + (1.0 - t_0) * _normalized_init(captured)
    assert torch.allclose(call["img"], expected, atol=1e-6)
    # Normalizing the mixture instead divides the noise term by bn_std as well.
    mixture = t_0 * noise_packed + (1.0 - t_0) * pack_flux2(captured["init"])
    assert not torch.allclose(call["img"], bn_normalize(mixture, BN_MEAN, BN_STD), atol=1e-2)


def test_the_head_of_a_split_chain_keeps_the_steps_of_its_window() -> None:
    """Respacing a window that starts at 0 would run the head of a two-node chain at the full step count."""
    captured = _drive(denoising_end=0.1)

    (call,) = captured["denoise"]
    clipped = clip_timestep_schedule_fractional(get_schedule_flux3(STEPS), 0.0, 0.1)
    assert len(clipped) - 1 < STEPS, "precondition: the window holds fewer steps than requested"
    assert call["timesteps"] == pytest.approx(clipped)


def test_the_inpaint_extension_gets_unit_noise_normalized_init_and_the_inverted_mask() -> None:
    captured = _drive(with_init=True, with_mask=True)

    (recorded,) = captured["inpaint"]
    assert torch.equal(recorded["noise"], pack_flux2(captured["noise"]))
    assert torch.allclose(recorded["init_latents"], _normalized_init(captured))
    # The denoise mask marks what to keep; the extension wants 1 where it regenerates.
    expected_mask = pack_flux2((1.0 - captured["mask"]).expand(1, 32, LATENT, LATENT))
    assert torch.allclose(recorded["inpaint_mask"], expected_mask, atol=1e-6)
    assert captured["denoise"][0]["inpaint_extension"] is not None


def test_a_degenerate_window_returns_the_denormalized_blend_without_loading_the_transformer() -> None:
    captured = _drive(with_init=True, denoising_start=0.5, denoising_end=0.5)

    assert captured["denoise"] == []
    assert not captured["transformer_info"].model_on_device.called
    (t_0,) = _expected_schedule(0.5, 0.5)
    blend = t_0 * pack_flux2(captured["noise"]) + (1.0 - t_0) * _normalized_init(captured)
    expected = unpack_flux2(bn_denormalize(blend, BN_MEAN, BN_STD), SIZE, SIZE)
    assert torch.allclose(captured["result"], expected, atol=1e-6)


def test_klein_base_runs_cfg_with_the_negative_conditioning() -> None:
    captured = _drive(variant=Flux3VariantType.KleinBase, cfg_scale=3.5)

    (call,) = captured["denoise"]
    assert call["flow_map"] is False and call["cfg_scale"] == 3.5
    assert torch.all(call["neg_txt"] == -1.0) and torch.all(call["txt"] == 1.0)


def test_klein_with_cfg_is_refused_before_any_model_loads() -> None:
    """A workflow with CFG on Klein used to load the ~10 GB transformer before it failed."""
    captured: dict = {}
    with pytest.raises(ValueError, match="samples without CFG"):
        _drive(variant=Flux3VariantType.Klein, cfg_scale=2.0, captured=captured)

    assert not captured["context"].models.load.called
