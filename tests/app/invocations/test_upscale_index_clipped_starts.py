"""The Upscale widget's Creativity translation holds against the real Z-Image and Anima nodes.

`flux_denoise` clips `denoising_start` by sigma value; `z_image_denoise` and `anima_denoise` clip it by
step index on a shifted schedule, so the same number enters them far noisier. The widget therefore
does not send them FLUX's start: `getIndexClippedDenoisingStart` (`features/upscale/core/graph.ts`)
picks the step whose sigma is nearest the one FLUX would enter at, from its own copy of each node's
shift (`settings.ts`) and clipping rule.

`upscaleIndexClippedStarts.json`, written by `indexClippedStarts.test.ts` (regenerate with `vitest -u`),
records what the widget sends. Each case here drives the real `_run_diffusion` -- the real shift, the
real `_get_sigmas`, the real clipping and the real img2img blend -- and reads the entry sigma back out
of the blended tensor. It has to be the step nearest FLUX's entry. A change to either node's shift,
schedule or clipping fails here instead of silently moving every Upscale preset; that includes a move
to value clipping, which would leave the translation entering far too late.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation

FIXTURE = (
    Path(__file__).parents[3]
    / "invokeai"
    / "frontend"
    / "webv2"
    / "src"
    / "features"
    / "upscale"
    / "core"
    / "__snapshots__"
    / "upscaleIndexClippedStarts.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


class _StopAfterBlend(Exception):
    """Ends the run at the first call after the img2img blend."""


def _entry(node, module: str, dtype_attr: str, latents: torch.Tensor, noise: torch.Tensor, patches) -> tuple:
    """Run `node._run_diffusion` to the blend; return the full schedule it built and the entry sigma.

    The input latents are zeros and the noise is ones, so `s * noise + (1 - s) * init` is `s` in every
    element: the entry sigma falls out of the tensor instead of being derived a second time.
    """
    schedule: dict[str, list[float]] = {}
    real_get_sigmas = node._get_sigmas

    def record(*args):
        schedule["sigmas"] = real_get_sigmas(*args)
        return schedule["sigmas"]

    captured: dict[str, torch.Tensor] = {}

    def capture(_context, blended):
        captured["latents"] = blended
        raise _StopAfterBlend

    context = MagicMock()
    context.tensors.load.return_value = latents
    with ExitStack() as stack:
        stack.enter_context(patch(f"{module}.TorchDevice.choose_torch_device", return_value=torch.device("cpu")))
        # float32: the entry sigma is read back out of the tensor, and bfloat16 keeps three digits.
        stack.enter_context(patch(f"{module}.TorchDevice.{dtype_attr}", return_value=torch.float32))
        stack.enter_context(patch.object(node, "_prepare_noise_tensor", return_value=noise))
        stack.enter_context(patch.object(node, "_get_sigmas", record))
        stack.enter_context(patch.object(node, "_prep_inpaint_mask", capture))
        for extra in patches:
            stack.enter_context(extra)
        with pytest.raises(_StopAfterBlend):
            node._run_diffusion(context)

    blended = captured["latents"]
    assert torch.allclose(blended, blended.flatten()[:1].expand_as(blended)), "the blend should be uniform"
    return schedule["sigmas"], float(blended.flatten()[0])


def z_image_entry(side: int, steps: int, denoising_start: float) -> tuple[list[float], float]:
    node = ZImageDenoiseInvocation.model_construct(
        id="upscale",
        latents=MagicMock(latents_name="latents"),
        add_noise=True,
        width=side,
        height=side,
        steps=steps,
        denoising_start=denoising_start,
        denoising_end=1.0,
        shift=None,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer"),
        seed=0,
        scheduler="euler",
    )
    regional = SimpleNamespace(regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16)))
    return _entry(
        node,
        "invokeai.app.invocations.z_image.z_image_denoise",
        "choose_bfloat16_safe_dtype",
        torch.zeros(1, 16, 8, 8),
        torch.ones(1, 16, 8, 8),
        [
            patch(
                "invokeai.app.invocations.z_image.z_image_denoise.ZImageRegionalPromptingExtension.from_text_conditionings",
                return_value=regional,
            ),
            patch.object(
                node,
                "_load_text_conditioning",
                return_value=[SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16), mask=None)],
            ),
        ],
    )


def anima_entry(side: int, steps: int, denoising_start: float) -> tuple[list[float], float]:
    node = AnimaDenoiseInvocation.model_construct(
        id="upscale",
        latents=MagicMock(latents_name="latents"),
        noise=None,
        denoise_mask=None,
        add_noise=True,
        width=side,
        height=side,
        steps=steps,
        denoising_start=denoising_start,
        denoising_end=1.0,
        guidance_scale=1.0,
        negative_conditioning=None,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer"),
        control_lllite=None,
        seed=0,
        scheduler="euler",
    )
    return _entry(
        node,
        "invokeai.app.invocations.anima.anima_denoise",
        "choose_anima_inference_dtype",
        torch.zeros(1, 16, 8, 8),
        torch.ones(1, 16, 1, 8, 8),
        [patch.object(node, "_load_text_conditionings", return_value=[SimpleNamespace(mask=None)])],
    )


ENTRY = {"z-image": z_image_entry, "anima": anima_entry}


def test_the_fixture_covers_both_index_clipping_nodes():
    """An empty or one-sided fixture would make every case below vacuous."""
    assert {case["base"] for case in CASES} == set(ENTRY)
    assert len(CASES) >= 60


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: f"{case['base']}-{case['side']}px-{case['steps']}st-creativity{case['creativity']}",
)
def test_the_pass_enters_at_the_step_nearest_where_flux_enters(case: dict) -> None:
    sigmas, entry = ENTRY[case["base"]](case["side"], case["steps"], case["denoisingStart"])
    target = case["fluxEntrySigma"]
    # The last sigma is 0, where nothing is left to denoise; ties go to the noisier step.
    nearest = min(sigmas[:-1], key=lambda sigma: (abs(sigma - target), -sigma))

    assert entry == pytest.approx(nearest, abs=1e-5), (
        f"the widget sends denoising_start={case['denoisingStart']:.6f}, which enters at sigma {entry:.4f}; "
        f"the step nearest FLUX's {target:.4f} is at {nearest:.4f}"
    )
