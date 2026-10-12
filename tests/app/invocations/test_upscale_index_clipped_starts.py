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

The blend only helps if the scheduler then starts where it left the latents. The fixture also records
which schedulers the widget offers each node, and that has to be exactly the set that does: one that
builds its own schedule would denoise the frame as if it were pure noise, or leave noise in it.
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
from invokeai.backend.anima.scheduler_driver import AnimaSchedulerDriver
from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_MAP, ZIMAGE_SCHEDULER_MAP

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
FIXTURE_DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASES = FIXTURE_DATA["cases"]
OFFERED_SCHEDULERS = FIXTURE_DATA["schedulers"]
Z_IMAGE_MODULE = "invokeai.app.invocations.z_image.z_image_denoise"
ANIMA_MODULE = "invokeai.app.invocations.anima.anima_denoise"


class _Stop(Exception):
    """Ends the run at the probe that raised it."""


def _run_until_stopped(node, module: str, dtype_attr: str, context, noise: torch.Tensor, patches) -> None:
    with ExitStack() as stack:
        stack.enter_context(patch(f"{module}.TorchDevice.choose_torch_device", return_value=torch.device("cpu")))
        # float32: the entry sigma is read back out of the tensor, and bfloat16 keeps three digits.
        stack.enter_context(patch(f"{module}.TorchDevice.{dtype_attr}", return_value=torch.float32))
        stack.enter_context(patch.object(node, "_prepare_noise_tensor", return_value=noise))
        for extra in patches:
            stack.enter_context(extra)
        with pytest.raises(_Stop):
            node._run_diffusion(context)


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
        raise _Stop

    context = MagicMock()
    context.tensors.load.return_value = latents
    _run_until_stopped(
        node,
        module,
        dtype_attr,
        context,
        noise,
        [patch.object(node, "_get_sigmas", record), patch.object(node, "_prep_inpaint_mask", capture), *patches],
    )

    blended = captured["latents"]
    assert torch.allclose(blended, blended.flatten()[:1].expand_as(blended)), "the blend should be uniform"
    return schedule["sigmas"], float(blended.flatten()[0])


def _z_image_node(side: int, steps: int, denoising_start: float, scheduler: str) -> tuple:
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
        scheduler=scheduler,
    )
    regional = SimpleNamespace(regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16)))
    conditioning = [
        patch(f"{Z_IMAGE_MODULE}.ZImageRegionalPromptingExtension.from_text_conditionings", return_value=regional),
        patch.object(
            node,
            "_load_text_conditioning",
            return_value=[SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16), mask=None)],
        ),
    ]
    return node, conditioning


def z_image_entry(side: int, steps: int, denoising_start: float) -> tuple[list[float], float]:
    node, conditioning = _z_image_node(side, steps, denoising_start, "euler")
    return _entry(
        node,
        Z_IMAGE_MODULE,
        "choose_bfloat16_safe_dtype",
        torch.zeros(1, 16, 8, 8),
        torch.ones(1, 16, 8, 8),
        conditioning,
    )


def z_image_scheduler_entry(side: int, steps: int, denoising_start: float, scheduler: str) -> float:
    """The sigma `z_image_denoise` hands the transformer first, read off the scheduler it set up."""
    node, conditioning = _z_image_node(side, steps, denoising_start, scheduler)
    made = []

    def recording(scheduler_class):
        def make(**kwargs):
            made.append(scheduler_class(**kwargs))
            return made[-1]

        return make

    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)
    # The first call after the scheduler has its timesteps.
    context.models.get_config.side_effect = _Stop
    _run_until_stopped(
        node,
        Z_IMAGE_MODULE,
        "choose_bfloat16_safe_dtype",
        context,
        torch.ones(1, 16, 8, 8),
        [
            patch.dict(ZIMAGE_SCHEDULER_MAP, {name: recording(cls) for name, cls in ZIMAGE_SCHEDULER_MAP.items()}),
            patch.object(node, "_prep_inpaint_mask", return_value=None),
            *conditioning,
        ],
    )
    (scheduler_instance,) = made
    return scheduler_instance.timesteps[0].item() / scheduler_instance.config.num_train_timesteps


def _anima_node(side: int, steps: int, denoising_start: float, scheduler: str) -> tuple:
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
        scheduler=scheduler,
    )
    return node, [patch.object(node, "_load_text_conditionings", return_value=[SimpleNamespace(mask=None)])]


def anima_entry(side: int, steps: int, denoising_start: float) -> tuple[list[float], float]:
    node, conditioning = _anima_node(side, steps, denoising_start, "euler")
    return _entry(
        node,
        ANIMA_MODULE,
        "choose_anima_inference_dtype",
        torch.zeros(1, 16, 8, 8),
        torch.ones(1, 16, 1, 8, 8),
        conditioning,
    )


def anima_scheduler_entry(side: int, steps: int, denoising_start: float, scheduler: str) -> float:
    """The sigma `anima_denoise` hands the transformer first, read off the real scheduler driver."""
    node, conditioning = _anima_node(side, steps, denoising_start, scheduler)
    first: dict[str, float] = {}

    def probe(**kwargs):
        first["sigma"] = next(AnimaSchedulerDriver(**kwargs).iterations()).sigma_curr
        raise _Stop

    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)
    _run_until_stopped(
        node,
        ANIMA_MODULE,
        "choose_anima_inference_dtype",
        context,
        torch.ones(1, 16, 1, 8, 8),
        [
            patch(f"{ANIMA_MODULE}.AnimaSchedulerDriver", probe),
            patch.object(node, "_prep_inpaint_mask", return_value=None),
            *conditioning,
        ],
    )
    return first["sigma"]


ENTRY = {"z-image": z_image_entry, "anima": anima_entry}
SCHEDULER_ENTRY = {"z-image": z_image_scheduler_entry, "anima": anima_scheduler_entry}
SCHEDULERS = {"z-image": ZIMAGE_SCHEDULER_MAP, "anima": ANIMA_SCHEDULER_MAP}
# Under half the narrowest step at any fixture entry (0.011), so no scheduler passes a step off. DPM++
# builds its own schedule, ending at 1/1000 rather than 0, and rounds its timesteps to integers: that
# moves its entry by up to 0.0013, which is not a step.
ENTRY_TOLERANCE = 0.005


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


@pytest.mark.parametrize("base", sorted(ENTRY))
def test_the_widget_offers_exactly_the_schedulers_that_start_where_the_blend_left_the_latents(base: str) -> None:
    cases = [case for case in CASES if case["base"] == base]
    blends = [ENTRY[base](case["side"], case["steps"], case["denoisingStart"])[1] for case in cases]
    misses: dict[str, float] = {}

    for scheduler in SCHEDULERS[base]:
        # Euler is not handed to a diffusers scheduler: the node steps the schedule it blended from.
        if scheduler == "euler":
            continue
        entries = [
            SCHEDULER_ENTRY[base](case["side"], case["steps"], case["denoisingStart"], scheduler) for case in cases
        ]
        worst = max(abs(entry - blend) for entry, blend in zip(entries, blends, strict=True))
        if worst > ENTRY_TOLERANCE:
            misses[scheduler] = worst

    starting_at_the_blend = sorted(set(SCHEDULERS[base]) - set(misses))
    assert sorted(OFFERED_SCHEDULERS[base]) == starting_at_the_blend, (
        f"the widget offers {base} {sorted(OFFERED_SCHEDULERS[base])}, but the schedulers that start where the "
        f"blend left the latents are {starting_at_the_blend} (worst misses: "
        + ", ".join(f"{name} {miss:.3f}" for name, miss in sorted(misses.items()))
        + "). Update `schedulers` in `features/upscale/core/settings.ts`, and FLUX.1's with Z-Image's: "
        + "`flux/denoise.py` uses the same diffusers schedulers."
    )
