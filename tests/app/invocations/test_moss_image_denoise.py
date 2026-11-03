"""`moss_image_denoise` at its boundary: the FLUX.2 raw latent it takes and returns, the inpaint mask it builds, and the
inputs it refuses before any weight reaches a device.

The sampling itself is `invokeai.backend.moss_image.denoise`'s, covered in tests/backend/moss_image. Here a tiny
vendored transformer runs at float32 on the CPU behind a fake invocation context that stands in only for the model
cache and the tensor and conditioning stores.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from invokeai.app.invocations.fields import DenoiseMaskField, LatentsField, MossImageConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField, VAEField
from invokeai.app.invocations.moss_image import moss_image_denoise as node_module
from invokeai.app.invocations.moss_image.moss_image_denoise import MossImageDenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.moss_image import working_memory as working_memory_module
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.moss_image.sampling import get_moss_noise, get_moss_sigmas, to_model_space
from invokeai.backend.moss_image.working_memory import MOSS_IMAGE_DENOISE_SPEC
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    MossImageConditioningInfo,
)
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
from invokeai.backend.working_memory.platform import platform_for

_TEXT_DIM = 24
# Not square, so a swapped height and width shows: a 12x8 raw latent, 6x4 model-space tokens.
_WIDTH = 64
_HEIGHT = 96


class _Loaded:
    """The parts of a cache `LoadedModel` the node uses, recording the working memory it was asked to reserve."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.compute_device = torch.device("cpu")
        self.reservations: list[WorkingMemoryEstimate | int | None] = []

    @contextmanager
    def model_on_device(
        self, working_mem_bytes: int | None = None, *, working_memory: WorkingMemoryEstimate | None = None
    ):
        self.reservations.append(working_memory if working_memory is not None else working_mem_bytes)
        yield (None, self.model)

    @contextmanager
    def model_in_ram(self):
        yield self.model


def _transformer() -> MOSSImageTransformer:
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=_TEXT_DIM, freq_dim=16, latent_channels=128
    ).eval()


def _vae() -> SimpleNamespace:
    bn = torch.nn.BatchNorm2d(128, eps=1e-4, affine=False)
    generator = torch.Generator().manual_seed(2)
    bn.running_mean.copy_(torch.randn(128, generator=generator))
    bn.running_var.copy_(torch.rand(128, generator=generator) + 0.5)
    return SimpleNamespace(bn=bn.eval())


def _identifier(key: str, submodel: SubModelType) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key, hash=key, name=key, base=BaseModelType.MossImage, type=ModelType.Main, submodel_type=submodel
    )


class _Context:
    def __init__(self, tensors: dict[str, torch.Tensor] | None = None, text_dim: int = _TEXT_DIM) -> None:
        self.transformer = _Loaded(_transformer())
        self.vae = _Loaded(_vae())
        self.tensors_in = dict(tensors or {})
        self.saved: dict[str, torch.Tensor] = {}
        self.previews: list[tuple[Any, BaseModelType]] = []
        generator = torch.Generator().manual_seed(3)
        conditionings = {
            name: ConditioningFieldData(
                conditionings=[
                    MossImageConditioningInfo(prompt_embeds=torch.randn(length, text_dim, generator=generator))
                ]
            )
            for name, length in (("pos", 7), ("neg", 3))
        }
        self.models = SimpleNamespace(load=self._load_model)
        self.tensors = SimpleNamespace(load=self.tensors_in.__getitem__, save=self._save)
        self.conditioning = SimpleNamespace(load=conditionings.__getitem__)
        self.util = SimpleNamespace(sd_step_callback=lambda state, base: self.previews.append((state, base)))

    def _load_model(self, identifier: ModelIdentifierField) -> _Loaded:
        return {"transformer": self.transformer, "vae": self.vae}[identifier.key]

    def _save(self, tensor: torch.Tensor) -> str:
        name = f"latents-{len(self.saved)}"
        self.saved[name] = tensor
        return name


def _invocation(**overrides: Any) -> MossImageDenoiseInvocation:
    fields: dict[str, Any] = {
        "transformer": TransformerField(transformer=_identifier("transformer", SubModelType.Transformer), loras=[]),
        "vae": VAEField(vae=_identifier("vae", SubModelType.VAE)),
        "positive_conditioning": MossImageConditioningField(conditioning_name="pos"),
        "negative_conditioning": MossImageConditioningField(conditioning_name="neg"),
        "cfg_scale": 3.0,
        "uncond_skip_layers": [1],
        "width": _WIDTH,
        "height": _HEIGHT,
        "steps": 3,
        "seed": 7,
    }
    fields.update(overrides)
    return MossImageDenoiseInvocation(**fields)


def _init_latents() -> torch.Tensor:
    return torch.randn(1, 32, _HEIGHT // 8, _WIDTH // 8, generator=torch.Generator().manual_seed(4))


def _run(context: _Context, **overrides: Any) -> torch.Tensor:
    output = _invocation(**overrides).invoke(context)  # type: ignore[arg-type]
    return context.saved[output.latents.latents_name]


def test_txt2img_returns_flux2_raw_latents_at_the_requested_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Whether CPU SDPA materializes scores is the build's business; the activation term is what is pinned here.
    monkeypatch.setattr(working_memory_module, "sdpa_score_matrix_bytes", lambda **_: 0)
    context = _Context()

    output = _invocation().invoke(context)  # type: ignore[arg-type]

    latents = context.saved[output.latents.latents_name]
    assert latents.shape == (1, 32, _HEIGHT // 8, _WIDTH // 8)
    assert latents.dtype is torch.float32 and torch.isfinite(latents).all()
    assert (output.width, output.height) == (_WIDTH, _HEIGHT)
    # One preview per step, rendered through MOSS-Image's own latent space.
    assert [base for _, base in context.previews] == [BaseModelType.MossImage] * 3
    # Reserved for the image's 4x6 tokens plus the longer prompt (7 tokens; the negative has 3): the two passes run
    # one after the other, so their prompts are not summed.
    (estimate,) = context.transformer.reservations
    assert isinstance(estimate, WorkingMemoryEstimate) and estimate.estimator == MOSS_IMAGE_DENOISE_SPEC.id
    k_token = shipped_coefficients(MOSS_IMAGE_DENOISE_SPEC, platform_for(torch.device("cpu"))).values["k_token"]
    assert estimate.reservation_bytes == int((24 + 7) * k_token)


def test_the_seed_alone_decides_the_noise() -> None:
    first = _run(_Context(), seed=7)

    assert torch.equal(_run(_Context(), seed=7), first)
    assert not torch.allclose(_run(_Context(), seed=8), first)


def test_a_window_ending_early_hands_the_loop_its_own_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    # The window is re-spaced to the requested step count, so counting steps cannot tell it from the full schedule;
    # the sigmas themselves can.
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return kwargs["latents"]

    monkeypatch.setattr(node_module, "denoise", capture)

    _run(_Context(), denoising_end=0.5)

    assert captured["sigmas"] == get_moss_sigmas(3, 0.0, 0.5)
    assert captured["sigmas"][0] == 1.0 and captured["sigmas"][-1] == 0.5


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_img2img_hands_the_loop_the_reference_start_and_schedule(
    monkeypatch: pytest.MonkeyPatch, cfg_scale: float
) -> None:
    """What reaches the backend loop, computed here from the definitions rather than read back from the node."""
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> torch.Tensor:
        captured.update(kwargs)
        return kwargs["latents"]

    monkeypatch.setattr(node_module, "denoise", capture)
    init = _init_latents()
    context = _Context({"init": init})

    _run(
        context,
        latents=LatentsField(latents_name="init"),
        denoising_start=0.5,
        steps=4,
        seed=11,
        cfg_scale=cfg_scale,
        uncond_skip_layers=[0, 2],
    )

    sigmas = get_moss_sigmas(4, 0.5, 1.0)
    assert captured["sigmas"] == sigmas
    # The noise is already in model space; the init latent is taken there through the VAE's BatchNorm.
    expected_start = sigmas[0] * get_moss_noise(11, _HEIGHT, _WIDTH) + (1.0 - sigmas[0]) * to_model_space(
        init, context.vae.model.bn
    )
    torch.testing.assert_close(captured["latents"], expected_start)
    assert captured["inpaint_extension"] is None
    assert captured["cfg_scale"] == cfg_scale
    assert captured["positive"].shape == (1, 7, _TEXT_DIM)
    if cfg_scale > 1.0:
        assert captured["negative"].shape == (1, 3, _TEXT_DIM)
        assert captured["uncond_skip_layers"] == [0, 2]
    else:
        assert captured["negative"] is None
        assert captured["uncond_skip_layers"] == []


def test_img2img_from_late_in_the_schedule_stays_close_to_the_init_latents() -> None:
    init = _init_latents()

    latents = _run(_Context({"init": init}), latents=LatentsField(latents_name="init"), denoising_start=0.98, steps=2)

    # Ignoring the init latents puts this near 1.8; starting from them at sigma 0.02 keeps it a few percent.
    assert ((latents - init).norm() / init.norm()).item() < 0.2


def test_inpainting_keeps_the_masked_out_region_of_the_init_latents() -> None:
    """The mask is resized to the raw latent's 8x grid and packed like a latent, not resized to the 16x token grid.

    The boundary sits inside a 2x2 patch: raw column 3 is kept and its patch neighbour, column 2, regenerated. A mask
    resized to the token grid would blend that patch and regenerate column 3 with it.
    """
    init = _init_latents()
    # Denoise masks are 1 where the image is kept: here from pixel 24 on, raw latent column 3.
    keep = torch.zeros(1, 1, _HEIGHT, _WIDTH)
    keep[..., 24:] = 1.0
    context = _Context({"init": init, "mask": keep})

    latents = _run(
        context,
        latents=LatentsField(latents_name="init"),
        denoise_mask=DenoiseMaskField(mask_name="mask"),
        denoising_start=0.4,
    )

    torch.testing.assert_close(latents[..., 3:], init[..., 3:], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(latents[..., 2], init[..., 2], atol=1e-2)


def test_an_empty_window_hands_the_init_latents_back_without_loading_anything() -> None:
    init = _init_latents()
    context = _Context({"init": init})

    latents = _run(context, latents=LatentsField(latents_name="init"), denoising_start=0.6, denoising_end=0.6)

    assert torch.equal(latents, init)
    assert context.transformer.reservations == []


@pytest.mark.parametrize("shape", [(1, 32, 8, 12), (1, 128, 6, 4), (2, 32, 12, 8)])
def test_init_latents_of_another_size_or_layout_are_refused(shape: tuple[int, ...]) -> None:
    context = _Context({"init": torch.randn(*shape)})

    with pytest.raises(ValueError, match=r"FLUX.2 VAE latents of shape \(1, 32, 12, 8\)"):
        _invocation(latents=LatentsField(latents_name="init")).invoke(context)  # type: ignore[arg-type]


def test_cfg_needs_a_negative_conditioning() -> None:
    with pytest.raises(ValueError, match="needs negative conditioning when cfg_scale > 1"):
        _invocation(negative_conditioning=None).invoke(_Context())  # type: ignore[arg-type]


def test_without_cfg_no_negative_conditioning_is_needed_and_the_skip_layers_are_unused() -> None:
    # The reference default [22] is out of range for this 3-block model, and must not matter without CFG.
    context = _Context()

    _invocation(negative_conditioning=None, cfg_scale=1.0, uncond_skip_layers=[22]).invoke(context)  # type: ignore[arg-type]

    assert len(context.saved) == 1


def test_out_of_range_skip_layers_are_refused_before_the_transformer_moves() -> None:
    context = _Context()

    with pytest.raises(
        ValueError, match=r"uncond_skip_layers \[3, 22\] out of range: .* has 3 blocks, numbered 0 to 2"
    ):
        _invocation(uncond_skip_layers=[0, 3, 22]).invoke(context)  # type: ignore[arg-type]
    assert context.transformer.reservations == []


def test_conditioning_from_another_encoder_width_is_refused() -> None:
    with pytest.raises(ValueError, match="encode the prompt with the Qwen3.5 9B encoder"):
        _invocation().invoke(_Context(text_dim=_TEXT_DIM + 8))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"denoise_mask": DenoiseMaskField(mask_name="mask")}, "denoise mask needs the starting latents"),
        ({"denoising_start": 0.3}, "denoising_start must be 0"),
    ],
)
def test_img2img_inputs_without_starting_latents_are_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _invocation(**overrides).invoke(_Context())  # type: ignore[arg-type]


def test_a_vae_without_batchnorm_statistics_is_refused() -> None:
    context = _Context()
    context.vae = _Loaded(SimpleNamespace())

    with pytest.raises(ValueError, match="needs a FLUX.2 VAE"):
        _invocation().invoke(context)  # type: ignore[arg-type]
