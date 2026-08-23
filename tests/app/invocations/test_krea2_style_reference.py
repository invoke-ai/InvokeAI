from types import SimpleNamespace

import pytest
import torch
from PIL import Image as PILImage

from invokeai.app.invocations.krea2_style_reference import (
    Krea2StyleReferenceInvocation,
    fit_image_to_box,
)


def _image(width: int, height: int, color: tuple[int, int, int] = (10, 120, 200)) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color)


@pytest.mark.parametrize("fit", ["crop", "contain", "stretch"])
@pytest.mark.parametrize(("source_width", "source_height"), [(640, 480), (480, 640), (1024, 1024), (37, 800)])
def test_every_fit_mode_produces_exactly_the_target_size(fit: str, source_width: int, source_height: int) -> None:
    # The reference's image tokens are appended to the target's and share its rotary embedding, so an
    # off-by-one here becomes a shape error deep inside attention.
    result = fit_image_to_box(_image(source_width, source_height), 512, 256, fit)
    assert result.size == (512, 256)


def test_crop_keeps_the_aspect_ratio_by_discarding_the_overhang() -> None:
    # A 2x2 checker: cropping a wide source to a square must keep the vertical proportions intact.
    source = PILImage.new("RGB", (400, 200), (0, 0, 0))
    source.paste(PILImage.new("RGB", (400, 100), (255, 255, 255)), (0, 0))

    result = fit_image_to_box(source, 200, 200, "crop")

    # The top half stays white and the bottom half black -- no vertical squashing.
    assert result.getpixel((100, 50)) == (255, 255, 255)
    assert result.getpixel((100, 150)) == (0, 0, 0)


def test_contain_letterboxes_on_white_without_distorting() -> None:
    result = fit_image_to_box(_image(400, 200, (0, 0, 0)), 200, 200, "contain")

    # 400x200 scaled to fit 200x200 gives 200x100, centred vertically with white bars above and below.
    assert result.getpixel((100, 100)) == (0, 0, 0)
    assert result.getpixel((100, 5)) == (255, 255, 255)
    assert result.getpixel((100, 195)) == (255, 255, 255)


def test_stretch_fills_the_whole_box() -> None:
    result = fit_image_to_box(_image(400, 200, (0, 0, 0)), 200, 200, "stretch")
    assert result.getpixel((100, 5)) == (0, 0, 0)
    assert result.getpixel((100, 195)) == (0, 0, 0)


def test_fit_rejects_a_degenerate_source() -> None:
    with pytest.raises(ValueError, match="invalid dimensions"):
        fit_image_to_box(PILImage.new("RGB", (0, 0)), 64, 64, "crop")


def _invocation(**overrides) -> Krea2StyleReferenceInvocation:
    defaults = {
        "image": SimpleNamespace(image_name="reference"),
        "vae": SimpleNamespace(vae=SimpleNamespace()),
        "width": 64,
        "height": 64,
        "fit": "crop",
        "style_strength": 1.0,
        "blocks": "7-27",
        "ref_k_strength": 1.06,
        "adain_strength": 0.85,
        "value_mode": "target_adain_plus_ref",
        "value_adain_strength": 0.65,
        "ref_value_mix": 1.0,
        "high_scale_start": 1.04,
        "high_scale_end": 0.0,
        "low_scale_start": 1.0,
        "low_scale_end": 1.10,
        "beta": 2.5,
    }
    defaults.update(overrides)
    return Krea2StyleReferenceInvocation.model_construct(**defaults)


def _context(saved: dict) -> SimpleNamespace:
    def save(tensor: torch.Tensor) -> str:
        saved["tensor"] = tensor
        return "saved"

    return SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name, _mode: _image(400, 200)),
        models=SimpleNamespace(load=lambda _identifier: object()),
        tensors=SimpleNamespace(save=save),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )


def test_invoke_encodes_the_reference_and_carries_the_settings(monkeypatch) -> None:
    encoded: dict = {}

    def fake_vae_encode(*, vae_info, image_tensor):
        encoded["image_tensor"] = image_tensor
        return torch.zeros(1, 16, 1, 8, 8)

    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_style_reference.QwenImageImageToLatentsInvocation.vae_encode",
        staticmethod(fake_vae_encode),
    )
    monkeypatch.setattr("invokeai.app.invocations.krea2_style_reference.TorchDevice.empty_cache", lambda: None)

    saved: dict = {}
    output = _invocation(style_strength=0.6, low_scale_end=1.25).invoke(_context(saved))

    # The image reaches the VAE at exactly the requested size, normalized to [-1, 1].
    assert encoded["image_tensor"].shape == (1, 3, 64, 64)
    assert encoded["image_tensor"].min() >= -1.0 and encoded["image_tensor"].max() <= 1.0

    field = output.style_reference
    assert field.reference_latents_name == "saved"
    assert (field.width, field.height) == (64, 64)
    assert field.style_strength == pytest.approx(0.6)
    assert field.low_scale_end == pytest.approx(1.25)
    assert field.blocks == "7-27"


def test_invoke_rejects_a_malformed_block_spec(monkeypatch) -> None:
    # Fails here rather than several nodes later, halfway through a denoise.
    with pytest.raises(ValueError, match="selects blocks"):
        _invocation(blocks="7-99").invoke(_context({}))
