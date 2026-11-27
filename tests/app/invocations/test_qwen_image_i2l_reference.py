"""`qwen_image_i2l` with `reference` enabled encodes an edit reference the way ai-toolkit trains edit LoRAs:
RGB without alpha, downscaled (never upscaled) to at most 1 MP, sides snapped to 16, bilinear resize."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.model import ModelIdentifierField, VAEField
from invokeai.app.invocations.vae.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _encode(monkeypatch, image: Image.Image, **fields) -> torch.Tensor:
    """Run the node and return the pixel tensor it hands to the VAE."""
    seen: list[torch.Tensor] = []

    def fake_vae_encode(vae_info, image_tensor, tiled=False, tile_size=0):
        seen.append(image_tensor)
        _, _, height, width = image_tensor.shape
        return torch.zeros(1, 16, 1, height // 8, width // 8)

    monkeypatch.setattr(QwenImageImageToLatentsInvocation, "vae_encode", staticmethod(fake_vae_encode))
    vae = ModelIdentifierField(key="vae", hash="h", name="vae", base=BaseModelType.QwenImage, type=ModelType.VAE)
    invocation = QwenImageImageToLatentsInvocation(image=ImageField(image_name="ref"), vae=VAEField(vae=vae), **fields)
    context = SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name: image),
        models=SimpleNamespace(load=lambda _identifier: None),
        config=SimpleNamespace(get=lambda: SimpleNamespace(force_tiled_decode=False)),
        tensors=SimpleNamespace(save=lambda tensor: "latents"),
    )
    invocation.invoke(context)
    return seen[0]


def _noise_image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    channels = len(mode)
    pixels = np.random.default_rng(0).integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels, mode=mode)


def _as_unit_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


@pytest.mark.parametrize(
    ("width", "height", "expected_height", "expected_width"),
    [
        # 3 MP, 4:3: area 1 MP keeps 4:3 -> 886.8 x 1182.4, snapped to 880 x 1184.
        (2048, 1536, 880, 1184),
        # A tall 2:1 image at 2 MP: 1448.2 x 724.1 -> 1456 x 720.
        (1000, 2000, 1456, 720),
        # Under 1 MP is not scaled, only snapped: 300 -> 304, 500 -> 496.
        (500, 300, 304, 496),
        # Already on the grid and under budget: untouched.
        (512, 256, 256, 512),
        # Sides never snap below 16.
        (10, 6, 16, 16),
    ],
)
def test_a_reference_is_downscaled_to_one_megapixel_and_snapped_to_16(
    monkeypatch, width, height, expected_height, expected_width
) -> None:
    tensor = _encode(monkeypatch, _noise_image(width, height), reference=True)

    assert tensor.shape == (1, 3, expected_height, expected_width)


def test_a_reference_is_resized_bilinearly_without_antialiasing(monkeypatch) -> None:
    image = _noise_image(2048, 1536)

    tensor = _encode(monkeypatch, image, reference=True)

    expected = F.interpolate(_as_unit_tensor(image), size=(880, 1184), mode="bilinear", antialias=False) * 2 - 1
    torch.testing.assert_close(tensor, expected)


def test_a_reference_on_the_grid_is_encoded_pixel_exact(monkeypatch) -> None:
    image = _noise_image(512, 256)

    tensor = _encode(monkeypatch, image, reference=True)

    assert torch.equal(tensor, _as_unit_tensor(image) * 2 - 1)


def test_a_reference_drops_alpha_instead_of_compositing_it(monkeypatch) -> None:
    # Fully transparent red: compositing onto white or black would change the colour, dropping alpha keeps it.
    image = Image.new("RGBA", (32, 32), (255, 0, 0, 0))

    tensor = _encode(monkeypatch, image, reference=True)

    assert torch.equal(tensor[0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))


def test_a_reference_refuses_an_explicit_size(monkeypatch) -> None:
    with pytest.raises(ValueError, match="sized automatically"):
        _encode(monkeypatch, _noise_image(64, 64), reference=True, width=32, height=32)


def test_without_reference_a_large_image_is_encoded_at_its_own_size(monkeypatch) -> None:
    # The default stays the Qwen-Image / Krea-2 img2img path: no 1 MP budget.
    tensor = _encode(monkeypatch, _noise_image(2048, 1536))

    assert tensor.shape == (1, 3, 1536, 2048)
