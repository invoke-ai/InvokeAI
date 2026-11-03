import numpy as np
from PIL import Image

from invokeai.backend.image_util.invisible_watermark import InvisibleWatermark


def _textured_rgb(size: int = 512) -> np.ndarray:
    """A smooth gradient with some texture: enough detail for the DWT-DCT watermark to embed and decode."""
    y, x = np.mgrid[0:size, 0:size]
    rgb = np.stack([(x * 255) // size, (y * 255) // size, ((x + y) * 127) // size], axis=-1)
    noise = np.random.default_rng(0).integers(0, 32, size=rgb.shape)
    return np.clip(rgb + noise, 0, 255).astype(np.uint8)


def test_watermarking_keeps_the_transparent_surround_of_a_masked_result():
    rgb = _textured_rgb()
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    alpha[128:384, 128:384] = 255
    image = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")

    watermarked = InvisibleWatermark.add_watermark(image, "InvokeAI")

    assert watermarked.mode == "RGBA"
    assert np.array_equal(np.array(watermarked.getchannel("A")), alpha)
    assert InvisibleWatermark.decode_watermark(watermarked, length=8) == "InvokeAI"


def test_watermarking_an_image_without_alpha_returns_it_opaque():
    image = Image.fromarray(_textured_rgb(), mode="RGB")

    watermarked = InvisibleWatermark.add_watermark(image, "InvokeAI")

    assert watermarked.mode == "RGBA"
    assert np.all(np.array(watermarked.getchannel("A")) == 255)
