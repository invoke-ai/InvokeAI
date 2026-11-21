"""taeh3 decodes as it did before it was built on the shared TAEHV decoder.

The expected values were recorded from the standalone taeh3 decoder (before the refactor) with the
weights and latents below; the refactored module reproduced them bit for bit. The tolerance only
absorbs convolution kernels that differ by platform.
"""

import math

import torch

from invokeai.backend.minimax_h3.taehv_decoder import TAEH3Decoder


def _reference_decoder() -> TAEH3Decoder:
    """Seeded, fan-in scaled weights, independent of the order the modules are built in."""
    decoder = TAEH3Decoder().eval()
    generator = torch.Generator().manual_seed(0)
    weights = {}
    for key, value in sorted(decoder.state_dict().items()):
        fan_in = math.prod(value.shape[1:]) if value.dim() > 1 else 1
        scale = 1.0 if value.dim() > 1 else 0.1
        weights[key] = torch.randn(value.shape, generator=generator) / math.sqrt(fan_in) * scale
    weights["decoder.22.bias"] = weights["decoder.22.bias"] + 0.5
    decoder.load_state_dict(weights)
    return decoder


def test_taeh3_frames_are_unchanged() -> None:
    latents = torch.randn(1, 24, 2, 2, 3, generator=torch.Generator().manual_seed(1))

    frames = _reference_decoder().decode(latents)

    assert frames.shape == (1, 8, 3, 32, 48)
    expected_frame_means = [0.519209, 0.494186, 0.548441, 0.484827, 0.507649, 0.504651, 0.540373, 0.488824]
    expected_pixels = [0.581767, 0.545331, 0.750652, 0.845425, 0.612469, 0.713125, 0.634564, 0.710431]
    expected_last_rgb = [0.353935, 0.205887, 0.245434]
    torch.testing.assert_close(frames.mean(dim=(2, 3, 4))[0], torch.tensor(expected_frame_means), rtol=0, atol=2e-5)
    torch.testing.assert_close(frames[0, :, 0, 5, 7], torch.tensor(expected_pixels), rtol=0, atol=2e-5)
    torch.testing.assert_close(frames[0, -1, :, 20, 30], torch.tensor(expected_last_rgb), rtol=0, atol=2e-5)
