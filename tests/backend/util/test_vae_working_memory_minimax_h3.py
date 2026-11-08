"""The MiniMax H3 VAE working-memory estimate, calibrated against measured decodes.

Measured 2026-08-09 on a W7900 (gfx1100, fp32, real released config), peak RESERVED — which is what
the reservation must cover, since the caching allocator holds more than is live:

    one 256px tile, 28 frames      0.56 GiB
    one 768x1344 chunk, 28 frames  1.81 GiB
    full 768x1344 decode,  90f     4.37 GiB
    full 768x1344 decode, 243f     7.92 GiB

The 243-frame case is the one that crashed: it needed 7.92 GiB against a 6.49 GiB reservation, and
partial loading packs VRAM up to exactly the reservation, so the shortfall was fatal.
"""

import torch

from invokeai.backend.util.vae_working_memory import (
    MINIMAX_H3_ALLOCATOR_HEADROOM,
    MINIMAX_H3_CHUNK_BYTES_PER_PIXEL,
    estimate_vae_working_memory_minimax_h3,
)

GIB = 1024**3

# Released FL2VA geometry: 5-token chunks, 2 tokens of overlap, 4x temporal compression (28 pixel
# frames per chunk call), 17-frame encode clips.
RELEASED = {"tokens_chunk_size": 5, "token_overlap": 2, "temporal_compression_ratio": 4, "clip_length": 17}
RELEASED_CHUNK_FRAMES = (5 + 2) * 4

# Deliberately NOT the estimator's fallback defaults, so a renamed attribute read is caught instead
# of silently resolving to the same number.
ODD = {"tokens_chunk_size": 3, "token_overlap": 1, "temporal_compression_ratio": 2, "clip_length": 9}
ODD_CHUNK_FRAMES = (3 + 1) * 2


class _FakeVAE(torch.nn.Module):
    def __init__(self, geometry: dict) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))  # fp32-pinned weights
        self.tokens_chunk_size = geometry["tokens_chunk_size"]
        self.token_overlap = geometry["token_overlap"]
        self.temporal_compression_ratio = geometry["temporal_compression_ratio"]

        class _Config:
            clip_length = geometry["clip_length"]
            latent_channels = 24

        self.config = _Config()


def _estimate(operation: str, frames: int, width: int = 768, height: int = 1344, geometry: dict = RELEASED) -> int:
    return estimate_vae_working_memory_minimax_h3(
        operation=operation,  # type: ignore[arg-type]
        vae=_FakeVAE(geometry),
        pixel_height=height,
        pixel_width=width,
        pixel_frames=frames,
    )


def _expected(chunk_frames: int, frames: int, width: int, height: int, clip_copies: int) -> int:
    chunk = chunk_frames * height * width * 4 * MINIMAX_H3_CHUNK_BYTES_PER_PIXEL
    clip = clip_copies * 3 * frames * height * width * 4
    return int((chunk + clip) * MINIMAX_H3_ALLOCATOR_HEADROOM)


def test_decode_matches_the_calibrated_formula():
    assert _estimate("decode", 243) == _expected(RELEASED_CHUNK_FRAMES, 243, 768, 1344, clip_copies=2)


def test_encode_uses_the_clip_length_and_one_clip_copy():
    assert _estimate("encode", 243) == _expected(RELEASED["clip_length"], 243, 768, 1344, clip_copies=1)


def test_geometry_is_read_from_the_instance_not_assumed():
    """A VAE whose chunking differs from the released defaults must be estimated on its own terms."""
    odd = _estimate("decode", 243, geometry=ODD)
    assert odd == _expected(ODD_CHUNK_FRAMES, 243, 768, 1344, clip_copies=2)
    # ...and that is genuinely different from the released-geometry answer, so a misread attribute
    # (which would fall back to the released defaults) cannot pass this test.
    assert odd != _estimate("decode", 243)
    assert _estimate("encode", 243, geometry=ODD) != _estimate("encode", 243)


def test_a_clip_shorter_than_one_chunk_only_pays_for_its_frames():
    assert _estimate("decode", 5) == _expected(5, 5, 768, 1344, clip_copies=2)


def test_covers_the_measured_peak_of_the_decode_that_crashed():
    # Measured 7.92 GiB reserved for 243 frames at 768x1344; 4.37 GiB at 90 frames.
    assert _estimate("decode", 243) > 7.92 * GIB
    assert _estimate("decode", 90) > 4.37 * GIB
    # ...without the runaway conservatism that would push a 24 GB card's VAE out of VRAM: the VAE
    # is 9.70 GiB, so the reservation plus weights has to leave room on a 24 GiB device.
    assert (_estimate("decode", 90, width=768, height=1024) / GIB) + 9.70 < 24


def test_grows_with_clip_length_and_canvas():
    assert _estimate("decode", 243) > _estimate("decode", 90)
    assert _estimate("decode", 90, width=768, height=1344) > _estimate("decode", 90, width=512, height=512)


def test_falls_back_to_released_geometry_when_attributes_are_missing():
    class _BareVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    bare = estimate_vae_working_memory_minimax_h3(
        operation="decode", vae=_BareVAE(), pixel_height=1344, pixel_width=768, pixel_frames=243
    )
    # Not a tautology against the fake: pinned to the released-geometry arithmetic directly.
    assert bare == _expected(RELEASED_CHUNK_FRAMES, 243, 768, 1344, clip_copies=2)
