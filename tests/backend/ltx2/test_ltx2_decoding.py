"""LTX-2 model plumbing: the tiling scope, the decode guards, and what the reservations price."""

from types import SimpleNamespace

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.ltx2.text_conditioning import (
    estimate_connector_working_memory,
    estimate_tower_working_memory,
)
from invokeai.backend.ltx2.video_decoding import decode_audio_latents, decode_video_latents, scoped_ltx2_tiling
from invokeai.backend.util.vae_working_memory import (
    estimate_audio_working_memory_ltx2,
    estimate_vae_working_memory_ltx2,
)

TILING_ATTRS = (
    "use_tiling",
    "use_framewise_encoding",
    "use_framewise_decoding",
    "tile_sample_min_height",
    "tile_sample_min_width",
    "tile_sample_min_num_frames",
    "tile_sample_stride_height",
    "tile_sample_stride_width",
    "tile_sample_stride_num_frames",
)


class VaeStub(torch.nn.Module):
    """The tiling surface of ``AutoencoderKLLTX2Video``, at its released defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.spatial_compression_ratio = 32
        self.temporal_compression_ratio = 8
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(self, **kwargs) -> None:
        self.use_tiling = True
        for name, value in kwargs.items():
            if value:
                setattr(self, name, value)


def _snapshot(vae: VaeStub) -> dict[str, object]:
    return {name: getattr(vae, name) for name in TILING_ATTRS}


def test_the_tiling_scope_restores_every_attribute_it_touches() -> None:
    """The VAE belongs to the model cache and is shared across invocations. `disable_tiling` clears
    neither the geometry nor the framewise flags, so leaving any of them set would silently change
    every later decode in the session."""
    vae = VaeStub()
    before = _snapshot(vae)

    with scoped_ltx2_tiling(vae, tile_size=768, temporal_tile=32):
        assert vae.use_tiling is True
        assert vae.use_framewise_decoding is True
        assert vae.tile_sample_min_height == 768
        assert vae.tile_sample_min_num_frames == 32

    assert _snapshot(vae) == before


def test_the_tiling_scope_restores_after_a_cancel() -> None:
    vae = VaeStub()
    before = _snapshot(vae)

    with pytest.raises(CanceledException), scoped_ltx2_tiling(vae, tile_size=512, temporal_tile=16):
        raise CanceledException

    assert _snapshot(vae) == before


@pytest.mark.parametrize(
    ("tile_size", "temporal_tile", "expected"),
    [
        # The released defaults, which the working-memory estimate is fitted to.
        (512, 16, (512, 448, 16, 8)),
        # Snapped down onto the compression grid: a tile that is not a whole number of latents
        # decodes a band the blend cannot line up.
        (700, 20, (672, 576, 16, 8)),
        # Never below one latent in either axis, however small the request.
        (1, 1, (32, 32, 8, 8)),
    ],
)
def test_the_tile_geometry_stays_on_the_compression_grid(
    tile_size: int, temporal_tile: int, expected: tuple[int, int, int, int]
) -> None:
    vae = VaeStub()

    with scoped_ltx2_tiling(vae, tile_size=tile_size, temporal_tile=temporal_tile):
        geometry = (
            vae.tile_sample_min_height,
            vae.tile_sample_stride_height,
            vae.tile_sample_min_num_frames,
            vae.tile_sample_stride_num_frames,
        )

    assert geometry == expected
    assert geometry[0] % vae.spatial_compression_ratio == 0
    assert geometry[2] % vae.temporal_compression_ratio == 0


@pytest.mark.parametrize(
    ("latents", "match"),
    [
        (torch.zeros(1, 128, 2, 4, 4), "channels"),
        (torch.zeros(2, 128, 2, 4, 4), "one 5D clip"),
        (torch.zeros(1, 128, 4, 4), "one 5D clip"),
    ],
)
def test_a_video_decode_refuses_latents_it_cannot_be_asked_to_decode(latents: torch.Tensor, match: str) -> None:
    """A channel or rank mismatch would otherwise surface as a shape error inside the decoder,
    after the VAE has been locked onto the device."""
    vae = VaeStub()
    vae.config = SimpleNamespace(latent_channels=32, scaling_factor=1.0)

    with pytest.raises(ValueError, match=match):
        decode_video_latents(vae, latents, tile_size=512, temporal_tile=16)


def test_an_audio_decode_refuses_unpacked_latents() -> None:
    """Packed rows are what the audio statistics are defined against; a 4D tensor here would
    denormalize against the wrong axis and come back as noise rather than an error."""
    with pytest.raises(ValueError, match=r"packed latents"):
        decode_audio_latents(VaeStub(), VaeStub(), torch.zeros(1, 8, 4, 16))


def test_the_video_reservation_follows_the_tile_and_the_clip() -> None:
    """It has to hold at the shapes it was measured at, and it has to keep responding to the knobs
    the node exposes -- a flat number would be wrong for every size but one."""
    vae = VaeStub()
    at = lambda **kwargs: estimate_vae_working_memory_ltx2("decode", vae, 704, 1248, 33, **kwargs) / 2**30  # noqa: E731

    # Measured on a W7900 at these shapes: 1.12 GiB, 3.85 GiB, 6.45 GiB.
    assert at(tile_size=256, temporal_tile=16) > 1.12
    assert at(tile_size=512, temporal_tile=16) > 3.85
    assert at(tile_size=768, temporal_tile=16) > 6.45

    assert at(tile_size=256, temporal_tile=16) < at(tile_size=512, temporal_tile=16)
    assert at(tile_size=512, temporal_tile=16) < at(tile_size=512, temporal_tile=32)
    # The clip term grows with the clip even though the tile term does not.
    assert estimate_vae_working_memory_ltx2("decode", vae, 704, 1248, 121) > estimate_vae_working_memory_ltx2(
        "decode", vae, 704, 1248, 33
    )


def test_the_tiled_encode_reservation_holds_at_the_shapes_it_was_measured_at() -> None:
    """A whole conditioning clip is encoded tiled, which is a different cost shape from the
    first-frame encode the untiled branch is fitted to -- reading that one here would over-reserve
    it roughly twofold, and cache the rest of the graph loses is not free."""
    vae = VaeStub()

    def tiled(frames: int, h: int = 704, w: int = 1248) -> float:
        return (
            estimate_vae_working_memory_ltx2("encode", vae, h, w, frames, tile_size=512, temporal_tile=16, tiled=True)
            / 2**30
        )

    # Measured on a W7900 at 512/16: 2.82, 2.87, 3.20, 3.78 and 4.02 GiB.
    assert tiled(33, 512, 768) > 2.82
    assert tiled(121, 512, 768) > 2.87
    assert tiled(121) > 3.20
    assert tiled(241) > 3.78
    assert tiled(121, 1088, 1920) > 4.02

    # The tile term is flat in the clip's length -- that is the whole reason tiling is used here --
    # so only the clip term grows, and a long clip must not be estimated as a short one.
    assert tiled(241) > tiled(121) > tiled(33)
    # And the untiled branch stays where it was: it is fitted to one frame, where the per-tile
    # fixed cost dominates, so it is far larger per element and must not be picked for a clip.
    untiled = estimate_vae_working_memory_ltx2("encode", vae, 704, 1248, 121)
    assert untiled > tiled(121) * 2**30 * 2


def test_the_video_reservation_covers_a_clip_far_longer_than_the_measured_ones() -> None:
    """The tile term is flat in clip length and holds all the fitted margin; the clip term is the
    one that grows, so the longest clip the field accepts is where an under-count would surface."""
    vae = VaeStub()
    # 481 frames at 1248x704: the temporal tiler holds ~2 clips of rows plus the concatenated clip,
    # which is 7.04 GiB of bf16 pixels on top of ~1.8 GiB of tile activation.
    assert estimate_vae_working_memory_ltx2("decode", vae, 704, 1248, 481, tile_size=512, temporal_tile=16) > int(
        8.9 * 2**30
    )


def test_the_video_reservation_prices_a_tile_the_clip_cannot_fill() -> None:
    """A clip shorter than one temporal tile still decodes a whole tile: the tile is two *latent*
    frames, which cover 16 pixel frames however few the clip has."""
    vae = VaeStub()
    short = estimate_vae_working_memory_ltx2("decode", vae, 704, 1248, 9, tile_size=512, temporal_tile=16)
    long = estimate_vae_working_memory_ltx2("decode", vae, 704, 1248, 33, tile_size=512, temporal_tile=16)

    assert short > 3.0 * 2**30
    assert short < long


def test_the_audio_reservation_is_a_line_in_the_clip_length() -> None:
    """Nothing in the audio path is tiled, so the whole soundtrack is decoded at once and an absurd
    request has to fail at the reservation rather than inside a forward."""
    # Measured on a W7900: 0.71, 1.30, 2.18 and 6.93 GiB.
    for latents, measured in ((126, 0.71), (251, 1.30), (501, 2.18), (1251, 6.93)):
        assert estimate_audio_working_memory_ltx2(latents) / 2**30 > measured

    # 481 frames at 1 fps is eight minutes of audio; asking the cache for that is the honest answer.
    assert estimate_audio_working_memory_ltx2(12025) > 70 * 2**30


def test_the_prompt_encode_reservations_cover_what_each_stage_holds() -> None:
    """The Gemma tower is the only encoder here that keeps every layer's hidden state and then
    stacks them, so its reservation has to follow the layer count and the padded length rather than
    being a constant -- and the connectors then hold that whole stack again, per modality."""
    tower = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=48, hidden_size=3840), dtype=torch.bfloat16)

    # Measured on a W7900 at 1024 tokens: 1.92 GiB for the tower, 1.74 GiB for the connectors.
    at_1024 = estimate_tower_working_memory(tower, 1024)
    assert at_1024 / 2**30 > 1.92
    assert at_1024 > estimate_tower_working_memory(tower, 512)

    smaller = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=24, hidden_size=3840), dtype=torch.bfloat16)
    assert estimate_tower_working_memory(smaller, 1024) < at_1024

    states = torch.zeros(1, 1024, 3840 * 49, dtype=torch.bfloat16)
    assert estimate_connector_working_memory(states) / 2**30 > 1.74
