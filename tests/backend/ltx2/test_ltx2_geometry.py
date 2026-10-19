"""LTX-2 latent geometry: what the canvas rules allow and how latents pack."""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.ltx2.constants import (
    LTX2_FRAME_MODULUS,
    LTX2_PATCH_SIZE,
    LTX2_PATCH_SIZE_T,
    LTX2_SPATIAL_COMPRESSION,
    LTX2_TEMPORAL_COMPRESSION,
)
from invokeai.backend.ltx2.packing import (
    audio_latent_count,
    base_canvas,
    latent_frame_count,
    pack_audio_latents,
    pack_video_latents,
    prepare_keyframe_coords,
    require_patch_geometry,
    resolve_canvas,
    snap_num_frames,
    snap_num_frames_down,
    unpack_audio_latents,
    unpack_video_latents,
    validate_canvas,
    validate_num_frames,
    video_sequence_length,
)


def test_a_packed_row_is_the_channel_vector_at_its_latent_position() -> None:
    """The packing is what places a token in space, so its order is a contract with the model.

    Rows run frame-major, then row-major within a frame; each row is that position's channels. The
    expectation is built by indexing the 5D tensor, not by rerunning the permutation.
    """
    latents = torch.randn(1, 4, 3, 2, 5)
    packed = pack_video_latents(latents)
    assert packed.shape == (1, 3 * 2 * 5, 4)

    for index, (frame, row, column) in enumerate((f, h, w) for f in range(3) for h in range(2) for w in range(5)):
        assert torch.equal(packed[0, index], latents[0, :, frame, row, column])


def test_unpacking_video_latents_restores_the_original_tensor() -> None:
    latents = torch.randn(1, 8, 4, 3, 6)
    assert torch.equal(unpack_video_latents(pack_video_latents(latents), 4, 3, 6), latents)


def test_an_audio_row_is_one_latent_frame_with_mel_bins_inside_channels() -> None:
    latents = torch.randn(1, 3, 4, 2)
    packed = pack_audio_latents(latents)
    assert packed.shape == (1, 4, 6)
    for frame in range(4):
        assert torch.equal(packed[0, frame], latents[0, :, frame, :].reshape(-1))
    assert torch.equal(unpack_audio_latents(packed, mel_bins=2), latents)


@pytest.mark.parametrize(("num_frames", "expected"), [(1, 1), (9, 2), (17, 3), (121, 16), (241, 31)])
def test_the_latent_frame_count_follows_the_causal_vae_grouping(num_frames: int, expected: int) -> None:
    """The first frame is encoded alone and every further group of eight shares a latent frame."""
    assert latent_frame_count(num_frames) == expected


def test_the_sequence_length_is_the_latent_grid(num_frames: int = 121) -> None:
    assert video_sequence_length(num_frames, 704, 1248) == 16 * 22 * 39


@pytest.mark.parametrize(("num_frames", "fps", "expected"), [(121, 24.0, 126), (9, 24.0, 9), (25, 25.0, 25)])
def test_the_audio_latent_count_covers_the_clip_duration(num_frames: int, fps: float, expected: int) -> None:
    """25 audio latents a second, from the clip's duration rather than its frame count."""
    assert audio_latent_count(num_frames, fps) == expected


@pytest.mark.parametrize("num_frames", [0, 2, 8, 10, 120])
def test_a_frame_count_off_the_grid_is_refused_with_the_nearest_valid_one(num_frames: int) -> None:
    with pytest.raises(ValueError, match="8n \\+ 1"):
        validate_num_frames(num_frames)


@pytest.mark.parametrize(("requested", "expected"), [(1, 1), (3, 1), (5, 9), (8, 9), (100, 97), (122, 121)])
def test_snapping_a_frame_count_rounds_a_tie_up(requested: int, expected: int) -> None:
    """A 5-frame request is equidistant from 1 and 9; a single still frame is not what it meant."""
    assert snap_num_frames(requested) == expected
    validate_num_frames(snap_num_frames(requested))


@pytest.mark.parametrize("requested", [1, 8, 9, 10, 16, 17, 96, 97, 121, 481, 1000])
def test_a_conditioning_clip_never_claims_more_frames_than_it_supplied(requested: int) -> None:
    """Conditioning snaps DOWN where a request snaps to the nearest: a clip has the frames it has,
    and rounding up would pad picture (or silence) for the model to hold clean. Stated as the
    property rather than a table -- the largest valid count that does not exceed the input."""
    snapped = snap_num_frames_down(requested)

    validate_num_frames(snapped)
    assert snapped <= requested
    assert requested - snapped < LTX2_FRAME_MODULUS


def test_snapping_down_a_count_under_one_frame_group_leaves_a_single_frame() -> None:
    """The floor, which the conditioning nodes then reject as too short rather than generating it."""
    assert snap_num_frames_down(0) == 1
    assert snap_num_frames_down(1) == 1
    assert snap_num_frames_down(8) == 1


@pytest.mark.parametrize("size", [(704, 1250), (700, 1248), (0, 1248)])
def test_a_canvas_off_the_32_grid_is_refused(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        validate_canvas(*size)


@pytest.mark.parametrize(
    ("source", "short_edge", "expected"),
    [
        ((1920, 1080), 704, (704, 1248)),
        ((1080, 1920), 704, (1248, 704)),
        ((512, 512), 768, (768, 768)),
        ((4000, 3000), 512, (512, 672)),
    ],
)
def test_the_canvas_pins_the_short_edge_and_stays_on_the_grid(
    source: tuple[int, int], short_edge: int, expected: tuple[int, int]
) -> None:
    height, width = resolve_canvas(source[0], source[1], short_edge)
    assert (height, width) == expected
    validate_canvas(height, width)
    assert min(height, width) == short_edge


def test_a_transformer_that_patches_differently_is_refused_before_anything_is_packed() -> None:
    """Every packing here assumes 1x1x1 patching; a 2x2 checkpoint would silently mis-shape."""
    from types import SimpleNamespace

    require_patch_geometry(SimpleNamespace(patch_size=1, patch_size_t=1))
    with pytest.raises(ValueError, match="patches the latent grid"):
        require_patch_geometry(SimpleNamespace(patch_size=2, patch_size_t=1))


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [((16, 9), (1792, 1024)), ((1, 1), (1024, 1024)), ((9, 16), (1024, 1792))],
)
def test_a_two_stage_canvas_lands_where_halving_it_stays_on_the_grid(
    aspect: tuple[int, int], expected: tuple[int, int]
) -> None:
    """The 64 grid exists for one reason: the base pass runs at half the canvas, and half of a
    32-grid number is not always one."""
    height, width = resolve_canvas(aspect[0], aspect[1], 1024, multiple=64)

    assert (width, height) == expected
    base_height, base_width = base_canvas(height, width)
    assert (base_width * 2, base_height * 2) == (width, height)
    assert base_width % 32 == 0 and base_height % 32 == 0


def test_a_canvas_that_cannot_be_halved_onto_the_grid_is_refused() -> None:
    # 1248x704 is a legitimate single-stage canvas, and its *width* is what refuses: 1248 % 64 is
    # 32, so halving it gives 624, which is not on the 32 grid. (704 halves to 352, which is.)
    with pytest.raises(ValueError, match="multiple of 64"):
        base_canvas(704, 1248)


def test_a_grid_that_is_not_a_multiple_of_the_canvas_one_is_refused() -> None:
    with pytest.raises(ValueError, match="canvas grid"):
        resolve_canvas(16, 9, 1024, multiple=48)


def _load_upstream_keyframe_coords():
    """`_prepare_keyframe_coords` lifted out of the installed diffusers source.

    The module it lives in cannot be imported here: it pulls a Gemma-4 symbol from a transformers
    version this integration deliberately does not require (the loader builds the text tower alone
    for exactly that reason). Compiling the one function out of the real file keeps the comparison
    against upstream's actual code rather than against a copy of it that would drift in step.
    """
    import ast

    # Located by path, never imported: importing it is what fails.
    spec = importlib.util.find_spec("diffusers.pipelines.ltx2.pipeline_ltx2_condition")
    assert spec is not None and spec.origin is not None, "diffusers LTX-2 condition pipeline not installed"
    source = pathlib.Path(spec.origin).read_text()
    tree = ast.parse(source)
    node = next(
        (
            item
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "_prepare_keyframe_coords"
        ),
        None,
    )
    assert node is not None, (
        "diffusers no longer defines `_prepare_keyframe_coords` on a pipeline class. Our port in "
        "`prepare_keyframe_coords` was derived from it, so find where the convention moved and "
        "re-point this comparison rather than deleting it."
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"torch": torch}
    exec(compile(module, "<upstream>", "exec"), namespace)  # noqa: S102

    return namespace["_prepare_keyframe_coords"]


@pytest.mark.parametrize(
    ("latent_frames", "num_pixel_frames", "pixel_frame_index", "fps"),
    [(1, 1, 120, 24.0), (1, 1, 0, 24.0), (2, 9, 48, 30.0), (4, 25, 96, 16.0), (1, 1, 480, 60.0)],
)
def test_keyframe_coords_match_the_released_pipelines_own(
    latent_frames: int, num_pixel_frames: int, pixel_frame_index: int, fps: float
) -> None:
    """Our port against the implementation it is a port OF, rather than against itself.

    `_prepare_keyframe_coords` is a method on the diffusers pipeline, so it is called here unbound
    with a stub carrying only the four geometry attributes it reads. If diffusers changes the
    convention -- the causal fix, the single-frame clamp, the seconds conversion -- this fails
    instead of our keyframes quietly landing at the wrong instant.
    """
    upstream = _load_upstream_keyframe_coords()
    stub = SimpleNamespace(
        transformer_spatial_patch_size=LTX2_PATCH_SIZE,
        transformer_temporal_patch_size=LTX2_PATCH_SIZE_T,
        vae_spatial_compression_ratio=LTX2_SPATIAL_COMPRESSION,
        vae_temporal_compression_ratio=LTX2_TEMPORAL_COMPRESSION,
    )
    latent_height, latent_width = 704 // LTX2_SPATIAL_COMPRESSION, 1248 // LTX2_SPATIAL_COMPRESSION

    expected = upstream(
        stub,
        keyframe_latent_num_frames=latent_frames,
        keyframe_latent_height=latent_height,
        keyframe_latent_width=latent_width,
        pixel_frame_idx=pixel_frame_index,
        num_pixel_frames=num_pixel_frames,
        fps=fps,
        device=torch.device("cpu"),
    )
    ours = prepare_keyframe_coords(
        latent_frames,
        latent_height,
        latent_width,
        pixel_frame_index=pixel_frame_index,
        num_pixel_frames=num_pixel_frames,
        fps=fps,
    )

    assert ours.shape == expected.shape
    assert torch.equal(ours, expected)


def test_a_single_frame_keyframe_occupies_one_instant_not_the_group_it_lands_in() -> None:
    """The clamp the port carries: without it a still frame would claim the whole 8-frame span the
    VAE scale implies, and the model would read it as eight frames of held picture."""
    coords = prepare_keyframe_coords(1, 2, 2, pixel_frame_index=120, num_pixel_frames=1, fps=24.0)
    start, end = coords[0, 0, :, 0], coords[0, 0, :, 1]

    assert torch.allclose(start, torch.full_like(start, 120 / 24.0))
    assert torch.allclose(end, torch.full_like(end, 121 / 24.0))
