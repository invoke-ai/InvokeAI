"""Qwen-Image-2.1's RGBA VAE: how a decode becomes an image, what an encode reads -- a canvas image or an edit's
reference -- and the tile geometry it runs in."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image

from invokeai.backend.qwen_image_2_1 import vae as vae_module
from invokeai.backend.qwen_image_2_1.vae import (
    MIN_TILE_SIZE,
    SPATIAL_SCALE,
    SPEC,
    choose_tile_size,
    default_tile_size,
    estimate_working_memory,
    reference_size,
    to_image,
    to_reference,
    to_vae_input,
    untiled_peak_bytes,
)
from invokeai.backend.util import wddm
from invokeai.backend.util.qwen_image_vae import patch_qwen_image_vae_tiling
from invokeai.backend.working_memory import platform as platform_module
from invokeai.backend.working_memory.coefficients import shipped_data


def _decode(alpha: torch.Tensor) -> torch.Tensor:
    """A (4, H, W) decode in [-1, 1]: mid-grey colour and the given alpha in 0..255."""
    rgb = torch.zeros(3, *alpha.shape)
    return torch.cat([rgb, (alpha.float() / 127.5 - 1)[None]])


def test_an_opaque_decode_is_saved_as_rgb() -> None:
    image = to_image(_decode(torch.full((64, 64), 252)))
    assert image.mode == "RGB"


def test_tiling_seams_in_alpha_do_not_make_an_opaque_image_transparent() -> None:
    alpha = torch.full((100, 100), 252)
    alpha[0, :5] = 181  # 0.05 % of pixels, as 256px tiles leave them
    assert to_image(_decode(alpha)).mode == "RGB"


def test_a_transparent_background_is_kept() -> None:
    alpha = torch.full((64, 64), 255)
    alpha[:, :32] = 0
    image = to_image(_decode(alpha))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0 and image.getpixel((63, 0))[3] == 255


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_values_map_exactly_as_the_pipelines_postprocess(dtype: torch.dtype) -> None:
    from diffusers.image_processor import VaeImageProcessor

    generator = torch.Generator().manual_seed(0)
    decoded = (torch.rand(4, 64, 64, generator=generator) * 2.4 - 1.2).to(dtype)
    decoded[3] = 1.0
    expected = VaeImageProcessor(vae_scale_factor=16).postprocess(decoded[None], output_type="pil")[0]
    assert np.array_equal(np.asarray(to_image(decoded)), np.asarray(expected.convert("RGB")))


def test_an_encode_ignores_the_images_alpha() -> None:
    image = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    pixels = to_vae_input(image)
    assert pixels.shape == (1, 4, 32, 32)
    assert torch.equal(pixels[0, 3], torch.ones(32, 32))
    assert pixels[0, 0].unique().tolist() == [1.0] and pixels[0, 1].unique().tolist() == [-1.0]


@pytest.fixture
def vae() -> AutoencoderKLQwenImage21:
    # Tiling is geometry on the module; a minimal network is enough to hold it.
    return AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1], num_res_blocks=1, temperal_downsample=[False]
    )


def test_tiles_stride_on_the_16px_grid_and_the_modules_geometry_is_restored(vae: AutoencoderKLQwenImage21) -> None:
    before = (vae.use_tiling, vae.tile_sample_min_height, vae.tile_sample_stride_height)
    with patch_qwen_image_vae_tiling(vae, 1008, SPATIAL_SCALE):
        assert vae.use_tiling
        assert (vae.tile_sample_min_height, vae.tile_sample_min_width) == (1008, 1008)
        # 3/4 of 1008 is 756, which the 8x Qwen-Image grid would keep; at 16x it must drop to 752.
        assert (vae.tile_sample_stride_height, vae.tile_sample_stride_width) == (752, 752)
    assert (vae.use_tiling, vae.tile_sample_min_height, vae.tile_sample_stride_height) == before


def test_the_geometry_is_restored_when_the_decode_fails(vae: AutoencoderKLQwenImage21) -> None:
    before = (vae.use_tiling, vae.tile_sample_min_height)
    with pytest.raises(RuntimeError), patch_qwen_image_vae_tiling(vae, 512, SPATIAL_SCALE):
        raise RuntimeError("out of memory")
    assert (vae.use_tiling, vae.tile_sample_min_height) == before


CPU = torch.device("cpu")
GPU = torch.device("cuda")


def _reserved(operation: str, size: int, tile: int | None, element_size: int, device: torch.device) -> int:
    return estimate_working_memory(operation, size, size, tile, element_size, device).reservation_bytes


def _reserved_frame(height: int, width: int, tile: int | None) -> int:
    return estimate_working_memory("decode", height, width, tile, 2, GPU).reservation_bytes


def _tile_term(height: int, width: int, tile: int | None) -> float:
    """What a bf16 decode's convolutions term prices: the part the untiled line holds."""
    estimate = estimate_working_memory("decode", height, width, tile, 2, GPU)
    return next(term.bytes for term in estimate.terms if term.name == "convolutions")


GIB = 1024**3


def _pin(monkeypatch: pytest.MonkeyPatch, conv_backend: str, allocator: str = "native", os: str = "linux") -> None:
    """Pin the platform a `GPU` estimate prices: the convolution backend, and the allocator and OS rather than the
    host's allocator backend and variables."""
    monkeypatch.setattr(torch.version, "hip", "10.0.0" if conv_backend == "miopen" else None)
    monkeypatch.setattr(
        platform_module, "_allocator_for", lambda device: allocator if device.type == "cuda" else "native"
    )
    monkeypatch.setattr(platform_module, "_host_os", lambda: os)


@pytest.mark.parametrize(
    ("operation", "size", "element_size", "expected"),
    [
        # The bf16 decode covers the RTX 4090's 8.38 GiB at 1024px by the shipped margin, past which it is measured.
        ("decode", 1152, 2, 8396 * 1152 * 1152 + 615 * 2**20),
        # The pricing shipped before the working-memory data: 4000 bytes per pixel per element byte, plus 0.6 GiB (to a MiB).
        ("encode", 1024, 2, 8000 * 1024 * 1024 + 615 * 2**20),
        ("decode", 1536, 4, 16000 * 1536 * 1536 + 615 * 2**20),
        ("encode", 1536, 4, 16000 * 1536 * 1536 + 615 * 2**20),
    ],
)
def test_on_cuda_an_untiled_operation_is_priced_on_the_stand_in(
    operation: str, size: int, element_size: int, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "cudnn")
    assert _reserved(operation, size, None, element_size, GPU) == expected
    # With no point of this operation and dtype to read, the tiling decision prices the same figure.
    if (operation, element_size) != ("decode", 2):
        assert untiled_peak_bytes(operation, size, size, element_size, GPU) == expected
    # A tile covering the frame runs the single pass, and is priced as one.
    assert _reserved(operation, size, size, element_size, GPU) == expected


@pytest.mark.parametrize(
    ("height", "width", "tile_height"),
    # 1536^2 tiles, and 1152x1536 ones clamped to the frame: both past the step.
    [(2048, 2048, 1536), (1152, 2048, 1152)],
)
def test_on_miopen_a_tile_past_the_step_costs_what_the_frame_of_its_size_does(
    height: int, width: int, tile_height: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "miopen")
    assert _tile_term(height, width, 1536) == _tile_term(tile_height, 1536, None)
    # The W7900's untiled 1536px decode measured 32.6 GiB.
    if tile_height == 1536:
        assert _tile_term(height, width, 1536) >= 32.6 * GIB


def test_on_cuda_a_large_frame_is_not_split_off_the_decode_line() -> None:
    features = {"operation": "decode", "height": 1536, "width": 1536, "tile": None, "dtype": "bf16"}
    assert SPEC.line({**features, "conv_backend": "cudnn"}).suffix == "decode_bf16"
    assert SPEC.line({**features, "conv_backend": "miopen"}).suffix == "decode_large_bf16"


def test_on_rocm_a_decode_is_priced_above_the_original_and_an_encode_below_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    original = 1024 * 1024 * 4000 * 2 + 615 * 2**20
    assert _reserved("encode", 1024, None, 2, GPU) < original < _reserved("decode", 1024, None, 2, GPU)
    # The measured W7900 decode peaks 11.81 GiB at 1024px (12.68 GB reserved): the reservation covers it.
    assert _reserved("decode", 1024, None, 2, GPU) >= _ROCM_DECODE_1024
    # A VAE on the CPU of a ROCm build is not on MIOpen.
    assert _reserved("decode", 1024, None, 2, CPU) == 8396 * 1024 * 1024 + 615 * 2**20


def test_a_frame_is_tiled_when_either_side_exceeds_the_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    # A tile covering the frame runs the single pass; a side past the tile prices one tile.
    assert _reserved("decode", 1024, 1024, 2, GPU) == _reserved("decode", 1024, None, 2, GPU)
    tiled = estimate_working_memory("decode", 1024, 1536, 1024, 2, GPU).reservation_bytes
    assert tiled < estimate_working_memory("decode", 1024, 1536, None, 2, GPU).reservation_bytes


def test_a_tile_is_clamped_to_the_frame_per_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    # A 512x1536 frame in 1024px tiles runs tiles of 512x1024, and a 1536x512 one tiles of 1024x512.
    assert _tile_term(512, 1536, 1024) == _tile_term(512, 1024, None)
    assert _tile_term(1536, 512, 1024) == _tile_term(1024, 512, None)
    # What tiling adds grows with the tile on MIOpen, so it is the same for both.
    assert _reserved_frame(1536, 512, 1024) == _reserved_frame(512, 1536, 1024)


def test_a_tile_prices_below_the_frame_it_covers_in_part(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    for tile in (512, 768, 1024):
        assert _reserved("decode", 2048, tile, 2, GPU) < _reserved("decode", 2048, None, 2, GPU)


@pytest.mark.parametrize(
    ("conv_backend", "allocator", "os", "frame", "tile", "measured_gib"),
    [
        ("cudnn", "native", "linux", 2048, 1024, 8.38),
        ("cudnn", "native", "linux", 2048, 768, 5.27),
        ("cudnn", "native", "linux", 2048, 512, 2.68),
        ("miopen", "native", "linux", 3072, 512, 4.25),
        ("miopen", "native", "windows", 3072, 512, 4.25),
        ("miopen", "native", "windows", 1536, 512, 4.24),
        ("miopen", "expandable", "windows", 2048, 512, 2.88),
    ],
)
def test_a_measured_tile_is_reserved_with_the_shipped_margin(
    conv_backend: str,
    allocator: str,
    os: str,
    frame: int,
    tile: int,
    measured_gib: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, conv_backend, allocator, os)
    assert _reserved("decode", frame, tile, 2, GPU) >= 1.05 * measured_gib * GIB


def test_on_miopen_a_bf16_decode_past_the_step_prices_on_its_own_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")

    def per_pixel(height: int, width: int) -> float:
        return _tile_term(height, width, None) / (height * width)

    # Measured: 1248^2 12510 bytes per pixel, 1024x1536 (1.573 Mpx) 12067, 1264^2 12887, 1280^2 14382. The real jump is
    # between 1264^2 and 1280^2, and the step leaves the first two on the low line.
    low, large = per_pixel(1024, 1024), per_pixel(1600, 1600)
    assert large > low
    assert per_pixel(1248, 1248) == per_pixel(1024, 1536) == low
    # Exactly at the step is still below it.
    assert per_pixel(1000, 1580) == low
    assert per_pixel(1264, 1264) == per_pixel(1280, 1280) == large
    # 1024x1536 peaks at 17.68 GiB: a 20 GiB card can reserve it.
    assert (
        1.05 * 17.68 * GIB <= estimate_working_memory("decode", 1024, 1536, None, 2, GPU).reservation_bytes < 20 * GIB
    )
    # A 1024px bf16 decode measures 11.81 GiB on the W7900: its reservation stays close to that.
    assert 1.05 * 11.81 * GIB <= _reserved("decode", 1024, None, 2, GPU) <= 1.1 * 11.81 * GIB


def test_only_a_known_element_size_is_priced() -> None:
    with pytest.raises(ValueError, match="1-byte elements"):
        estimate_working_memory("decode", 1024, 1024, None, 1, CPU)
    with pytest.raises(ValueError, match="8-byte elements"):
        untiled_peak_bytes("decode", 1024, 1024, 8, CPU)


# W7900 bf16 untiled peaks, as measured (reserved bytes).
_ROCM_DECODE_1024 = 12683575296
_ROCM_DECODE_1088 = 14847836160
_ROCM_ENCODE_1536 = 11362369536


def test_on_rocm_the_tiling_decision_follows_the_measured_untiled_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    # Measured points are their measured bytes; between two, the chord in bytes.
    assert untiled_peak_bytes("decode", 1024, 1024, 2, GPU) == _ROCM_DECODE_1024
    between = (1024 * 1088 - 1024 * 1024) / (1088 * 1088 - 1024 * 1024)
    expected = _ROCM_DECODE_1024 + (_ROCM_DECODE_1088 - _ROCM_DECODE_1024) * between
    assert untiled_peak_bytes("decode", 1024, 1088, 2, GPU) == pytest.approx(expected, abs=1)
    # Past the last measured area (1600px) it is that point plus what the reservation grows by from it.
    last = untiled_peak_bytes("decode", 1600, 1600, 2, GPU)
    grown = _reserved("decode", 2048, None, 2, GPU) - _reserved("decode", 1600, None, 2, GPU)
    assert untiled_peak_bytes("decode", 2048, 2048, 2, GPU) == last + grown

    # On a card that holds 12.5 GiB a measured 11.81 GiB decode stays whole, though it reserves 12.86 GiB; and an encode
    # reads its own curve (10.58 GiB at 1536px), not the decode's (32.6 GiB).
    monkeypatch.setattr(vae_module, "should_pretile_vae_decode", lambda device, needed: needed > 12.5 * GIB)
    assert _reserved("decode", 1024, None, 2, GPU) > 12.5 * GIB
    assert choose_tile_size("decode", 1024, 1024, 2, GPU, tiled=False, tile_size=768, auto_tile=True) is None
    assert untiled_peak_bytes("encode", 1536, 1536, 2, GPU) == _ROCM_ENCODE_1536
    assert choose_tile_size("encode", 1536, 1536, 2, GPU, tiled=False, tile_size=768, auto_tile=True) is None
    assert choose_tile_size("decode", 1536, 1536, 2, GPU, tiled=False, tile_size=768, auto_tile=True) == 768


def test_nothing_is_tiled_unless_asked_or_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stand in for the device: it holds 12 GiB of working memory.
    monkeypatch.setattr(vae_module, "should_pretile_vae_decode", lambda device, needed: needed > 12 * GIB)

    def choose(size: int, **kwargs) -> int | None:
        defaults = {"tiled": False, "tile_size": 0, "auto_tile": True}
        return choose_tile_size("decode", size, size, 2, CPU, **(defaults | kwargs))

    assert choose(1024) is None
    # 2048x2048 untiled needs ~32 GiB: tiled, at the device's tile size.
    assert choose(2048) == 1024
    assert choose(2048, auto_tile=False) is None
    assert choose(1024, tiled=True) == 1024


def test_the_default_tile_is_the_largest_whose_working_memory_fits_half_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, "miopen")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=42 * GIB))
    # On a 42 GiB MIOpen card a 1024px fp32 decode tile reserves 23.32 GiB, over half of it -- priced as a tile: as an
    # untiled 1024px frame it would be 19.75 GiB, and fit. A 1024px encode tile (6.64 GiB) fits.
    assert default_tile_size("decode", GPU, 4) == 768
    assert default_tile_size("encode", GPU, 4) == 1024


def test_an_encode_takes_its_own_default_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen", "expandable", "windows")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=16 * GIB))
    # On a 16 GiB card with expandable segments a 1024px bf16 encode tile fits in half of it; a decode tile does not.
    choose = {"tiled": True, "tile_size": 0, "auto_tile": False}
    assert choose_tile_size("encode", 2048, 2048, 2, GPU, **choose) == 1024
    assert choose_tile_size("decode", 2048, 2048, 2, GPU, **choose) == 768


@pytest.mark.parametrize(
    ("conv_backend", "allocator", "os"),
    [
        ("cudnn", "native", "linux"),
        ("miopen", "native", "linux"),
        ("miopen", "expandable", "windows"),
        ("miopen", "native", "windows"),
    ],
)
@pytest.mark.parametrize(("tile", "below"), [(1024, 768), (768, 512)])
def test_the_default_bf16_decode_tile_changes_exactly_at_each_memory_tier(
    conv_backend: str, allocator: str, os: str, tile: int, below: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, conv_backend, allocator, os)
    # A tile is picked once its reservation, priced as one tile of a frame twice its side, fits half the card.
    threshold = 2 * _reserved("decode", 2 * tile, tile, 2, GPU)
    for total, expected in ((threshold, tile), (threshold - 1, below)):
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda device, total=total: SimpleNamespace(total_memory=total)
        )
        assert default_tile_size("decode", GPU, 2) == expected


@pytest.mark.parametrize(
    ("allocator", "os", "decode_1024_gib", "encode_1536_gib"),
    [
        ("native", "linux", 11.81, 10.58),
        # The RX 9060 XT's own points, under each allocator.
        ("native", "windows", 14.06, 10.58),
        ("expandable", "windows", 10.49, 8.07),
        # Off Windows expandable segments are priced as the native allocator.
        ("expandable", "linux", 11.81, 10.58),
    ],
)
def test_the_tiling_decision_reads_the_points_of_the_platform_it_is_priced_as(
    allocator: str, os: str, decode_1024_gib: float, encode_1536_gib: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "miopen", allocator, os)
    assert untiled_peak_bytes("decode", 1024, 1024, 2, GPU) == pytest.approx(decode_1024_gib * GIB, rel=0.002)
    assert untiled_peak_bytes("encode", 1536, 1536, 2, GPU) == pytest.approx(encode_1536_gib * GIB, rel=0.002)


def test_below_its_first_point_the_tiling_decision_takes_the_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    # The W7900's first untiled fp32 decode is at 1024px; scaled down to 512px through the origin it would drop the
    # line's base.
    assert untiled_peak_bytes("decode", 512, 512, 4, GPU) == _reserved("decode", 512, None, 4, GPU)


@pytest.mark.parametrize(
    ("conv_backend", "allocator", "os", "below", "first"),
    # Just below the first untiled bf16 decode the reservation lies above that point's bytes: 1008px against the RTX
    # 4090's 1024px, 496x512 against the RX 9060 XT's 512px.
    [("cudnn", "native", "linux", (1008, 1008), 1024), ("miopen", "native", "windows", (496, 512), 512)],
)
def test_the_tiling_decision_never_falls_as_a_frame_grows_into_the_first_point(
    conv_backend: str, allocator: str, os: str, below: tuple[int, int], first: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, conv_backend, allocator, os)
    at_point = untiled_peak_bytes("decode", first, first, 2, GPU)
    assert estimate_working_memory("decode", *below, None, 2, GPU).reservation_bytes > at_point
    assert untiled_peak_bytes("decode", *below, 2, GPU) == at_point


def test_one_point_decides_the_tiling_decision_from_its_size_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen", "expandable", "windows")
    # One untiled fp32 decode is measured on this platform: 9.16 GiB at 768px. Past it, the point plus the reservation's
    # growth from it.
    at_point = untiled_peak_bytes("decode", 768, 768, 4, GPU)
    assert at_point == pytest.approx(9.16 * GIB, rel=0.002)
    grown = _reserved("decode", 1024, None, 4, GPU) - _reserved("decode", 768, None, 4, GPU)
    assert untiled_peak_bytes("decode", 1024, 1024, 4, GPU) == at_point + grown


@pytest.mark.parametrize(
    ("size", "total_gib", "tiled"),
    [
        # 8.38 GiB measured at 1024px; 1152px peaks near 10.56 GiB, under the 10.8 GiB ceiling of a 12 GiB card,
        # though it reserves 10.98.
        (1152, 12, False),
        # Below the point, the reservation: 7.57 GiB, over the 7.2 GiB ceiling (8.38 scaled through the origin is 7.12).
        (944, 8, True),
    ],
)
def test_on_nvidia_the_tiling_decision_grows_from_the_measured_point(
    size: int, total_gib: int, tiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "cudnn")
    monkeypatch.setattr(wddm, "local_video_memory", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=total_gib * GIB)
    )
    chosen = choose_tile_size("decode", size, size, 2, GPU, tiled=False, tile_size=0, auto_tile=True)
    assert (chosen is not None) == tiled


@pytest.mark.parametrize(("total_gib", "expected"), [(9.5, None), (9.3, 512)])
def test_on_nvidia_a_1024px_decode_tiles_on_its_measured_bytes(
    total_gib: float, expected: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "cudnn")
    monkeypatch.setattr(wddm, "local_video_memory", lambda device: None)
    total = int(total_gib * GIB)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=total))
    # The tiling ceiling is 90% of the card: 9.18e9 bytes of 9.5 GiB, 8.99e9 of 9.3 GiB, around the 8997956485 measured.
    # The 8.80 GiB (9449M) reserved would tile on both.
    assert choose_tile_size("decode", 1024, 1024, 2, GPU, tiled=False, tile_size=0, auto_tile=True) == expected


def test_the_tiling_decision_reads_the_larger_of_two_shapes_of_one_area(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    expected = untiled_peak_bytes("decode", 1024, 1024, 2, GPU)
    data = dict(shipped_data())
    file = data["qwen_image_2_1_vae.json"]
    measured = next(
        point
        for point in file.points
        if point.platform == {"conv_backend": "miopen", "allocator": "native"}
        and point.features["operation"] == "decode"
        and point.features["dtype"] == "bf16"
        and point.features["height"] == point.features["width"] == 1024
    )
    # A 512x2048 decode, the same area as 1024x1024, measured after it at twice its bytes.
    wider = measured.model_copy(
        update={
            "features": {**measured.features, "height": 512, "width": 2048},
            "measured_bytes": 2 * measured.measured_bytes,
        }
    )
    data["qwen_image_2_1_vae.json"] = file.model_copy(update={"points": [*file.points, wider]})
    monkeypatch.setattr(vae_module, "shipped_data", lambda: data)
    vae_module._measured_untiled_peaks.cache_clear()
    try:
        assert untiled_peak_bytes("decode", 1024, 1024, 2, GPU) == 2 * expected
    finally:
        vae_module._measured_untiled_peaks.cache_clear()


@pytest.mark.parametrize(
    "update",
    [{"in_fit": False}, {"method": "probe_demand"}, {"version": 2}],
    ids=["out_of_fit", "lower_bound_only", "other_version"],
)
def test_the_tiling_decision_ignores_points_that_cap_no_line(update: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    expected = untiled_peak_bytes("decode", 1024, 1024, 2, GPU)
    data = dict(shipped_data())
    file = data["qwen_image_2_1_vae.json"]
    measured = next(
        point
        for point in file.points
        if point.platform == {"conv_backend": "miopen", "allocator": "native"}
        and point.features["operation"] == "decode"
        and point.features["dtype"] == "bf16"
        and point.features["height"] == point.features["width"] == 1024
    )
    extra = measured.model_copy(update={"measured_bytes": 2 * measured.measured_bytes, **update})
    data["qwen_image_2_1_vae.json"] = file.model_copy(update={"points": [*file.points, extra]})
    monkeypatch.setattr(vae_module, "shipped_data", lambda: data)
    vae_module._measured_untiled_peaks.cache_clear()
    try:
        assert untiled_peak_bytes("decode", 1024, 1024, 2, GPU) == expected
    finally:
        vae_module._measured_untiled_peaks.cache_clear()


def test_the_shipped_data_agrees_with_itself() -> None:
    """A stand-in entry copied from another goes stale silently when its source is refitted; and a point's estimate
    reads its backend from its features, which must name the platform it was measured on."""
    entries = {
        tuple(sorted(entry.platform.items())): entry.values
        for data in shipped_data().values()
        for entry in data.coefficients
        if entry.estimator == "vae.qwen_image_2_1"
    }
    copies = {
        (("allocator", "cudaMallocAsync"), ("conv_backend", "cudnn")): (
            ("allocator", "native"),
            ("conv_backend", "cudnn"),
        ),
    }
    for copy, source in copies.items():
        assert entries[copy] == entries[source], copy
    for data in shipped_data().values():
        for point in data.points:
            if point.estimator == "vae.qwen_image_2_1":
                assert point.features["conv_backend"] == point.platform["conv_backend"], point.provenance


# What the RX 9060 XT measured against the W7900 at the same features: the scale a Windows stand-in takes the W7900's
# line at, by allocator and line. Bytes as measured, so a stand-in solved back from a bound rather than grounded shows.
_RX9060XT_OVER_W7900 = {
    ("native", "decode_large_bf16"): 15099494400 / 12683575296,  # the 1024px bf16 decode
    ("native", "decode_fp32"): 12463374336 / 13457424384,  # the 2048px fp32 decode at a 768px tile
    ("expandable", "decode_fp32"): 10634657792 / 13457424384,
    ("native", "tiled_encode_fp32"): 6784286720 / 6784286720,  # the 2048px fp32 encode at a 1024px tile
    ("expandable", "tiled_encode_fp32"): 5391777792 / 6784286720,
}


def test_a_stand_in_on_an_amd_entry_is_grounded_and_covers_its_own_points() -> None:
    """A stand-in on Windows stands for what the RX 9060 XT did not measure: the W7900's line at the ratio the two
    measured at the same features, or, where expandable segments have no decode step, the entry's own decode line. Either
    way it covers the entry's own points on that line by the shipped margin, as `build_shipped` raises it to."""
    entries = [
        entry
        for data in shipped_data().values()
        for entry in data.coefficients
        if entry.estimator == "vae.qwen_image_2_1" and entry.platform["conv_backend"] == "miopen"
    ]
    points = [point for data in shipped_data().values() for point in data.points if point.estimator == SPEC.id]
    w7900 = next(entry.values for entry in entries if "os" not in entry.platform)
    checked = 0
    for entry in (entry for entry in entries if "os" in entry.platform):
        allocator = entry.platform["allocator"]
        for line in {name[2:] for name in entry.stand_ins}:
            k, b = entry.values[f"k_{line}"], entry.values[f"b_{line}"]
            if (allocator, line) == ("expandable", "decode_large_bf16"):
                floor = (entry.values["k_decode_bf16"], entry.values["b_decode_bf16"])
            else:
                scale = _RX9060XT_OVER_W7900[(allocator, line)]
                floor = (scale * w7900[f"k_{line}"], scale * w7900[f"b_{line}"])
            assert k >= floor[0] and b >= floor[1], (entry.platform, line)
            for point in points:
                placed = SPEC.line(point.features)
                if point.platform == entry.platform and placed.suffix == line and placed.sits_on is None:
                    assert k * placed.size + b >= 1.05 * point.measured_bytes, (entry.platform, placed.label)
            checked += 1
    assert checked == len(_RX9060XT_OVER_W7900) + 1, "a stand-in nobody grounded, or the check is broken"


def test_a_tiled_stand_in_on_cudnn_prices_at_least_two_rgba_copies_of_the_frame_and_a_tile_margin() -> None:
    """Nothing else holds a tiled stand-in no point measures. On cuDNN the tiled line is over the frame's pixels: the
    slope is two RGBA copies of the image, 8 elements per pixel; the base at least the margin at a 768px tile, 266 MiB in
    bf16, scaled by the element size. (MIOpen's tiled stand-ins, over the tile's pixels, are the W7900's lines scaled.)"""
    elements = {"bf16": 2, "fp32": 4}
    checked = 0
    for data in shipped_data().values():
        for entry in data.coefficients:
            if entry.estimator != "vae.qwen_image_2_1" or entry.platform["conv_backend"] != "cudnn":
                continue
            for line in {name[2:] for name in entry.stand_ins if "_tiled_" in name}:
                element = elements[line.rsplit("_", 1)[1]]
                assert entry.values[f"k_{line}"] >= 8 * element, (entry.platform, line)
                assert entry.values[f"b_{line}"] >= 133 * element * 2**20, (entry.platform, line)
                checked += 1
    assert checked, "the check is broken"


@pytest.mark.parametrize(("dedicated", "expected"), [(True, 512), (False, 1024)], ids=["discrete", "integrated"])
def test_a_discrete_xpu_picks_its_tile_from_its_memory(dedicated: bool, expected: int, monkeypatch) -> None:
    monkeypatch.setattr(vae_module, "device_has_dedicated_vram", lambda device: dedicated)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: SimpleNamespace(total_memory=8 * GIB))
    # Priced as cuDNN: on 8 GiB only a 512px bf16 decode tile (2.93 GiB) fits in half. An integrated XPU shares system
    # RAM and keeps the largest tile.
    assert default_tile_size("decode", torch.device("xpu"), 2) == expected


def test_a_tile_size_field_below_the_floor_is_raised_to_it() -> None:
    assert choose_tile_size("decode", 2048, 2048, 2, CPU, tiled=True, tile_size=16, auto_tile=False) == MIN_TILE_SIZE
    assert choose_tile_size("decode", 2048, 2048, 2, CPU, tiled=True, tile_size=768, auto_tile=False) == 768


@pytest.mark.parametrize(
    ("precision", "expected"),
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)],
)
def test_the_pipelines_vae_loads_at_the_configured_precision_but_never_float16(
    precision: torch.dtype, expected: torch.dtype, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invokeai.backend.model_manager.load.model_loaders.qwen_image_2_1 import QwenImage21DiffusersModel
    from invokeai.backend.model_manager.taxonomy import SubModelType

    loaded: dict[str, torch.dtype] = {}

    class _VAE:
        @staticmethod
        def from_pretrained(_path, *, torch_dtype, **_kwargs):
            loaded["dtype"] = torch_dtype
            return torch.nn.Identity()

    loader = object.__new__(QwenImage21DiffusersModel)
    loader._torch_dtype = precision
    loader._torch_device = torch.device("cpu")
    monkeypatch.setattr(loader, "get_hf_load_class", lambda *_args: _VAE)

    loader._load_model(SimpleNamespace(path=str(tmp_path)), SubModelType.VAE)

    assert loaded["dtype"] is expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        # sqrt(1024^2 * 5/3) = 1322.0 -> 41.3 x 32 -> 1312; 1322.0 / (5/3) = 793.2 -> 24.8 x 32 -> 800.
        ((500, 300), (1312, 800)),
        ((1024, 1024), (1024, 1024)),
        # sqrt(1024^2 * 3) = 1773.6 -> 1760; 591.2 -> 576.
        ((3000, 1000), (1760, 576)),
    ],
)
def test_a_reference_is_read_at_one_megapixel_on_the_32px_grid(
    size: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert reference_size(*size) == expected


def test_a_reference_encodes_exactly_as_the_pipeline_preprocesses_it() -> None:
    from diffusers.image_processor import VaeImageProcessor

    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (300, 500, 4), generator=generator, dtype=torch.uint8).numpy()
    image = Image.fromarray(pixels, mode="RGBA")
    width, height = reference_size(*image.size)
    expected = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).preprocess(
        image, width=width, height=height
    )

    # The alpha is the reference's own, unlike a canvas encode.
    assert torch.equal(to_vae_input(to_reference(image), keep_alpha=True), expected.float())
