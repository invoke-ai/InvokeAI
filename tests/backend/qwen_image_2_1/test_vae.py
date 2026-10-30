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
    choose_tile_size,
    default_tile_size,
    estimate_working_memory,
    reference_size,
    to_image,
    to_reference,
    to_vae_input,
    untiled_peak_bytes,
)
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


GIB = 1024**3


def _pin(monkeypatch: pytest.MonkeyPatch, conv_backend: str, allocator: str = "native", os: str = "linux") -> None:
    """Pin the platform a `GPU` estimate prices: the convolution backend, and the allocator and OS rather than the
    host's allocator backend and variables."""
    monkeypatch.setattr(torch.version, "hip", "10.0.0" if conv_backend == "miopen" else None)
    monkeypatch.setattr(
        platform_module, "_allocator_for", lambda device: allocator if device.type == "cuda" else "native"
    )
    monkeypatch.setattr(platform_module, "_host_os", lambda: os)


@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_on_cuda_both_operations_keep_the_original_pricing(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # The pricing shipped before the working-memory data: 4000 bytes per pixel per element byte, plus 0.6 GiB (to a MiB).
    _pin(monkeypatch, "cudnn")
    for size, element_size in [(1024, 2), (1536, 4)]:
        original = size * size * 4000 * element_size + 615 * 2**20
        assert _reserved(operation, size, None, element_size, GPU) == original
        # The tiling decision prices the same figure there: CUDA tiles exactly the frames it tiled before.
        assert untiled_peak_bytes(operation, size, size, element_size, GPU) == original
    # A tile covering the frame runs the single pass, and is priced as one.
    assert _reserved(operation, 1024, 1024, 2, GPU) == _reserved(operation, 1024, None, 2, GPU)


def test_on_rocm_a_decode_is_priced_above_the_original_and_an_encode_below_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    original = 1024 * 1024 * 4000 * 2 + 615 * 2**20
    assert _reserved("encode", 1024, None, 2, GPU) < original < _reserved("decode", 1024, None, 2, GPU)
    # The measured W7900 decode peaks 11.81 GiB at 1024px (12.68 GB reserved): the reservation covers it.
    assert _reserved("decode", 1024, None, 2, GPU) >= _ROCM_DECODE_1024
    # A VAE on the CPU of a ROCm build is not on MIOpen.
    assert _reserved("decode", 1024, None, 2, CPU) == original


def test_a_frame_is_tiled_when_either_side_exceeds_the_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    # A tile covering the frame runs the single pass; a side past the tile prices one tile, whatever the frame.
    assert _reserved("decode", 1024, 1024, 2, GPU) == _reserved("decode", 1024, None, 2, GPU)
    tiled = estimate_working_memory("decode", 1024, 1536, 1024, 2, GPU).reservation_bytes
    assert tiled < estimate_working_memory("decode", 1024, 1536, None, 2, GPU).reservation_bytes
    assert tiled == _reserved("decode", 4096, 1024, 2, GPU)


@pytest.mark.parametrize("tile", [512, 768, 1024])
def test_a_tile_is_priced_per_tile_not_per_frame(tile: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "miopen")
    assert _reserved("decode", 2048, tile, 2, GPU) == _reserved("decode", 3072, tile, 2, GPU)
    assert _reserved("decode", 2048, tile, 2, GPU) < _reserved("decode", 2048, None, 2, GPU)


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
    # Past the last measured area it is the reservation.
    assert untiled_peak_bytes("decode", 2048, 2048, 2, GPU) == _reserved("decode", 2048, None, 2, GPU)

    # On a card that holds 12.5 GiB a measured 11.81 GiB decode stays whole, though it reserves ~15 GiB; and an encode
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
    # On a 42 GiB MIOpen card a 1024px fp32 decode tile reserves ~23.06 GiB, over half of it -- priced as a tile: as an
    # untiled 1024px frame it would be 19.75 GiB, and fit. A 1024px encode tile (~7.05 GiB) fits.
    assert default_tile_size("decode", GPU, 4) == 768
    assert default_tile_size("encode", GPU, 4) == 1024


def test_on_windows_expandable_segments_the_tiling_decision_follows_the_native_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Invoke's Windows default on AMD carries the native entry's values as stand-ins, and no points of its own.
    _pin(monkeypatch, "miopen", allocator="expandable", os="windows")
    assert untiled_peak_bytes("decode", 1024, 1024, 2, GPU) == _ROCM_DECODE_1024
    assert untiled_peak_bytes("encode", 1536, 1536, 2, GPU) == _ROCM_ENCODE_1536


@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("element_size", [2, 4])
def test_on_cuda_a_tile_keeps_the_original_pricing_widened_to_the_measured_tiles(
    operation: str, element_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, "cudnn")
    for tile in (448, 512, 768, 1024):
        # The original 4000 bytes per pixel per element byte plus 0.6 GiB (to a MiB), times 1.065: what the RTX 4090's
        # 512px tile, measured at 2.68 GiB, needs.
        expected = tile * tile * 4000 * element_size * 1.065 + 615 * 2**20
        assert _reserved(operation, 2048, tile, element_size, GPU) == pytest.approx(expected, abs=1)


def test_on_a_24_gib_cuda_card_the_default_tile_is_the_one_the_original_pricing_chose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, "cudnn")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=24 * GIB))

    def original(element_size: int) -> int:
        fits = (s for s in (1024, 768, 512) if s * s * 4000 * element_size + int(0.6 * GIB) <= 12 * GIB)
        return next(fits, 512)

    assert default_tile_size("decode", GPU, 2) == original(2) == 1024
    assert default_tile_size("encode", GPU, 4) == original(4) == 768


def test_on_rocm_a_tile_below_the_smallest_measured_one_is_priced_as_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # Per tile pixel, MIOpen's smaller tiles cost more: 17.4 KB at 512px, ~13.6 KB at 768px. Unmeasured below 512px.
    _pin(monkeypatch, "miopen")
    assert _reserved("decode", 2048, 448, 2, GPU) == _reserved("decode", 2048, 512, 2, GPU)
    assert _reserved("decode", 2048, 768, 2, GPU) > _reserved("decode", 2048, 512, 2, GPU)


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
        (("allocator", "expandable"), ("conv_backend", "miopen"), ("os", "windows")): (
            ("allocator", "native"),
            ("conv_backend", "miopen"),
        ),
    }
    for copy, source in copies.items():
        assert entries[copy] == entries[source], copy
    for data in shipped_data().values():
        for point in data.points:
            if point.estimator == "vae.qwen_image_2_1":
                assert point.features["conv_backend"] == point.platform["conv_backend"], point.provenance


@pytest.mark.parametrize(("dedicated", "expected"), [(True, 512), (False, 1024)], ids=["discrete", "integrated"])
def test_a_discrete_xpu_picks_its_tile_from_its_memory(dedicated: bool, expected: int, monkeypatch) -> None:
    monkeypatch.setattr(vae_module, "device_has_dedicated_vram", lambda device: dedicated)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: SimpleNamespace(total_memory=8 * GIB))
    # Priced as cuDNN: on 8 GiB only a 512px bf16 decode tile (~2.7 GiB) fits in half. An integrated XPU shares system
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
