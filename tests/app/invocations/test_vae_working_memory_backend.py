"""Which convolution line the diffusers-AutoencoderKL estimators price, and what the SD estimator adds up.

SD1/SDXL, SD3, FLUX.1 and FLUX.2 share measured lines per operation, dtype and platform (convolution backend,
allocator, and on MIOpen's native allocator the OS); CogView4 carries its own. The selection is invisible on whichever
machine runs the suite -- on the RX 9060 XT an fp16 SDXL decode costs a third of a bf16 one, and the SD nodes run fp16
whatever dtype the cached VAE holds -- so it is pinned here rather than left to the card.
"""

import itertools
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.vae.latents_to_image import LatentsToImageInvocation
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType
from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import (
    VAE_COGVIEW4_SPEC,
    VAE_FLUX2_SPEC,
    VAE_FLUX_SPEC,
    VAE_SD3_SPEC,
    VAE_SD_SPEC,
    estimate_untiled_vae_decode,
    estimate_vae_working_memory_cogview4,
    estimate_vae_working_memory_flux,
    estimate_vae_working_memory_flux2,
    estimate_vae_working_memory_sd3,
    estimate_vae_working_memory_sd15_sdxl,
    sd_vae_run_dtype,
)
from invokeai.backend.working_memory import platform as platform_module
from invokeai.backend.working_memory.bounds import evaluate_point, point_id
from invokeai.backend.working_memory.coefficients import ResolvedCoefficients, shipped_coefficients, shipped_data
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.platform import Platform, priced_as

PRE_DTYPE_BYTES_PER_PIXEL_PER_ELEMENT_BYTE = {
    "cudnn": {"decode": 2200, "encode": 1100},
    "miopen": {"decode": 3600, "encode": 2750},
}
"""What these estimators reserved before lines per dtype, per convolution backend, with no base: the floor no stand-in
and nothing unmeasured may price a slope below. Written out rather than read from the data, so a lowered floor fails
here. It covers the convolution lines; the one place v2 can reserve less than v1 there is the SD fp32 upcast, which
v1 guessed at 250 MiB and v2 prices as what the cast adds (160 MiB from fp16, nothing from fp32)."""
ELEMENT_SIZES = {"fp16": 2, "bf16": 2, "fp32": 4}
MIB = 2**20

MEASURED = [
    Platform("cudnn", "native", "linux"),
    Platform("cudnn", "cudaMallocAsync", "linux"),
    Platform("miopen", "native", "linux"),
    Platform("miopen", "native", "windows"),
    Platform("miopen", "expandable", "windows"),
]


def _vae(cls=AutoencoderKL, dtype: torch.dtype = torch.float16, numel: int = 1, params=None) -> MagicMock:
    vae = MagicMock(spec=cls)
    vae.parameters.side_effect = lambda: iter(params if params is not None else [torch.zeros(numel, dtype=dtype)])
    vae.config.latent_channels = 16
    # The stock geometry a tiled estimate snaps its tile with, as the tiling helpers do: an 8x downsample and the
    # overlap the VAE carries.
    vae.config.block_out_channels = (128, 256, 512, 512)
    vae.tile_overlap_factor = 0.25
    return vae


@pytest.fixture
def on(monkeypatch):
    """Run the estimate as on `platform`: the torch build, the allocator a CUDA device reports, and the OS."""

    def pin(platform: Platform) -> torch.device:
        monkeypatch.setattr(torch.version, "hip", "7.2.0" if platform.conv_backend == "miopen" else None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_allocator_backend",
            lambda: "cudaMallocAsync" if platform.allocator == "cudaMallocAsync" else "native",
        )
        if platform.allocator == "expandable":
            monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        monkeypatch.setattr(platform_module, "_host_os", lambda: platform.os)
        return torch.device("cuda", 0)

    # The mid-block score matrix is the other term in these estimates and has its own tests; hold it at zero so the
    # assertions below are about the convolution line alone.
    with (
        patch("invokeai.backend.util.vae_working_memory._vae_mid_block_score_matrix_bytes", return_value=0),
        patch("invokeai.backend.util.vae_working_memory.sdpa_score_matrix_bytes", return_value=0),
    ):
        yield pin


def _line(spec, platform: Platform, operation: str, dtype: str, pixels: float) -> float:
    values = shipped_coefficients(spec, platform).values
    return values[f"k_{operation}_{dtype}"] * pixels + values[f"b_{operation}_{dtype}"]


LATENTS_1024 = torch.zeros(1, 16, 128, 128)


@pytest.mark.parametrize("platform", MEASURED, ids=str)
@pytest.mark.parametrize(
    "spec, estimate, vae_dtype, run_dtype",
    [
        # The SD nodes cast the VAE to fp16 for every GPU run that does not ask for fp32: bf16 never runs.
        pytest.param(
            VAE_SD_SPEC,
            lambda device, vae: estimate_vae_working_memory_sd15_sdxl("decode", LATENTS_1024, vae, None, False, device),
            torch.bfloat16,
            "fp16",
            id="sd",
        ),
        pytest.param(
            VAE_SD3_SPEC,
            lambda device, vae: estimate_vae_working_memory_sd3("decode", LATENTS_1024, vae, device),
            torch.float32,
            "fp32",
            id="sd3",
        ),
        pytest.param(
            VAE_COGVIEW4_SPEC,
            lambda device, vae: estimate_vae_working_memory_cogview4("decode", LATENTS_1024, vae, device),
            torch.bfloat16,
            "bf16",
            id="cogview4",
        ),
        pytest.param(
            VAE_FLUX_SPEC,
            lambda device, vae: estimate_vae_working_memory_flux("decode", LATENTS_1024, vae, device=device),
            torch.bfloat16,
            "bf16",
            id="flux",
        ),
        pytest.param(
            VAE_FLUX2_SPEC,
            lambda device, vae: estimate_vae_working_memory_flux2(
                "decode", torch.zeros(1, 32, 128, 128), vae, device=device
            ),
            torch.float32,
            "fp32",
            id="flux2",
        ),
    ],
)
def test_a_decode_reserves_the_line_of_its_run_dtype_on_its_platform(
    on, platform, spec, estimate, vae_dtype, run_dtype
) -> None:
    cls = AutoencoderKLFlux2 if spec is VAE_FLUX2_SPEC else AutoencoderKL
    result = estimate(on(platform), _vae(cls, vae_dtype))

    assert result.features["dtype"] == run_dtype
    assert result.reservation_bytes == int(_line(spec, platform, "decode", run_dtype, 1024 * 1024))


def test_the_platforms_differ_where_the_measurements_do(on) -> None:
    """The point of keying on the allocator and, for MIOpen's native allocator, the OS: one SDXL fp16 decode, five
    prices."""
    decodes = {
        platform: estimate_vae_working_memory_sd15_sdxl(
            "decode", LATENTS_1024, _vae(), None, False, on(platform)
        ).reservation_bytes
        for platform in MEASURED
    }
    assert decodes[Platform("cudnn", "cudaMallocAsync", "linux")] < decodes[Platform("cudnn", "native", "linux")]
    assert decodes[Platform("miopen", "expandable", "windows")] < decodes[Platform("miopen", "native", "windows")]
    assert decodes[Platform("miopen", "native", "windows")] != decodes[Platform("miopen", "native", "linux")]


def test_the_cogview4_vae_is_priced_on_its_own_lines_without_a_score_matrix(on) -> None:
    """Its VAE is wider than the family's and has no mid-block attention: an FLUX.1-sized decode costs ~1.9x."""
    device = on(Platform("cudnn", "native", "linux"))
    cogview4 = estimate_vae_working_memory_cogview4("decode", LATENTS_1024, _vae(dtype=torch.bfloat16), device)
    flux = estimate_vae_working_memory_flux("decode", LATENTS_1024, _vae(dtype=torch.bfloat16), device=device)
    assert "score_matrix_bytes" not in cogview4.features
    assert cogview4.reservation_bytes > 1.8 * flux.reservation_bytes


@pytest.mark.parametrize(
    "operation, fp32, device, run_dtype",
    [
        ("decode", False, torch.device("cuda", 0), torch.float16),
        ("decode", True, torch.device("cuda", 0), torch.float32),
        # The decode node computes in fp32 on the CPU whatever it is asked; the encode node does not.
        ("decode", False, torch.device("cpu"), torch.float32),
        ("encode", False, torch.device("cpu"), torch.float16),
        ("encode", True, torch.device("cuda", 0), torch.float32),
    ],
)
@pytest.mark.parametrize("vae_dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
def test_the_sd_estimate_prices_the_dtype_the_node_casts_to_not_the_cached_one(
    on, operation, fp32, device, run_dtype, vae_dtype
) -> None:
    """The node casts the VAE from the dtype it was loaded in to the one it runs, and prices the latter."""
    on(Platform("cudnn", "native", "linux"))
    assert sd_vae_run_dtype(operation, fp32, device) == run_dtype
    image = LATENTS_1024[:, :4] if operation == "decode" else torch.zeros(1, 3, 1024, 1024)
    result = estimate_vae_working_memory_sd15_sdxl(operation, image, _vae(dtype=vae_dtype), None, fp32, device)
    assert result.features["dtype"] == {torch.float16: "fp16", torch.float32: "fp32"}[run_dtype]


def _tiny_sd_vae() -> AutoencoderKL:
    return AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=32,
    )


@pytest.fixture
def cached_vae():
    """The VAE as a node loads it: cached for the CPU, locked and unlocked by the model cache. Each lock's working
    memory is recorded."""
    caches: list[ModelCache] = []

    def load(vae: AutoencoderKL, estimates: list) -> LoadedModel:
        logger = MagicMock()
        logger.getEffectiveLevel.return_value = logging.INFO
        cache = ModelCache(
            execution_device_working_mem_gb=1.0,
            enable_partial_loading=True,
            keep_ram_copy_of_weights=True,
            execution_device="cpu",
            storage_device="cpu",
            logger=logger,
        )
        caches.append(cache)
        cache.put("vae", vae)
        vae_info = LoadedModel(config=None, cache_record=cache.get("vae"), cache=cache)
        model_on_device = vae_info.model_on_device

        def recording_model_on_device(working_memory):
            estimates.append(working_memory)
            return model_on_device(working_memory=working_memory)

        vae_info.model_on_device = recording_model_on_device  # type: ignore[method-assign]
        return vae_info

    yield load
    for cache in caches:
        cache.shutdown()


@pytest.mark.parametrize("fp32", [False, True])
def test_every_decode_runs_the_dtype_its_estimate_priced_and_prices_the_same_upcast(cached_vae, fp32) -> None:
    """On the CPU both requests decode in fp32: the node and the estimate share `sd_vae_run_dtype`. The cache returns the
    VAE to bf16 when a decode's lock ends, so the next decode's estimate prices the same upcast: reading a VAE left in
    fp32, it reserved nothing for the cast."""
    vae, estimates, decoded_in = _tiny_sd_vae().to(torch.bfloat16), [], []
    decode = vae.decode

    def recording_decode(*args, **kwargs):
        decoded_in.append(next(vae.decoder.parameters()).dtype)
        return decode(*args, **kwargs)

    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 4, 8, 8)
    context.config.get.return_value.force_tiled_decode = False
    context.models.load.return_value = cached_vae(vae, estimates)
    context.images.save.return_value = MagicMock(image_name="x.png", width=8, height=8)
    node = LatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="x"), vae=MagicMock(seamless_axes=[]), tiled=False, tile_size=0, fp32=fp32
    )

    with patch.object(vae, "decode", side_effect=recording_decode):
        node.invoke(context)
        node.invoke(context)

    assert decoded_in == [torch.float32, torch.float32]
    assert [estimate.features["dtype"] for estimate in estimates] == ["fp32", "fp32"]
    assert estimates[1].features["upcast_weight_bytes"] == estimates[0].features["upcast_weight_bytes"] > 0


@pytest.mark.parametrize("upcast, run_dtype", [(False, torch.float16), (True, torch.float32)])
def test_the_encode_node_runs_the_dtype_its_estimate_priced_and_returns_latents_in_the_vaes(
    cached_vae, upcast, run_dtype
) -> None:
    vae, estimates, encoded_in = _tiny_sd_vae().to(torch.bfloat16), [], []

    def encode(vae, image):
        encoded_in.append(next(vae.encoder.parameters()).dtype)
        return torch.zeros(1, 4, 2, 2)

    with patch.object(ImageToLatentsInvocation, "_encode_to_tensor", side_effect=encode):
        latents = ImageToLatentsInvocation.vae_encode(
            cached_vae(vae, estimates), upcast, False, torch.zeros(1, 3, 16, 16)
        )

    (estimate,) = estimates
    assert encoded_in == [run_dtype]
    assert estimate.features["dtype"] == {torch.float16: "fp16", torch.float32: "fp32"}[run_dtype]
    assert latents.dtype == torch.bfloat16


def test_a_failed_encode_leaves_the_vae_in_the_dtype_it_was_loaded_in(cached_vae) -> None:
    vae = _tiny_sd_vae().to(torch.bfloat16)

    with patch.object(ImageToLatentsInvocation, "_encode_to_tensor", side_effect=RuntimeError("encode failed")):
        with pytest.raises(RuntimeError, match="encode failed"):
            ImageToLatentsInvocation.vae_encode(cached_vae(vae, []), True, False, torch.zeros(1, 3, 16, 16))

    assert {p.dtype for p in vae.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize(
    "vae_dtype, fp32, growth",
    [
        (torch.float16, True, 1000 * 2),
        (torch.bfloat16, True, 1000 * 2),
        (torch.float32, True, 0),
        # Casting down frees memory rather than taking it.
        (torch.float32, False, 0),
        (torch.float16, False, 0),
    ],
)
def test_an_sd_upcast_reserves_what_the_cast_adds_to_the_weights(on, vae_dtype, fp32, growth) -> None:
    """The node casts the VAE in place once the cache has placed it, so the growth comes out of the reservation."""
    platform = Platform("cudnn", "native", "linux")
    device = on(platform)
    result = estimate_vae_working_memory_sd15_sdxl(
        "decode", LATENTS_1024[:, :4], _vae(dtype=vae_dtype, numel=1000), None, fp32, device
    )
    run_dtype = "fp32" if fp32 else "fp16"
    assert result.reservation_bytes == int(_line(VAE_SD_SPEC, platform, "decode", run_dtype, 1024 * 1024) + growth)


def _tiled_line(spec, platform: Platform, operation: str, dtype: str, pixels: int) -> float:
    values = shipped_coefficients(spec, platform).values
    return values[f"k_tiled_{operation}_{dtype}"] * pixels + values[f"b_tiled_{operation}_{dtype}"]


def _tiled(spec, platform: Platform, operation: str, dtype: str, tile_pixels: int, pixels: int) -> float:
    """A tiled operation as the shipped values price it: the untiled line at the tile plus the tiled line over the
    image, and for a decode the larger of that and the assembly line over the image."""
    tile_phase = _line(spec, platform, operation, dtype, tile_pixels) + _tiled_line(
        spec, platform, operation, dtype, pixels
    )
    if operation == "encode":
        return tile_phase
    values = shipped_coefficients(spec, platform).values
    return max(tile_phase, values[f"k_assembled_decode_{dtype}"] * pixels + values[f"b_assembled_decode_{dtype}"])


def _image(operation: str, height: int, width: int, channels: int) -> torch.Tensor:
    """What a node hands its estimator, without the memory: latents for a decode, the image for an encode."""
    if operation == "decode":
        return torch.zeros(1, channels, 1, 1).expand(1, channels, height // 8, width // 8)
    return torch.zeros(1, 3, 1, 1).expand(1, 3, height, width)


TILED = [
    pytest.param(
        VAE_SD_SPEC,
        lambda op, image, vae, tile, device: estimate_vae_working_memory_sd15_sdxl(op, image, vae, tile, True, device),
        AutoencoderKL,
        4,
        "fp32",
        1000 * 2,
        id="sd-fp32-upcast",
    ),
    pytest.param(
        VAE_SD_SPEC,
        lambda op, image, vae, tile, device: estimate_vae_working_memory_sd15_sdxl(op, image, vae, tile, False, device),
        AutoencoderKL,
        4,
        "fp16",
        0,
        id="sd-fp16",
    ),
    pytest.param(
        VAE_FLUX_SPEC,
        lambda op, image, vae, tile, device: estimate_vae_working_memory_flux(op, image, vae, tile, device),
        AutoencoderKL,
        16,
        "bf16",
        0,
        id="flux",
    ),
    pytest.param(
        VAE_FLUX2_SPEC,
        lambda op, image, vae, tile, device: estimate_vae_working_memory_flux2(op, image, vae, tile, device),
        AutoencoderKLFlux2,
        32,
        "bf16",
        0,
        id="flux2",
    ),
]


@pytest.mark.parametrize("side", [2048, 8192])
@pytest.mark.parametrize("platform", MEASURED, ids=str)
@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("spec, estimate, cls, channels, dtype, growth", TILED)
def test_a_tiled_operation_is_its_tile_plus_the_tiled_line_and_a_decode_the_larger_of_that_and_its_assembly(
    on, side, platform, operation, spec, estimate, cls, channels, dtype, growth
) -> None:
    """No margin on the tile: what tiling adds over the tile's own demand is the tiled line's, in the image's pixels.
    A decode then assembles the image from its tiles, a phase of its own priced on the assembly line in the image's
    pixels, and reserves the larger phase; an encode has no such phase. The buffers each holds take over only past the
    measured sides, and each operation counts its own."""
    vae = _vae(cls, torch.float16 if spec is VAE_SD_SPEC else torch.bfloat16, numel=1000)
    result = estimate(operation, _image(operation, side, side, channels), vae, 512, on(platform))

    assert result.features["tile"] == 512
    assert result.reservation_bytes == int(_tiled(spec, platform, operation, dtype, 512 * 512, side * side) + growth)
    names = {term.name for term in result.terms}
    assert ("assembled" in names) == (operation == "decode")
    assert {name for name in names if name.endswith("_buffers")} == {BUFFER_TERMS[operation]}


def test_the_4090_native_sd_fp32_decode_at_a_512px_tile_on_an_8192px_image_reserves_its_assembly(on) -> None:
    """Measured at 4.94 GiB, the larger of a cold and a warm run (`data/vae.json`): the decoded tiles, the rows and the
    image they assemble into, and the node's post-processing, on the native allocator, about three times what the
    tile itself takes. The assembly line prices it; the data test holds it to its upper bound."""
    measured = 5307891712
    device = on(Platform("cudnn", "native", "windows"))
    vae = _vae(dtype=torch.float32)
    result = estimate_vae_working_memory_sd15_sdxl("decode", _image("decode", 8192, 8192, 4), vae, 512, True, device)

    assert result.features["dtype"] == "fp32"
    assert any(term.included and term.name == "assembled" for term in result.terms)
    assert measured <= result.reservation_bytes


@pytest.mark.parametrize("spec, estimate, cls, channels, dtype, growth", TILED)
def test_an_elongated_image_prices_the_tile_clamped_per_axis(on, spec, estimate, cls, channels, dtype, growth) -> None:
    """The VAE slices tiles out of the image, so a 512 px tile over a 384 px axis is 384 px on it: an 8192 px tile on a
    1024x8200 image would otherwise price 8192x8192."""
    platform = Platform("cudnn", "native", "linux")
    device = on(platform)
    vae = _vae(cls, torch.float16 if spec is VAE_SD_SPEC else torch.bfloat16, numel=1000)
    with patch("invokeai.backend.util.vae_working_memory._vae_mid_block_score_matrix_bytes", return_value=0) as score:
        result = estimate("decode", _image("decode", 384, 4096, channels), vae, 512, device)

    expected = _tiled(spec, platform, "decode", dtype, 384 * 512, 384 * 4096)
    assert result.reservation_bytes == int(expected + growth)
    if spec is not VAE_FLUX2_SPEC:
        # The score matrix is the tile's too, clamped the same way (FLUX.2 sizes its own; see its tests).
        assert score.call_args.args[:2] == (384, 512)


@pytest.mark.parametrize("tile", [1024, 2048])
@pytest.mark.parametrize("spec, estimate, cls, channels, dtype, growth", TILED)
def test_a_tile_covering_the_image_is_priced_as_the_single_pass_it_runs(
    on, tile, spec, estimate, cls, channels, dtype, growth
) -> None:
    """`AutoencoderKL` tiles only when a side is strictly larger than the tile."""
    platform = Platform("miopen", "native", "windows")
    vae = _vae(cls, torch.float16 if spec is VAE_SD_SPEC else torch.bfloat16, numel=1000)
    result = estimate("decode", _image("decode", 1024, 1024, channels), vae, tile, on(platform))

    assert result.features["tile"] is None
    assert result.reservation_bytes == int(_line(spec, platform, "decode", dtype, 1024 * 1024) + growth)


@pytest.mark.parametrize("field", [0, -8])
def test_an_sd_tile_field_of_0_or_below_takes_the_tile_the_vae_carries(on, field) -> None:
    """The nodes patch the VAE's tile only for a field above 0, and leave its own in place otherwise: the SDXL VAE's
    sample size, 1024 px."""
    vae = _vae()
    vae.tile_sample_min_size = 1024
    result = estimate_vae_working_memory_sd15_sdxl(
        "encode", _image("encode", 2048, 2048, 3), vae, field, False, on(Platform("cudnn", "native", "linux"))
    )
    assert result.features["tile"] == 1024


def _sd_shaped_vae() -> AutoencoderKL:
    """A real SD-shaped `AutoencoderKL`: four blocks, so the 8x downsample its tile is snapped at, channels cut to
    almost nothing. Its own tile, its `sample_size` of 256 px, is smaller than the images below, so a field of 0 tiles
    too."""
    return AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=2,
        sample_size=256,
    )


def _tiny_autoencoder() -> AutoencoderTiny:
    """A real `AutoencoderTiny`, which names its 8x downsample `spatial_scale_factor` and carries a 512 px tile."""
    return AutoencoderTiny(encoder_block_out_channels=(4, 4, 4, 4), decoder_block_out_channels=(4, 4, 4, 4))


def _flux2_shaped_vae() -> AutoencoderKLFlux2:
    return AutoencoderKLFlux2(
        block_out_channels=(4, 4, 4, 4), layers_per_block=1, latent_channels=32, norm_num_groups=2, sample_size=64
    )


def _run_sd_node(
    cached_vae, operation: str, vae: AutoencoderKL | AutoencoderTiny, field: int, image: torch.Tensor
) -> list:
    """Run `l2i` (decode) or `i2l` (encode) tiled at the tile field `field` on the CPU, the VAE loaded through
    `cached_vae`, and return the estimates the node priced, one per `model_on_device`. The VAE's own `decode` and
    `encode` still run; a caller spies on them."""
    estimates: list = []
    vae_info = cached_vae(vae, estimates)
    if operation == "decode":
        context = MagicMock()
        context.tensors.load.return_value = image
        context.config.get.return_value.force_tiled_decode = False
        context.models.load.return_value = vae_info
        context.images.save.return_value = MagicMock(image_name="x.png", width=8, height=8)
        node = LatentsToImageInvocation.model_construct(
            latents=MagicMock(latents_name="x"),
            vae=MagicMock(seamless_axes=[]),
            tiled=True,
            tile_size=field,
            fp32=False,
        )
        node.invoke(context)
    else:
        # Upcast, so the encode runs in fp32 as the CPU decode does: no half-precision kernels on the CPU.
        ImageToLatentsInvocation.vae_encode(vae_info, True, True, image, field)
    return estimates


@pytest.mark.parametrize(
    "build, field, runs",
    [
        pytest.param(_sd_shaped_vae, 0, 256, id="kl-own-tile"),
        pytest.param(_sd_shaped_vae, 24, 32, id="kl-24"),
        pytest.param(_sd_shaped_vae, 200, 192, id="kl-200"),
        pytest.param(_sd_shaped_vae, 512, 512, id="kl-512"),
        pytest.param(_tiny_autoencoder, 0, 512, id="tiny-own-tile"),
        pytest.param(_tiny_autoencoder, 24, 32, id="tiny-24"),
        pytest.param(_tiny_autoencoder, 200, 192, id="tiny-200"),
        pytest.param(_tiny_autoencoder, 512, 512, id="tiny-512"),
        # The tiny autoencoder carries a 0.125 overlap, at which 104 px would snap to 64 px; the node patches 0.25.
        pytest.param(_tiny_autoencoder, 104, 96, id="tiny-104"),
    ],
)
@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_the_sd_nodes_price_the_tile_their_vae_runs(cached_vae, operation, build, field, runs) -> None:
    """`patch_vae_tiling_params` gives the VAE the largest tile at or below the field that it can assemble at the
    overlap the node patches in, 32 px at the least, and the node's estimate prices that tile rather than the field: a
    24 px field priced at 24 px under-reserves the 32 px tile that runs, a 200 px one priced at 200 px over-reserves a
    192 px tile. A field of 0 leaves the VAE's own tile, which the estimate reads off it. The geometry is read when the
    node calls the VAE."""
    vae, ran = build(), []
    real = getattr(vae, operation)

    def record(*args, **kwargs):
        ran.append((vae.tile_sample_min_size, vae.tile_latent_min_size))
        return real(*args, **kwargs)

    setattr(vae, operation, MagicMock(side_effect=record))
    (estimate,) = _run_sd_node(cached_vae, operation, vae, field, _image(operation, 32, 544, 4))

    assert ran == [(runs, runs // 8)]
    assert estimate.features["tile"] == runs


@pytest.mark.parametrize("field, runs", [(0, 512), (24, 128), (200, 192), (512, 512)])
@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("family", ["flux", "flux2"])
def test_the_flux_estimates_price_the_tile_scoped_vae_tiling_runs(
    flux_shaped_vae, family, operation, field, runs
) -> None:
    """The FLUX.1 and FLUX.2 nodes hand the same field to their estimator and to `scoped_vae_tiling`, which resolves it
    (0 to 512 px, 128 px at the least) and snaps it to a tile the VAE can assemble; the estimate prices that tile."""
    if family == "flux":
        vae, estimate, channels = flux_shaped_vae(), estimate_vae_working_memory_flux, 16
    else:
        vae, estimate, channels = _flux2_shaped_vae(), estimate_vae_working_memory_flux2, 32
    with scoped_vae_tiling(vae, field):
        ran = (vae.tile_sample_min_size, vae.tile_latent_min_size)

    result = estimate(operation, _image(operation, 32, 544, channels), vae, field, torch.device("cpu"))

    assert ran == (runs, runs // 8)
    assert result.features["tile"] == runs


@pytest.mark.parametrize("family", ["sd", "flux", "flux2"])
def test_a_field_as_large_as_the_image_that_runs_a_smaller_tile_prices_the_tiled_decode(
    cached_vae, flux_shaped_vae, family
) -> None:
    """A 200 px field on a 200 px image reads as a covering tile, which the VAE would decode in a single pass; it runs a
    192 px tile, smaller than the image, so the VAE tiles and the estimate has to price the tiled decode."""
    latents = torch.zeros(1, {"sd": 4, "flux": 16, "flux2": 32}[family], 25, 25)
    vae = {"sd": _sd_shaped_vae, "flux": flux_shaped_vae, "flux2": _flux2_shaped_vae}[family]()
    vae.tiled_decode = MagicMock(wraps=vae.tiled_decode)

    if family == "sd":
        (result,) = _run_sd_node(cached_vae, "decode", vae, 200, latents)
    else:
        estimate = estimate_vae_working_memory_flux if family == "flux" else estimate_vae_working_memory_flux2
        result = estimate("decode", latents, vae, 200, torch.device("cpu"))
        with torch.no_grad(), scoped_vae_tiling(vae, 200):
            vae.decode(latents, return_dict=False)

    assert vae.tiled_decode.called
    assert result.features["tile"] == 192


@pytest.mark.parametrize("side", [2048, 8192])
def test_a_tiled_tiny_autoencoder_is_priced_on_the_lines_without_a_score_matrix(on, side) -> None:
    """The SD estimator prices the tiny autoencoder on the AutoencoderKL lines, which cost many times what its own
    convolutions do (64 channels wide against 512), and with no mid-block attention, so no score matrix."""
    platform = Platform("cudnn", "native", "linux")
    vae = _vae(AutoencoderTiny)
    result = estimate_vae_working_memory_sd15_sdxl(
        "decode", _image("decode", side, side, 4), vae, 512, False, on(platform)
    )

    assert result.features["vae_class"] == "AutoencoderTiny" and "score_matrix_bytes" not in result.features
    assert result.reservation_bytes == int(_tiled(VAE_SD_SPEC, platform, "decode", "fp16", 512 * 512, side * side))
    assert "mid_block_scores" not in {term.name for term in result.terms}


def _flux2_decode(platform: Platform, side: int, batch: int, on) -> WorkingMemoryEstimate:
    latents = torch.zeros(1, 32, 1, 1).expand(batch, 32, side // 8, side // 8)
    return estimate_vae_working_memory_flux2(
        "decode", latents, _vae(AutoencoderKLFlux2, torch.bfloat16), 512, on(platform)
    )


def test_a_tiled_batch_whose_tile_phase_decides_is_that_many_single_samples(on) -> None:
    """On cuDNN with the native allocator a 2048 px decode's tile phase is the larger: the tile, the tiled line and its
    base all scale with the samples."""
    platform = Platform("cudnn", "native", "linux")
    single, batch = (_flux2_decode(platform, 2048, n, on) for n in (1, 3))

    assert not any(term.included and term.name.startswith("assembled") for term in single.terms)
    assert batch.reservation_bytes == pytest.approx(3 * single.reservation_bytes, abs=3)


def test_a_negative_assembly_base_is_taken_once_for_a_batch(on) -> None:
    """``k * P + max(b, n * b)``: the base of an assembly line that falls below zero is not multiplied down by the
    samples, which would price a batch's assembly below one sample's line over the same pixels."""
    platform = Platform("cudnn", "native", "linux")
    features = dict(_flux2_decode(platform, 8192, 2, on).features)
    values = dict(shipped_coefficients(VAE_FLUX2_SPEC, platform).values)
    values |= {"k_assembled_decode_bf16": 100, "b_assembled_decode_bf16": -100 * MIB}

    result = VAE_FLUX2_SPEC.evaluate(features, ResolvedCoefficients(values=values, source="test"))

    assert result.reservation_bytes == 100 * 2 * 8192 * 8192 - 100 * MIB


def _assembly_buffers(height: int, width: int, tile: int, element_size: int, samples: int = 1) -> int:
    """What diffusers' `tiled_decode` holds when it concatenates the image, walked tile by tile as it slices the
    latents: every decoded tile (its slice clipped at the image's far edges, 8 output pixels per latent), the rows
    blended from them and the image, 3 channels each. Counted here rather than read from the estimator."""
    latent_height, latent_width, latent_tile = height // 8, width // 8, tile // 8
    stride = int(latent_tile * 0.75)
    tile_pixels = 0
    for top in range(0, latent_height, stride):
        for left in range(0, latent_width, stride):
            rows = min(top + latent_tile, latent_height) - top
            columns = min(left + latent_tile, latent_width) - left
            tile_pixels += 64 * rows * columns
    return 3 * element_size * samples * (tile_pixels + 2 * height * width)


def _encode_buffers(
    height: int, width: int, tile: int, latent_channels: int, element_size: int, samples: int = 1
) -> int:
    """What diffusers' `_tiled_encode` holds when it concatenates the moments, walked tile by tile as it slices the
    image: the input image, every encoded tile (its slice clipped at the image's far edges, 8 pixels per latent,
    rounded down), the rows of the crops the blends keep and the moments concatenated from them, each twice
    `latent_channels` wide. Counted here rather than read from the estimator."""
    stride, latent_tile = int(tile * 0.75), tile // 8
    keep = latent_tile - latent_tile // 4
    tile_latents = kept_latents = 0
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            rows = (min(top + tile, height) - top) // 8
            columns = (min(left + tile, width) - left) // 8
            tile_latents += rows * columns
            kept_latents += min(rows, keep) * min(columns, keep)
    moments = 2 * latent_channels * (tile_latents + 2 * kept_latents)
    return element_size * samples * (3 * height * width + moments)


LATENT_CHANNELS = {VAE_SD_SPEC.id: 4, VAE_FLUX_SPEC.id: 16, VAE_FLUX2_SPEC.id: 32}


def _buffers(spec, operation: str, height: int, width: int, tile: int, element_size: int, samples: int = 1) -> int:
    if operation == "decode":
        return _assembly_buffers(height, width, tile, element_size, samples)
    return _encode_buffers(height, width, tile, LATENT_CHANNELS[spec.id], element_size, samples)


BUFFER_TERMS = {"decode": "assembly_buffers", "encode": "encode_buffers"}


def _buffer_term(result: WorkingMemoryEstimate, operation: str) -> WorkingMemoryTerm:
    [term] = [term for term in result.terms if term.name == BUFFER_TERMS[operation]]
    return term


def _tiled_features(spec, operation: str, height: int, width: int, tile: int, dtype: str) -> dict:
    """The features each estimator records for a single-sample tiled operation, the score matrix held at zero."""
    features = {
        "operation": operation,
        "height": height,
        "width": width,
        "tile": tile,
        "dtype": dtype,
        "score_matrix_bytes": 0,
    }
    if spec is VAE_SD_SPEC:
        features |= {"vae_class": "AutoencoderKL", "upcast_weight_bytes": 0}
    if spec is VAE_FLUX2_SPEC:
        features["batch_size"] = 1
    return features


@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_a_tiled_operation_reserves_at_least_the_buffers_it_holds_up_to_32768_px(operation) -> None:
    """No node field bounds an operation's size, an image made in the graph reaches the encode nodes without PIL's
    decompression-bomb check, and the fitted lines are extrapolated past the 12288 px the tiled points measure: on
    several platforms they fall below the buffers the operation cannot do without. Every platform and dtype each
    estimator ships reserves at least them, at the tiles the app sets."""
    sides, tiles = (12288, 16384, 24576, 32768), (512, 768, 1024)
    short = []
    for spec in (VAE_SD_SPEC, VAE_FLUX_SPEC, VAE_FLUX2_SPEC):
        prefix = f"k_{operation}_"
        dtypes = sorted({name.removeprefix(prefix) for name in spec.coefficients if name.startswith(prefix)})
        buffers = {(side, tile): _buffers(spec, operation, side, side, tile, 1) for side in sides for tile in tiles}
        for platform in MEASURED:
            coefficients = shipped_coefficients(spec, platform)
            for dtype, side, tile in itertools.product(dtypes, sides, tiles):
                features = _tiled_features(spec, operation, side, side, tile, dtype)
                reserved = spec.evaluate(features, coefficients).reservation_bytes
                if reserved < buffers[(side, tile)] * ELEMENT_SIZES[dtype]:
                    short.append(f"{spec.id} {platform} {dtype} {side}px tile {tile}")
    assert not short


def test_the_reviewed_flux_decode_reserves_its_assembly_buffers(on) -> None:
    """A 24576 px FLUX.1 bf16 decode at a 512 px tile on Windows ROCm with expandable segments: its fitted phases
    price 11.83 GiB, but the decoded tiles, the rows and the image alone take 12.70 GiB."""
    device = on(Platform("miopen", "expandable", "windows"))
    latents = torch.zeros(1, 16, 1, 1).expand(1, 16, 3072, 3072)
    result = estimate_vae_working_memory_flux("decode", latents, _vae(dtype=torch.bfloat16), 512, device)

    assert result.reservation_bytes == _assembly_buffers(24576, 24576, 512, 2)
    assert result.reservation_bytes >= 12.70 * 2**30


def test_a_large_flux_fp32_encode_reserves_its_input_and_moments(on) -> None:
    """A 24576 px FLUX.1 fp32 encode at a 512 px tile on Windows ROCm with expandable segments, the size a 16x upscale
    of a 1536 px image hands the encode node: its fitted phase prices 0.95x of what the input image and the moments it
    assembles take."""
    device = on(Platform("miopen", "expandable", "windows"))
    image = _image("encode", 24576, 24576, 3)
    result = estimate_vae_working_memory_flux("encode", image, _vae(dtype=torch.float32), 512, device)

    assert result.reservation_bytes == _encode_buffers(24576, 24576, 512, 16, 4)


def test_a_non_square_decode_counts_the_clipped_edge_tiles_it_decodes(on) -> None:
    """At a 512 px tile the latents are stepped 48 at a time: 16392 px is 2049 latents, so the last row of tiles is
    33 latents tall, and 24584 px is 3073, so the last column is 49 wide. The buffers count those as they are decoded,
    neither as whole tiles nor as the image alone. FLUX.2 bf16 under `cudaMallocAsync`, where they decide."""
    device = on(Platform("cudnn", "cudaMallocAsync", "linux"))
    latents = torch.zeros(1, 32, 1, 1).expand(1, 32, 16392 // 8, 24584 // 8)
    vae = _vae(AutoencoderKLFlux2, torch.bfloat16)
    result = estimate_vae_working_memory_flux2("decode", latents, vae, 512, device)

    assert result.reservation_bytes == _assembly_buffers(16392, 24584, 512, 2)


@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_a_tile_covering_one_axis_counts_the_tiles_it_slices_along_the_other(on, operation) -> None:
    """A 1024 px tile over a 512x32768 image covers its height, so the VAE still tiles, in one row of 512 px tall
    tiles: neither the image alone nor whole tiles."""
    device = on(Platform("cudnn", "native", "linux"))
    vae = _vae(AutoencoderKLFlux2, torch.bfloat16)
    result = estimate_vae_working_memory_flux2(operation, _image(operation, 512, 32768, 32), vae, 1024, device)

    assert _buffer_term(result, operation).bytes == _buffers(VAE_FLUX2_SPEC, operation, 512, 32768, 1024, 2)


def test_a_tiled_batch_reserves_the_assembly_buffers_of_every_sample(on) -> None:
    """`tiled_decode` slices the whole batch at once, so every buffer holds every sample."""
    result = _flux2_decode(Platform("cudnn", "cudaMallocAsync", "linux"), 16384, 3, on)

    assert result.reservation_bytes == _assembly_buffers(16384, 16384, 512, 2, samples=3)


def test_a_tiled_batch_encode_counts_the_buffers_of_every_sample(on) -> None:
    """`_tiled_encode` slices the whole batch at once too, input image included."""
    device = on(Platform("cudnn", "native", "linux"))
    image = torch.zeros(1, 3, 1, 1).expand(2, 3, 4096, 4096)
    result = estimate_vae_working_memory_flux2("encode", image, _vae(AutoencoderKLFlux2, torch.bfloat16), 512, device)

    assert _buffer_term(result, "encode").bytes == _encode_buffers(4096, 4096, 512, 32, 2, samples=2)


@pytest.mark.parametrize("fp32, element_size", [(True, 4), (False, 2)])
@pytest.mark.parametrize("field, tile", [(200, 192), (0, 512)])
@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_the_sd_estimate_counts_the_buffers_of_the_tile_and_dtype_its_node_runs(
    on, operation, field, tile, fp32, element_size
) -> None:
    """A field above 0 runs the tile `diffusers_tile` snaps it to, a field of 0 the VAE's own; the node runs fp32 when
    asked and fp16 otherwise. A 2056x3080 image, so the edge tiles are clipped."""
    vae = _vae()
    vae.tile_sample_min_size = 512
    image = _image(operation, 2056, 3080, 4)
    result = estimate_vae_working_memory_sd15_sdxl(
        operation, image, vae, field, fp32, on(Platform("cudnn", "native", "linux"))
    )

    assert result.features["tile"] == tile
    assert _buffer_term(result, operation).bytes == _buffers(VAE_SD_SPEC, operation, 2056, 3080, tile, element_size)


def _recording(module: torch.nn.Module, shapes: list) -> None:
    """Record the shape of every output `module` returns."""
    forward = module.forward

    def record(*args, **kwargs):
        out = forward(*args, **kwargs)
        shapes.append(out.shape)
        return out

    module.forward = record


def test_the_counted_buffers_are_the_tiles_diffusers_decodes_and_two_images() -> None:
    """The count rests on diffusers' `tiled_decode`: run the real one on an image whose sides are not tile-aligned and
    sum the tiles its decoder returns."""
    vae, shapes = _flux2_shaped_vae(), []
    _recording(vae.decoder, shapes)
    latents = torch.zeros(1, 32, 41, 70)
    result = estimate_vae_working_memory_flux2("decode", latents, vae, 128, torch.device("cpu"))
    with torch.no_grad(), scoped_vae_tiling(vae, 128):
        image = vae.decode(latents, return_dict=False)[0]

    assert len(shapes) == 4 * 6
    decoded = sum(shape[-2] * shape[-1] for shape in shapes)
    assert _buffer_term(result, "decode").bytes == 3 * 4 * (decoded + 2 * image.shape[-2] * image.shape[-1])


def test_the_counted_buffers_are_the_input_and_the_moment_tiles_diffusers_encodes_twice_over() -> None:
    """The count rests on diffusers' `_tiled_encode`: run the real one on an image whose sides are not multiples of 8
    and sum the moment tiles its encoder returns; the rows blended from them and the concatenated moments come to the
    moments it returns, each."""
    vae, shapes = _flux2_shaped_vae(), []
    _recording(vae.encoder, shapes)
    image = torch.zeros(1, 3, 330, 565)
    result = estimate_vae_working_memory_flux2("encode", image, vae, 128, torch.device("cpu"))
    with torch.no_grad(), scoped_vae_tiling(vae, 128):
        moments = vae.encode(image).latent_dist.parameters

    assert len(shapes) == 4 * 6
    encoded = sum(shape[-3] * shape[-2] * shape[-1] for shape in shapes)
    expected = 4 * (3 * 330 * 565 + encoded + 2 * moments[0].numel())
    assert _buffer_term(result, "encode").bytes == expected


def test_no_shipped_tiled_operation_is_decided_by_its_buffers() -> None:
    """The fit rates a tiled operation's points on its fitted phases alone (`fit.two_phase_envelope`,
    `fit.upper_envelope`). A point the buffers decided would be rated on a phase the estimate does not reserve, so none
    may be one: the buffers only take over past the measured sides."""
    specs = {spec.id: spec for spec in (VAE_SD_SPEC, VAE_FLUX_SPEC, VAE_FLUX2_SPEC)}
    data = shipped_data().values()
    points = [
        point for file in data for point in file.points if point.estimator in specs and point.features.get("tile")
    ]
    operations = {point.features["operation"] for point in points}
    assert operations == {"decode", "encode"}, f"shipped tiled points measure only {operations} -- the check is broken"
    for point in points:
        spec = specs[point.estimator]
        buffers = _buffer_term(evaluate_point(point, spec, data), str(point.features["operation"]))
        assert not buffers.included, point_id(point, spec)


ESTIMATES = [
    pytest.param(
        lambda operation, image, vae, tile, device: estimate_vae_working_memory_sd15_sdxl(
            operation, image, vae, tile, False, device
        ),
        16,
        id="sd",
    ),
    pytest.param(
        lambda operation, image, vae, tile, device: estimate_vae_working_memory_flux(
            operation, image, vae, tile, device
        ),
        16,
        id="flux",
    ),
    pytest.param(
        lambda operation, image, vae, tile, device: estimate_vae_working_memory_flux2(
            operation, image, vae, tile, device
        ),
        32,
        id="flux2",
    ),
]


def _vae_at_half_overlap() -> MagicMock:
    vae = _vae(dtype=torch.bfloat16)
    vae.tile_sample_min_size = 512
    vae.tile_overlap_factor = 0.5
    return vae


@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("estimate, channels", ESTIMATES)
def test_a_tiled_operation_at_an_overlap_its_buffers_are_not_counted_at_is_refused(
    on, operation, estimate, channels
) -> None:
    """The buffers are counted on the tile grid at the 0.25 overlap every shipped AutoencoderKL tiles at, from the
    recorded features alone; a VAE tiling at another would be priced on the wrong grid. The SD nodes run a field of 0
    at the VAE's own overlap; the FLUX nodes every field."""
    device = on(Platform("cudnn", "native", "linux"))
    with pytest.raises(ValueError, match=r"tiles at an overlap of 0\.5"):
        estimate(operation, _image(operation, 2048, 2048, channels), _vae_at_half_overlap(), 0, device)


@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_an_sd_field_above_0_is_priced_at_the_overlap_the_node_patches_not_the_vaes(on, operation) -> None:
    """The SD nodes patch the overlap to 0.25 along with the tile for a field above 0, so the VAE's own does not run."""
    device = on(Platform("cudnn", "native", "linux"))
    image = _image(operation, 2048, 2048, 4)
    result = estimate_vae_working_memory_sd15_sdxl(operation, image, _vae_at_half_overlap(), 512, False, device)

    assert result.features["tile"] == 512


@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("estimate, channels", ESTIMATES)
def test_a_covering_tile_is_priced_whatever_overlap_the_vae_carries(on, operation, estimate, channels) -> None:
    """A tile covering the image runs the single pass, which has no tile grid to count."""
    device = on(Platform("cudnn", "native", "linux"))
    result = estimate(operation, _image(operation, 512, 512, channels), _vae_at_half_overlap(), 0, device)

    assert result.features["tile"] is None


@pytest.mark.parametrize("operation", ["decode", "encode"])
def test_the_tiny_autoencoder_is_priced_whatever_overlap_it_tiles_at(on, operation) -> None:
    """It tiles at its own 0.125 and keeps no tiles: a decode assembles into one fp32 buffer, 12 bytes per pixel,
    below the 18 or more the buffers count for it, and an encode holds its input and one fp32 latent."""
    device = on(Platform("cudnn", "native", "linux"))
    vae = _tiny_autoencoder()
    assert vae.tile_overlap_factor == 0.125
    result = estimate_vae_working_memory_sd15_sdxl(operation, _image(operation, 2048, 2048, 4), vae, 0, False, device)

    assert result.features["tile"] == 512


def test_a_dtype_the_estimator_has_no_line_for_is_refused_by_name(on) -> None:
    """The FLUX.2 loaders never run fp16; a VAE that does anyway must not fail mid-generation on a bare KeyError."""
    device = on(Platform("cudnn", "native", "linux"))
    with pytest.raises(ValueError, match=r"'vae\.flux2' has no fp16 decode line on Platform\(conv_backend='cudnn'"):
        estimate_vae_working_memory_flux2(
            "decode", torch.zeros(1, 32, 64, 64), _vae(AutoencoderKLFlux2, torch.float16), device=device
        )


@pytest.mark.parametrize(
    "leading, dtype",
    [
        pytest.param(torch.zeros(4, dtype=torch.float8_e4m3fn), "bf16", id="fp8-storage"),
        pytest.param(torch.zeros(4, dtype=torch.int8), "bf16", id="packed-integer"),
    ],
)
def test_the_compute_dtype_skips_weights_stored_narrower_than_they_compute(on, leading, dtype) -> None:
    """An SDNQ VAE's first parameter can be its packed integer storage; it computes in its floating-point layers'."""
    device = on(Platform("cudnn", "native", "linux"))
    vae = _vae(params=[leading, torch.zeros(1, dtype=torch.bfloat16)])
    assert estimate_vae_working_memory_flux("decode", LATENTS_1024, vae, device=device).features["dtype"] == dtype


def test_an_sdnq_vae_is_priced_in_its_packed_weights_compute_dtype_not_an_fp32_bias(on) -> None:
    """The packed weight dequantizes to its `compute_dtype`, and every other operand of its layer is cast to that dtype
    with it, so a bias stored in fp32 does not make the layer compute in fp32."""
    device = on(Platform("cudnn", "native", "linux"))
    packed_weight = SDNQTensor(
        data=torch.zeros(4, 4, dtype=torch.int8),
        quantization_type=SDNQQuantizationType.INT8_SYM,
        tensor_shape=torch.Size([4, 4]),
        compute_dtype=torch.bfloat16,
        scale=torch.ones(4, 1, dtype=torch.float32),
    )
    vae = _vae(params=[packed_weight, torch.zeros(4, dtype=torch.float32)])
    assert estimate_vae_working_memory_flux("decode", LATENTS_1024, vae, device=device).features["dtype"] == "bf16"


def test_every_stand_in_slope_is_at_least_the_pre_dtype_figure() -> None:
    """A stand-in is a declared conservative value on a measured platform; nothing else guards it."""
    stand_ins = [entry for data in shipped_data().values() for entry in data.coefficients if entry.stand_ins]
    assert stand_ins, "the shipped data declares no stand-ins -- the check is broken"
    for entry in stand_ins:
        for name in entry.stand_ins:
            if "_tiled_" in name or "_assembled_" in name:
                # A tiled line sits on the untiled one and pairs with its assembly line; the floor below is the
                # untiled lines'.
                continue
            part, operation, dtype = name.split("_")
            value = entry.values[name]
            if part == "k":
                floor = PRE_DTYPE_BYTES_PER_PIXEL_PER_ELEMENT_BYTE[entry.platform["conv_backend"]][operation]
                assert value >= floor * ELEMENT_SIZES[dtype], f"{entry.estimator} {entry.platform} {name}"
            else:
                assert value >= 0


@pytest.mark.parametrize(
    "spec", [VAE_SD_SPEC, VAE_SD3_SPEC, VAE_COGVIEW4_SPEC, VAE_FLUX_SPEC, VAE_FLUX2_SPEC], ids=lambda spec: spec.id
)
@pytest.mark.parametrize(
    "platform",
    [
        Platform("mps", "native", "other"),
        Platform("xpu", "native", "windows"),
        Platform("cpu", "native", "linux"),
        Platform("cudnn", "expandable", "linux"),
        Platform("miopen", "expandable", "linux"),
        Platform("miopen", "expandable", "other"),
        Platform("miopen", "cudaMallocAsync", "linux"),
        Platform("miopen", "cudaMallocAsync", "windows"),
    ],
    ids=str,
)
def test_an_unmeasured_platform_never_prices_a_slope_below_the_pre_dtype_figure(spec, platform) -> None:
    measured = priced_as(platform)
    values = shipped_coefficients(spec, platform).values
    for name, value in values.items():
        if "_tiled_" in name or "_assembled_" in name:
            continue  # See `test_an_unmeasured_platform_prices_each_tiled_value_at_its_largest_measured_one`.
        part, operation, dtype = name.split("_")
        if part == "k":
            floor = PRE_DTYPE_BYTES_PER_PIXEL_PER_ELEMENT_BYTE[measured.conv_backend][operation] * ELEMENT_SIZES[dtype]
            assert value >= floor, f"{name} on {platform}: {value} under {floor}"
        else:
            assert value >= 0


@pytest.mark.parametrize("spec", [VAE_SD_SPEC, VAE_FLUX_SPEC, VAE_FLUX2_SPEC], ids=lambda spec: spec.id)
@pytest.mark.parametrize(
    "platform",
    [
        Platform("mps", "native", "other"),
        Platform("xpu", "native", "windows"),
        Platform("cpu", "native", "linux"),
        Platform("cudnn", "expandable", "linux"),
        Platform("miopen", "expandable", "linux"),
        Platform("miopen", "cudaMallocAsync", "windows"),
        # Measured for its untiled lines, never tiled: its tiled lines are stand-ins at the same figures.
        Platform("miopen", "native", "linux"),
    ],
    ids=str,
)
def test_an_unmeasured_platform_prices_each_tiled_value_at_its_largest_measured_one(spec, platform) -> None:
    """The measured tiled lines differ per platform by up to 0.5 GiB of base, the assembly lines by up to 3.4 GiB; a
    platform no tiled point measures takes, value by value, the largest of the four measured ones, a line that lies on
    or above each measured platform's at every size."""
    tiled_platforms = [p for p in MEASURED if p != Platform("miopen", "native", "linux")]
    values = shipped_coefficients(spec, platform).values
    for name in (name for name in spec.coefficients if "_tiled_" in name or "_assembled_" in name):
        largest = max(shipped_coefficients(spec, measured).values[name] for measured in tiled_platforms)
        assert values[name] == largest, f"{name} on {platform}"


@pytest.mark.parametrize("platform", [p for p in MEASURED if p != Platform("miopen", "native", "linux")], ids=str)
@pytest.mark.parametrize(
    "spec, dtype",
    [(VAE_FLUX_SPEC, "fp16"), (VAE_FLUX_SPEC, "fp32"), (VAE_FLUX2_SPEC, "fp32")],
    ids=["flux-fp16", "flux-fp32", "flux2-fp32"],
)
def test_a_dtype_no_tiled_point_measures_takes_the_larger_of_the_sdxl_line_and_its_own_bf16_line(
    platform, spec, dtype
) -> None:
    """No tiled point measures FLUX.1 in fp16 or fp32, or FLUX.2 in fp32. Their tiled and assembly lines stand in at,
    value by value, the larger of the SDXL VAE's line of that dtype and the family's own bf16 line scaled by the
    element size (x2 for fp32): the FLUX encoders emit 4 to 8 times the SDXL VAE's moment channels, and a FLUX decode's
    node and allocator show in its own bf16 lines, not in the SDXL VAE's."""
    values = shipped_coefficients(spec, platform).values
    sdxl = shipped_coefficients(VAE_SD_SPEC, platform).values
    scale = ELEMENT_SIZES[dtype] // ELEMENT_SIZES["bf16"]
    for line in ("tiled_decode", "tiled_encode", "assembled_decode"):
        for part in ("k", "b"):
            expected = max(sdxl[f"{part}_{line}_{dtype}"], values[f"{part}_{line}_bf16"] * scale)
            assert values[f"{part}_{line}_{dtype}"] == expected, f"{part}_{line}_{dtype}"


def test_expandable_segments_off_windows_are_priced_as_the_native_allocator_there(on) -> None:
    """Measured on Windows only, where they are Invoke's ROCm default. On Linux they are a user override nobody
    measured: an SDXL fp16 decode there must not take the Windows 2.5 GiB at 1024px, nor go under the pre-dtype 7200 MiB."""
    linux = estimate_vae_working_memory_sd15_sdxl(
        "decode", LATENTS_1024[:, :4], _vae(), None, False, on(Platform("miopen", "expandable", "linux"))
    )
    native = estimate_vae_working_memory_sd15_sdxl(
        "decode", LATENTS_1024[:, :4], _vae(), None, False, on(Platform("miopen", "native", "linux"))
    )
    assert linux.reservation_bytes == native.reservation_bytes >= 3600 * 2 * 1024 * 1024
    assert "priced as conv_backend=miopen,allocator=native" in linux.source


@pytest.mark.parametrize(
    "spec, estimate, vae_dtype, dtype",
    [
        pytest.param(
            VAE_SD_SPEC,
            lambda device, vae: estimate_vae_working_memory_sd15_sdxl("decode", LATENTS_1024, vae, None, False, device),
            torch.float16,
            "fp16",
            id="sd",
        ),
        pytest.param(
            VAE_SD3_SPEC,
            lambda device, vae: estimate_vae_working_memory_sd3("decode", LATENTS_1024, vae, device),
            torch.bfloat16,
            "bf16",
            id="sd3",
        ),
        pytest.param(
            VAE_COGVIEW4_SPEC,
            lambda device, vae: estimate_vae_working_memory_cogview4("decode", LATENTS_1024, vae, device),
            torch.bfloat16,
            "bf16",
            id="cogview4",
        ),
        pytest.param(
            VAE_FLUX_SPEC,
            lambda device, vae: estimate_vae_working_memory_flux("decode", LATENTS_1024, vae, device=device),
            torch.bfloat16,
            "bf16",
            id="flux",
        ),
        pytest.param(
            VAE_FLUX2_SPEC,
            lambda device, vae: estimate_vae_working_memory_flux2(
                "decode", torch.zeros(1, 32, 128, 128), vae, device=device
            ),
            torch.bfloat16,
            "bf16",
            id="flux2",
        ),
    ],
)
def test_a_planned_untiled_decode_is_priced_as_its_decode_node_prices_it(spec, estimate, vae_dtype, dtype) -> None:
    """The calibration plan sizes its cases without a VAE; its features must stay the decode nodes' own, the score
    matrix included (held to a stand-in that depends on its shape)."""
    device = torch.device("cuda", 0)
    with patch(
        "invokeai.backend.util.vae_working_memory.sdpa_score_matrix_bytes",
        side_effect=lambda **kwargs: kwargs["num_heads"] * kwargs["seq_len"] ** 2 * 3,
    ):
        cls = AutoencoderKLFlux2 if spec is VAE_FLUX2_SPEC else AutoencoderKL
        node = estimate(device, _vae(cls, vae_dtype))
        planned = estimate_untiled_vae_decode(spec.id, 1024, 1024, dtype, device)

    assert dict(planned.features) == dict(node.features)
    assert planned.reservation_bytes == node.reservation_bytes
