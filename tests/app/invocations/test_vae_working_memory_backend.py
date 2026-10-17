"""Which convolution line the diffusers-AutoencoderKL estimators price, and what the SD estimator adds up.

SD1/SDXL, SD3, FLUX.1 and FLUX.2 share measured lines per operation, dtype and platform (convolution backend,
allocator, and on MIOpen's native allocator the OS); CogView4 carries its own. The selection is invisible on whichever
machine runs the suite -- on the RX 9060 XT an fp16 SDXL decode costs a third of a bf16 one, and the SD nodes run fp16
whatever dtype the cached VAE holds -- so it is pinned here rather than left to the card.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.vae.latents_to_image import LatentsToImageInvocation
from invokeai.backend.util.vae_working_memory import (
    VAE_COGVIEW4_SPEC,
    VAE_FLUX2_SPEC,
    VAE_FLUX_SPEC,
    VAE_SD3_SPEC,
    VAE_SD_SPEC,
    estimate_vae_working_memory_cogview4,
    estimate_vae_working_memory_flux,
    estimate_vae_working_memory_flux2,
    estimate_vae_working_memory_sd3,
    estimate_vae_working_memory_sd15_sdxl,
    sd_vae_run_dtype,
)
from invokeai.backend.working_memory import platform as platform_module
from invokeai.backend.working_memory.coefficients import shipped_coefficients, shipped_data
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
    """The cached VAE holds whatever the previous run cast it to; the node casts it again before it runs."""
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


def _vae_info(vae: AutoencoderKL, estimates: list) -> MagicMock:
    def model_on_device(working_memory):
        estimates.append(working_memory)
        cm = MagicMock()
        cm.__enter__.return_value = (None, vae)
        cm.__exit__.return_value = None
        return cm

    vae_info = MagicMock(model=vae, compute_device=torch.device("cpu"))
    vae_info.model_on_device.side_effect = model_on_device
    return vae_info


@pytest.mark.parametrize("fp32", [False, True])
def test_the_decode_node_casts_the_vae_to_the_dtype_its_estimate_priced(fp32) -> None:
    """On the CPU both requests decode in fp32: the node and the estimate share `sd_vae_run_dtype`."""
    vae, estimates = _tiny_sd_vae().to(torch.bfloat16), []
    context = MagicMock()
    context.tensors.load.return_value = torch.zeros(1, 4, 8, 8)
    context.config.get.return_value.force_tiled_decode = False
    context.models.load.return_value = _vae_info(vae, estimates)
    context.images.save.return_value = MagicMock(image_name="x.png", width=8, height=8)
    node = LatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="x"), vae=MagicMock(seamless_axes=[]), tiled=False, tile_size=0, fp32=fp32
    )

    node.invoke(context)

    (estimate,) = estimates
    assert estimate.features["dtype"] == "fp32"
    assert {p.dtype for p in vae.parameters()} == {torch.float32}


@pytest.mark.parametrize("upcast, run_dtype", [(False, torch.float16), (True, torch.float32)])
def test_the_encode_node_casts_the_vae_to_the_dtype_its_estimate_priced(upcast, run_dtype) -> None:
    vae, estimates, encoded_in = _tiny_sd_vae().to(torch.bfloat16), [], []

    def encode(vae, image):
        encoded_in.append(next(vae.encoder.parameters()).dtype)
        return torch.zeros(1, 4, 2, 2)

    with patch.object(ImageToLatentsInvocation, "_encode_to_tensor", side_effect=encode):
        ImageToLatentsInvocation.vae_encode(_vae_info(vae, estimates), upcast, False, torch.zeros(1, 3, 16, 16))

    (estimate,) = estimates
    assert encoded_in == [run_dtype]
    assert estimate.features["dtype"] == {torch.float16: "fp16", torch.float32: "fp32"}[run_dtype]


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


@pytest.mark.parametrize("fp32, run_dtype, growth", [(False, "fp16", 0), (True, "fp32", 1000 * 2)])
def test_a_tiled_sd_decode_is_one_tile_with_its_margin_the_base_once_and_the_upcast(
    on, fp32, run_dtype, growth
) -> None:
    platform = Platform("miopen", "expandable", "windows")
    result = estimate_vae_working_memory_sd15_sdxl(
        "decode", torch.zeros(1, 4, 192, 192), _vae(numel=1000), 512, fp32, on(platform)
    )
    assert result.reservation_bytes == int(_line(VAE_SD_SPEC, platform, "decode", run_dtype, 512 * 512 * 1.25) + growth)


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


def test_every_stand_in_slope_is_at_least_the_pre_dtype_figure() -> None:
    """A stand-in is a declared conservative value on a measured platform; nothing else guards it."""
    stand_ins = [entry for data in shipped_data().values() for entry in data.coefficients if entry.stand_ins]
    assert stand_ins, "the shipped data declares no stand-ins -- the check is broken"
    for entry in stand_ins:
        for name in entry.stand_ins:
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
        part, operation, dtype = name.split("_")
        if part == "k":
            floor = PRE_DTYPE_BYTES_PER_PIXEL_PER_ELEMENT_BYTE[measured.conv_backend][operation] * ELEMENT_SIZES[dtype]
            assert value >= floor, f"{name} on {platform}: {value} under {floor}"
        else:
            assert value >= 0


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
