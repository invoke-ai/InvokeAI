"""scripts/calibrate_vae_tiling_working_memory.py: the operations it plans, the node's work it runs as a window, the
tile the VAE then runs and the estimate prices, and the point files it writes for the data builder.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). These run its operations on
tiny VAEs of the real classes on the CPU, in float32: the geometry that decides tiling is the class's, not the width's.
The plumbing it shares with the other calibration scripts is `scripts/working_memory/fresh_points.py`'s, tested in
`tests/test_fresh_points_working_memory.py`.
"""

import json
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.vae.z_image_latents_to_image import ZImageLatentsToImageInvocation
from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params
from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling
from invokeai.backend.working_memory.bounds import evaluate_point
from invokeai.backend.working_memory.coefficients import MeasuredPoint, shipped_data
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import fresh_window
from invokeai.backend.working_memory.spec import import_estimator_modules
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import main

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_vae_tiling_working_memory.py"
RELEASED_SDXL_VAE_CONFIG = (
    Path(__file__).parent / "model_identification/stripped_models/8e8a1892-a19c-433f-a13d-b502e62a814a/vae/config.json"
)
"""`stabilityai/stable-diffusion-xl-refiner-1.0`'s VAE config, which is the SDXL VAE's (`_name_or_path` sdxl-vae)."""
CPU = torch.device("cpu")
SIDE = 640
"""Above the 512px a 0 tile field resolves to for FLUX.1 and FLUX.2, so every tiled case below really tiles."""
TINY_SD_SAMPLE_SIZE = 128
"""The tiny SD VAE's `sample_size`: the tile an SD node's 0 field leaves it at."""


def _load_script():
    spec = spec_from_file_location("calibrate_vae_tiling_working_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
Point = script.Point


@pytest.mark.platform_independent
def test_the_grid_references_each_tile_untiled_and_tiles_the_sides_above_the_tile_the_vae_runs() -> None:
    """A side at or below the tile runs the single pass, which the untiled reference at the tile's side measures. SD's
    0 field leaves the SDXL VAE at its own 1024px, so its sentinel tiles only the sides above that."""
    points = script.plan(families=["flux", "sd"], sizes=[1024, 2048], tiles=[512, 1024, 0])

    flux = [(512, None), (1024, None), (1024, 512), (2048, 512), (2048, 1024), (1024, 0), (2048, 0)]
    sd = [(512, None), (1024, None), (1024, 512), (2048, 512), (2048, 1024), (2048, 0)]
    expected = [("flux", op, "bfloat16", *cell) for op in ("decode", "encode") for cell in flux]
    expected += [
        ("sd", op, dtype, *cell) for dtype in ("float16", "float32") for op in ("decode", "encode") for cell in sd
    ]
    assert [(p.family, p.operation, p.dtype, p.side, p.tile) for p in points] == expected


def test_the_sdxl_geometry_is_the_released_vae_config() -> None:
    released = json.loads(RELEASED_SDXL_VAE_CONFIG.read_text(encoding="utf-8"))

    assert {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in released.items()
        if not key.startswith("_")
    } == dict(script.SDXL_VAE_CONFIG)


@pytest.mark.parametrize("tile", [512, 1024, 0, 320])
@pytest.mark.parametrize("family", ["flux", "flux2", "sd"])
def test_the_applied_tile_is_what_the_nodes_helper_sets_on_the_real_vae(family: str, tile: int) -> None:
    """The plan and the summary name the tile the VAE runs: `scoped_vae_tiling` for FLUX.1 and FLUX.2, the SD nodes'
    `patch_vae_tiling_params` for a field above 0 and the VAE's own `sample_size` for 0."""
    vae = script.build_vae(family)
    if family != "sd":
        with scoped_vae_tiling(vae, tile):
            runs = vae.tile_sample_min_size
    elif tile > 0:
        with patch_vae_tiling_params(vae, tile_sample_min_size=tile, tile_overlap_factor=0.25):
            runs = vae.tile_sample_min_size
    else:
        runs = vae.tile_sample_min_size

    assert script.applied_tile(family, tile) == runs


def _tiny_vae(family: str) -> torch.nn.Module:
    """The family's VAE class with four blocks, hence the real 8x compression and tiling geometry, at width 4."""
    torch.manual_seed(0)
    tiny = {"block_out_channels": (4, 4, 4, 4), "layers_per_block": 1, "norm_num_groups": 2}
    if family == "flux2":
        return AutoencoderKLFlux2(**tiny, latent_channels=8).eval()
    blocks = {"down_block_types": ("DownEncoderBlock2D",) * 4, "up_block_types": ("UpDecoderBlock2D",) * 4}
    if family == "flux":
        flux = {"latent_channels": 16, "sample_size": 1024, "scaling_factor": 0.3611, "shift_factor": 0.1159}
        return AutoencoderKL(**tiny, **blocks, **flux, use_quant_conv=False, use_post_quant_conv=False).eval()
    sd = {"latent_channels": 4, "sample_size": TINY_SD_SAMPLE_SIZE, "scaling_factor": 0.13025}
    return AutoencoderKL(**tiny, **blocks, **sd).eval()


def _record_tiles(vae: torch.nn.Module, operation: str) -> list[int]:
    """The sides, in output pixels, of the tiles the VAE's encoder or decoder is run on."""
    seen: list[int] = []
    network = vae.decoder if operation == "decode" else vae.encoder
    scale = script.LATENT_SCALE_FACTOR if operation == "decode" else 1
    network.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[-1] * scale))
    return seen


@pytest.mark.parametrize("tile", [None, 256, 0], ids=["untiled", "tile256", "sentinel"])
@pytest.mark.parametrize("operation", ["decode", "encode"])
@pytest.mark.parametrize("family", ["flux", "flux2", "sd"])
def test_the_vae_runs_the_tile_its_node_sets_and_the_estimate_prices_that_tile(family, operation, tile) -> None:
    """The window's tiling setup is the node's: the field's tile, the 0 sentinel at 512px through `scoped_vae_tiling`
    and at the VAE's own `sample_size` for SD, and none untiled. The point's features, from the node's estimator
    call, price the tile that ran."""
    vae = _tiny_vae(family)
    point = Point(family, operation, "float32", SIDE, tile)
    seen = _record_tiles(vae, operation)
    tensor = script.host_input(point, vae)

    with torch.no_grad():
        script.operation(point, vae, tensor, CPU)()

    runs = {None: SIDE, 256: 256, 0: TINY_SD_SAMPLE_SIZE if family == "sd" else 512}[tile]
    assert max(seen) == runs
    assert script.estimate(point, vae, tensor, CPU).features["tile"] == (None if tile is None else runs)


def _z_image_context(vae: AutoencoderKL, latents: torch.Tensor) -> MagicMock:
    """What `z_image_l2i` reaches through its context, the VAE resident on the CPU and no tiling forced."""
    vae_info = MagicMock(model=vae, compute_device=CPU)
    vae_info.model_on_device.return_value.__enter__.return_value = (None, vae)
    context = MagicMock()
    context.tensors.load.return_value = latents
    context.models.load.return_value = vae_info
    context.config.get.return_value.force_tiled_decode = False
    context.config.get.return_value.auto_tiled_decode = False
    context.images.save.return_value = MagicMock(image_name="decoded.png", width=SIDE, height=SIDE)
    return context


@pytest.mark.parametrize("tile", [None, 256], ids=["untiled", "tiled"])
def test_the_flux_decode_window_makes_the_image_z_image_l2i_saves(tile) -> None:
    """The FLUX.1 decode point runs Z-Image's node through its post-processing: the image it returns is the one the
    node saves, pixel for pixel."""
    vae = _tiny_vae("flux")
    point = Point("flux", "decode", "float32", SIDE, tile)
    latents = script.host_input(point, vae)
    context = _z_image_context(vae, latents)

    ZImageLatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="latents"),
        vae=MagicMock(seamless_axes=[]),
        tiled=tile is not None,
        tile_size=tile or 0,
    ).invoke(context)
    with torch.no_grad():
        ours = np.asarray(script.operation(point, vae, latents, CPU)())

    saved = np.asarray(context.images.save.call_args.kwargs["image"])
    assert ours.shape == (SIDE, SIDE, 3) and saved.std() > 0
    np.testing.assert_array_equal(ours, saved)


class _Resident:
    """A loaded model's window with the VAE already resident on the CPU."""

    def __init__(self, vae: torch.nn.Module) -> None:
        self.model = vae
        self.compute_device = CPU

    @contextmanager
    def model_on_device(self, working_memory=None):
        yield None, self.model


@pytest.mark.parametrize("tile", [None, 256, 0], ids=["untiled", "tile256", "sentinel"])
def test_the_sd_encode_window_returns_the_latents_i2l_returns(monkeypatch: pytest.MonkeyPatch, tile) -> None:
    """The SD encode point runs `i2l`'s window, its tiling switch, patch and sampling included: with the same seed, the
    latents are the node's."""
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_torch_device", staticmethod(lambda: torch.device("cpu"))
    )
    vae = _tiny_vae("sd")
    point = Point("sd", "encode", "float32", SIDE, tile)
    image = script.host_input(point, vae)

    torch.manual_seed(1)
    node = ImageToLatentsInvocation.vae_encode(
        _Resident(vae), upcast=True, tiled=tile is not None, image_tensor=image, tile_size=tile or 0
    )
    torch.manual_seed(1)
    with torch.no_grad():
        ours = script.operation(point, vae, image, CPU)()

    assert ours.shape == (1, 4, SIDE // 8, SIDE // 8)
    torch.testing.assert_close(ours, node, rtol=0, atol=0)


def _reading(**values) -> CounterReading:
    fields = {
        "reserved": 1000,
        "allocated": 800,
        "peak_reserved": 1000,
        "peak_allocated": 800,
        "num_ooms": 0,
        "num_alloc_retries": 0,
        "driver": 1200,
        "peak_driver": None,
        "paged": 0,
        "device_used": None,
        "method": "windows-d3dkmt",
    }
    return CounterReading(**(fields | values))


CUDNN_NATIVE = {"conv_backend": "cudnn", "allocator": "native", "os": "windows"}
TILED = [
    Point("flux", "decode", "bfloat16", 2048, 512),
    Point("flux2", "encode", "bfloat16", 4096, 0),
    Point("sd", "encode", "float16", 3072, 1024),
]
"""Tiled points of each family, their features as a CPU prices them (an SD encode runs fp16 there as on a GPU)."""


def _record(point, demand_per_reserved: float) -> dict:
    """The file record of `point` as a measuring process on an RTX 4090 writes it, its demand this fraction of what
    the shipped data reserves for it there."""
    vae = script.build_vae(point.family).to(getattr(torch, point.dtype))
    side = point.side // script.LATENT_SCALE_FACTOR if point.operation == "decode" else point.side
    channels = vae.config.latent_channels if point.operation == "decode" else 3
    estimate = script.estimate(point, vae, torch.empty(1, channels, side, side, device="meta"), CPU)
    specs = import_estimator_modules()
    draft = MeasuredPoint(
        estimator=estimate.estimator,
        version=estimate.version,
        platform=CUDNN_NATIVE,
        features=dict(estimate.features),
        measured_bytes=1,
        method="demand_peak",
        device="NVIDIA GeForce RTX 4090",
        provenance="test",
    )
    reserved = evaluate_point(draft, specs[estimate.estimator], shipped_data().values()).reservation_bytes
    demand = int(reserved * demand_per_reserved)
    window = fresh_window(
        _reading(), _reading(peak_reserved=1000 + demand, driver=1200 + demand), oom=False, spill=False
    )
    common = {key: getattr(draft, key) for key in ("estimator", "version", "platform", "features", "device")}
    line = fresh_points.point_line(point, common=common, provenance="test", windows=[window, window], raw={})
    record = fresh_points.file_record(line)
    assert record is not None
    return record


def _build(tmp_path: Path, records: list[dict]) -> int:
    point_file = tmp_path / "tiled.jsonl"
    point_file.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return main(["--points", str(point_file)])


def test_the_builder_holds_a_runs_tiled_points_to_both_bounds(tmp_path: Path, capsys) -> None:
    """A tiled point is in the fit of its estimator's tiled line, so it caps it: within 1.3x its points join the line,
    beyond it the build is refused."""
    assert _build(tmp_path, [_record(point, 1 / 1.1) for point in TILED]) == 0
    assert f"{len(TILED)} fresh-process points from 1 point files" in capsys.readouterr().out

    assert _build(tmp_path, [_record(point, 1 / 1.6) for point in TILED]) == 1
    assert "-tile512-NVIDIAGeForceRTX4090-demand_peak: reserves 1.60x" in capsys.readouterr().err


def test_a_tiled_point_above_its_estimate_raises_its_estimators_tiled_line(tmp_path: Path, capsys) -> None:
    """The untiled line beneath stays: the tiled line covers what the tile does not. An encode, whose tiled line is its
    only one: a decode's point may raise its assembly line instead. The review of the refit says which line moved;
    whether the build then writes depends on how close the line's other points sit to their bounds, which this does
    not test (`test_the_builder_holds_a_runs_tiled_points_to_both_bounds` does)."""
    _build(tmp_path, [_record(TILED[2], 1.1)])
    out = capsys.readouterr().out
    assert "line vae.json: vae.sd {'conv_backend': 'cudnn', 'allocator': 'native'} tiled_encode_fp16:" in out
    assert "raised" in out and " encode_fp16:" not in out
