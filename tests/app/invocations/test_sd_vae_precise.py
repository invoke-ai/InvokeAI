"""SD1.5/SDXL `l2i` and `i2l` with precise image storage, on CPU with a tiny real `AutoencoderKL`."""

import copy
import itertools
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

import invokeai.app.invocations.vae.latents_to_image as l2i_module
import invokeai.backend.image_util.raster.precise_asset as precise_asset
from invokeai.app.invocations.vae.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.vae.latents_to_image import LatentsToImageInvocation
from invokeai.app.invocations.vae.precise_io import save_decoded_image
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.startup import init_database
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
    raster_resized_to_grid_as_tensor,
)
from invokeai.backend.util.logging import InvokeAILogger

SCALING_FACTOR = 0.18215


def _descriptor(channels: RasterChannels, **overrides) -> RasterDescriptor:
    has_alpha = channels in (RasterChannels.LA, RasterChannels.RGBA)
    fields = {
        "channels": channels,
        "color": ColorEncoding.SRGB,
        "value_range": ValueRange.UNIT,
        "alpha": AlphaMode.STRAIGHT if has_alpha else AlphaMode.NONE,
        "channel_count": len(channels.value),
    }
    fields.update(overrides)
    return RasterDescriptor(**fields)


def _tiny_vae(overshoot: float = 1.0) -> AutoencoderKL:
    """A real SD-shaped VAE (two blocks: 2x spatial compression, 4 latent channels) with random weights.

    `overshoot` scales the decoder's last conv so its output leaves [-1, 1] and the unclamped path has something to keep.
    """
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        scaling_factor=SCALING_FACTOR,
    ).eval()
    with torch.no_grad():
        vae.decoder.conv_out.weight.mul_(overshoot)
        vae.decoder.conv_out.bias.add_(0.1 * overshoot)
    return vae


def _loaded(vae: AutoencoderKL, base: BaseModelType = BaseModelType.StableDiffusion1) -> MagicMock:
    info = MagicMock()
    info.model = vae
    info.compute_device = torch.device("cpu")
    info.config.base = base

    @contextmanager
    def model_on_device(working_memory=None):
        yield (None, vae)

    info.model_on_device = model_on_device
    return info


@pytest.fixture
def world(tmp_path: Path):
    """Real image service, records and PNG storage behind a context whose model, tensor and config parts are small."""
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    config._root = tmp_path
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    invoker.services.configuration.image_subfolder_strategy = "flat"
    invoker.services.configuration.multiuser = False
    invoker.services.configuration.precise_image_storage = True
    storage.start(invoker)
    db = init_database(config=config, logger=InvokeAILogger.get_logger(), image_files=storage)
    counter = itertools.count()
    invoker.services.names.create_image_name.side_effect = lambda: f"image-{next(counter)}.png"
    invoker.services.database = db
    invoker.services.image_records = ImageRecordStorage(db)
    invoker.services.image_files = storage
    invoker.services.board_image_records.get_board_for_image.return_value = None
    invoker.services.urls.get_image_url.return_value = "url"
    invoker.services.logger = MagicMock()
    service = ImageService()
    service.start(invoker)
    invoker.services.images = service
    data = SimpleNamespace(
        queue_item=SimpleNamespace(
            user_id=None,
            project_id=None,
            session_id=None,
            session=SimpleNamespace(graph=None),
            workflow=None,
            root_item_id=None,
            _workflow_json_loaded=False,
            _workflow_json_snapshot=None,
        ),
        invocation=SimpleNamespace(is_intermediate=False, id="node"),
    )
    images = ImagesInterface(invoker.services, data, MagicMock())

    saved_tensors: dict[str, torch.Tensor] = {}

    def save_tensor(tensor: torch.Tensor) -> str:
        name = f"tensor-{len(saved_tensors)}"
        saved_tensors[name] = tensor
        return name

    vae_holder: dict[str, MagicMock] = {}
    context = SimpleNamespace(
        images=images,
        tensors=SimpleNamespace(load=lambda name: saved_tensors[name].clone(), save=save_tensor),
        models=SimpleNamespace(load=lambda _field: vae_holder["info"]),
        util=MagicMock(),
        config=SimpleNamespace(get=lambda: SimpleNamespace(force_tiled_decode=False)),
    )
    return SimpleNamespace(
        service=service,
        database=db,
        storage=storage,
        images=images,
        context=context,
        configuration=invoker.services.configuration,
        tensors=saved_tensors,
        vae_holder=vae_holder,
        use_vae=lambda info: vae_holder.__setitem__("info", info),
    )


def _decode_node(**fields) -> LatentsToImageInvocation:
    return LatentsToImageInvocation.model_construct(
        id="l2i",
        latents=SimpleNamespace(latents_name="latents"),
        vae=SimpleNamespace(vae="vae", seamless_axes=fields.pop("seamless_axes", [])),
        tiled=fields.pop("tiled", False),
        tile_size=fields.pop("tile_size", 0),
        fp32=fields.pop("fp32", False),
        **fields,
    )


def _latents(seed: int = 1, size: int = 8) -> torch.Tensor:
    return torch.randn(1, 4, size, size, generator=torch.Generator().manual_seed(seed)) * 2.0


def _legacy_decode(vae: AutoencoderKL, latents: torch.Tensor) -> tuple[torch.Tensor, Image.Image]:
    """The pre-change decode body: the unit-range decoder output and the clamped PIL image made from it."""
    with torch.no_grad():
        decoded = vae.decode(latents / SCALING_FACTOR, return_dict=False)[0]
    unit = decoded / 2 + 0.5
    legacy = VaeImageProcessor.numpy_to_pil(unit.clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy())[0]
    return unit, legacy


def _stored_bytes(world, image_name: str) -> bytes:
    return world.storage.get_path(image_name).read_bytes()


def _run_decode(world, vae: AutoencoderKL, node: LatentsToImageInvocation, latents: torch.Tensor):
    world.use_vae(_loaded(vae))
    world.tensors["latents"] = latents
    return node.invoke(world.context).image.image_name


# --- decode ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("overshoot", [1.0, 8.0])
def test_decode_with_precise_storage_disabled_saves_the_legacy_png_and_no_precise_asset(world, overshoot) -> None:
    vae = _tiny_vae(overshoot)
    latents = _latents()
    world.configuration.precise_image_storage = False
    _, legacy = _legacy_decode(vae, latents)

    name = _run_decode(world, vae, _decode_node(fp32=True), latents)

    expected_name = world.images.save(legacy).image_name
    assert _stored_bytes(world, name) == _stored_bytes(world, expected_name)
    assert world.images.get_raster(name).dtype == np.uint8


@pytest.mark.parametrize("fp32", [True, False])
def test_precise_decode_keeps_the_unclamped_output_and_the_gallery_png_stays_legacy(world, fp32: bool) -> None:
    vae = _tiny_vae(overshoot=8.0)
    latents = _latents()
    unit, legacy = _legacy_decode(vae, latents)
    # The setup must actually overshoot, or the unclamped path would be indistinguishable from the clamped one.
    assert unit.min() < 0.0 and unit.max() > 1.0

    # The CPU decodes in fp32 whatever `fp32` says (the run dtype is a helper-level rule, tested below).
    name = _run_decode(world, vae, _decode_node(fp32=fp32), latents)

    raster = world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == _descriptor(RasterChannels.RGB)
    expected = unit[0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(raster.view(), expected)
    assert raster.view().min() < 0.0 and raster.view().max() > 1.0
    world.configuration.precise_image_storage = False
    assert _stored_bytes(world, name) == _stored_bytes(world, world.images.save(legacy).image_name)


@pytest.mark.parametrize("poisoned", ["all", "one-channel"])
def test_non_finite_decoder_output_falls_back_to_the_ordinary_png_and_warns(world, poisoned: str) -> None:
    vae = _tiny_vae()
    with torch.no_grad():
        if poisoned == "all":
            vae.decoder.conv_out.bias.fill_(float("nan"))
        else:
            vae.decoder.conv_out.bias[1] = float("nan")
    latents = _latents()
    world.context.logger = MagicMock()

    name = _run_decode(world, vae, _decode_node(fp32=True), latents)
    world.configuration.precise_image_storage = False
    disabled = _run_decode(world, vae, _decode_node(fp32=True), latents)

    assert _stored_bytes(world, name) == _stored_bytes(world, disabled)
    assert world.images.get_raster(name).dtype == np.uint8
    assert world.database.queries.image_precise_assets.get(name) is None
    assert not any(path.suffix == ".safetensors" for path in world.storage.get_path(name).parent.rglob("*"))
    world.context.logger.warning.assert_called_once()
    assert "not finite" in world.context.logger.warning.call_args.args[0]


def test_a_decode_outside_the_precise_bounds_falls_back_to_the_ordinary_png_and_warns(world, monkeypatch) -> None:
    vae = _tiny_vae(overshoot=8.0)
    latents = _latents()
    world.context.logger = MagicMock()
    # The asset bound is 16384 per side; a smaller one stands in for an oversized decode (the image is 16x16).
    monkeypatch.setattr(precise_asset, "MAX_DIMENSION", 8)

    name = _run_decode(world, vae, _decode_node(fp32=True), latents)
    monkeypatch.undo()
    world.configuration.precise_image_storage = False
    disabled = _run_decode(world, vae, _decode_node(fp32=True), latents)

    assert _stored_bytes(world, name) == _stored_bytes(world, disabled)
    assert world.database.queries.image_precise_assets.get(name) is None
    world.context.logger.warning.assert_called_once()
    assert "Precise image storage skipped" in world.context.logger.warning.call_args.args[0]


def test_the_gallery_png_of_a_precise_decode_is_the_legacy_image_not_the_quantized_raster(world) -> None:
    # A value that quantize_unit rounds up but a deliberately different legacy image does not: the policy must pass
    # the legacy image on, whatever the raster would quantize to.
    raster = RasterImage.from_array(np.full((4, 4, 3), 0.9, np.float32), _descriptor(RasterChannels.RGB))
    legacy = Image.fromarray(np.full((4, 4, 3), 7, np.uint8), "RGB")

    dto = save_decoded_image(world.context, legacy, raster)

    with Image.open(world.storage.get_path(dto.image_name)) as png:
        assert png.tobytes() == legacy.tobytes()
    np.testing.assert_array_equal(world.images.get_raster(dto.image_name).view(), raster.view())


def test_fp16_decode_stores_a_float16_raster_equal_to_the_fp16_reference(world, monkeypatch) -> None:
    vae = _tiny_vae(overshoot=8.0)
    latents = _latents()
    reference_vae = copy.deepcopy(vae).half()
    with torch.no_grad():
        decoded = reference_vae.decode(latents.half() / SCALING_FACTOR, return_dict=False)[0]
    unit = decoded / 2 + 0.5
    assert unit.dtype == torch.float16 and (unit.min() < 0.0 or unit.max() > 1.0)
    # The CPU forces fp32 for a decode; the dtype rule itself is the helper's, so pin it to what a GPU run picks.
    monkeypatch.setattr(l2i_module, "sd_vae_run_dtype", lambda *_args: torch.float16)

    name = _run_decode(world, vae, _decode_node(fp32=False), latents)

    raster = world.images.get_raster(name)
    assert raster.dtype == np.float16
    np.testing.assert_array_equal(raster.view(), unit[0].permute(1, 2, 0).numpy())


@pytest.mark.parametrize(
    "variant",
    [
        {"tiled": True, "tile_size": 16},
        {"seamless_axes": ["x", "y"]},
        {"tiled": True, "tile_size": 16, "seamless_axes": ["x"]},
    ],
    ids=["tiled", "seamless", "tiled+seamless"],
)
def test_tiled_and_seamless_decode_work_in_both_modes(world, variant) -> None:
    vae = _tiny_vae(overshoot=8.0)
    latents = _latents(size=16)

    precise = _run_decode(world, vae, _decode_node(fp32=True, **variant), latents)
    world.configuration.precise_image_storage = False
    disabled = _run_decode(world, vae, _decode_node(fp32=True, **variant), latents)

    stored = world.images.get_raster(precise)
    assert (stored.width, stored.height, stored.dtype) == (32, 32, np.float32)
    assert stored.view().min() < 0.0 or stored.view().max() > 1.0
    assert world.images.get_raster(disabled).dtype == np.uint8
    assert _stored_bytes(world, precise) == _stored_bytes(world, disabled)


# --- encode ---------------------------------------------------------------------------------------------------------


def _encode(world, vae: AutoencoderKL, image_name: str, seed: int = 11, **fields) -> torch.Tensor:
    world.use_vae(_loaded(vae, fields.pop("base", BaseModelType.StableDiffusion1)))
    node = ImageToLatentsInvocation.model_construct(
        id="i2l",
        image=SimpleNamespace(image_name=image_name),
        vae=SimpleNamespace(vae="vae"),
        tiled=fields.pop("tiled", False),
        tile_size=fields.pop("tile_size", 0),
        fp32=fields.pop("fp32", True),
        color_compensation=fields.pop("color_compensation", "None"),
    )
    torch.manual_seed(seed)  # the latent distribution is sampled
    return world.tensors[node.invoke(world.context).latents.latents_name]


def _reference_latents(
    vae: AutoencoderKL, chw_unit: torch.Tensor, seed: int = 11, tiled: bool = False, tile_size: int = 0
) -> torch.Tensor:
    """Encodes an explicitly built [0, 1] CHW tensor through the same VAE entry point, seeded identically."""
    torch.manual_seed(seed)
    return ImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae),
        upcast=True,
        tiled=tiled,
        image_tensor=(chw_unit * 2.0 - 1.0)[None],
        tile_size=tile_size,
    )


def _store_png(world, array: np.ndarray, mode: str) -> str:
    image = Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array, mode)
    return world.service.create(image, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name


def _store_precise(world, array: np.ndarray, channels: RasterChannels) -> str:
    return world.images.save_raster(RasterImage.from_array(array, _descriptor(channels))).image_name


def _chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _noise(shape: tuple[int, ...], dtype, low=0.0, high=1.0, seed=3) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, size=shape).astype(dtype)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_a_png_backed_source_encodes_exactly_as_the_eight_bit_path_did(world, mode: str) -> None:
    vae = _tiny_vae()
    pixels = np.random.default_rng(2).integers(0, 256, size=(16, 24, len(mode)), dtype=np.uint8)
    name = _store_png(world, pixels, mode)
    rgb = {
        "RGB": pixels,
        "RGBA": pixels[:, :, :3],
        "L": np.repeat(pixels, 3, axis=2),
        "LA": np.repeat(pixels[:, :, :1], 3, axis=2),
    }[mode]

    latents = _encode(world, vae, name)

    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, _chw(rgb) / 255.0).numpy())


def test_a_non_multiple_of_8_png_keeps_the_lanczos_resize_of_the_old_path(world) -> None:
    vae = _tiny_vae()
    pixels = np.random.default_rng(2).integers(0, 256, size=(19, 21, 3), dtype=np.uint8)
    name = _store_png(world, pixels, "RGB")
    from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

    with world.service.get_pil_image(name).copy() as pil:
        old = image_resized_to_grid_as_tensor(pil.convert("RGB"))
    torch.manual_seed(11)
    expected = ImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae), upcast=True, tiled=False, image_tensor=old[None], tile_size=0
    )

    np.testing.assert_array_equal(_encode(world, vae, name).numpy(), expected.numpy())
    assert expected.shape[-2:] == (8, 8)


@pytest.mark.parametrize(
    ("dtype", "channels", "to_unit"),
    [
        (np.float32, RasterChannels.RGB, lambda a: a),
        (np.float16, RasterChannels.RGB, lambda a: a.astype(np.float32)),
        (np.uint16, RasterChannels.RGB, lambda a: a.astype(np.float32) / 65535.0),
        (np.float32, RasterChannels.RGBA, lambda a: a[:, :, :3]),
        (np.uint16, RasterChannels.RGBA, lambda a: a[:, :, :3].astype(np.float32) / 65535.0),
        (np.float32, RasterChannels.LA, lambda a: np.repeat(a[:, :, :1], 3, axis=2)),
        (np.float16, RasterChannels.L, lambda a: np.repeat(a.astype(np.float32), 3, axis=2)),
    ],
)
def test_a_precise_source_encodes_its_unclamped_values(world, dtype, channels: RasterChannels, to_unit) -> None:
    vae = _tiny_vae()
    count = len(channels.value)
    if dtype == np.uint16:
        array = np.random.default_rng(4).integers(0, 65536, size=(16, 24, count), dtype=np.uint16)
    else:
        array = _noise((16, 24, count), dtype, low=-0.4, high=1.6)
    name = _store_precise(world, array, channels)

    latents = _encode(world, vae, name)

    expected = _reference_latents(vae, _chw(to_unit(array)))
    np.testing.assert_array_equal(latents.numpy(), expected.numpy())


def test_a_precise_source_differs_from_encoding_its_eight_bit_png(world) -> None:
    vae = _tiny_vae()
    array = _noise((16, 16, 3), np.float32, low=-0.5, high=1.5)
    precise_name = _store_precise(world, array, RasterChannels.RGB)
    with world.images.get_pil(precise_name) as derivative:
        png_name = world.service.create(derivative, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name

    from_png = _encode(world, vae, png_name)
    from_float = _encode(world, vae, precise_name)

    assert not torch.allclose(from_png, from_float)
    # The PNG derivative clips to [0, 1]: a clamped float tensor reproduces it, the unclamped one does not.
    clamped = _reference_latents(vae, _chw(np.clip(array, 0, 1)))
    assert torch.allclose(from_png, clamped, atol=0.05)


def test_a_precise_source_that_is_not_a_multiple_of_8_is_trimmed_and_bicubic_resized(world) -> None:
    vae = _tiny_vae()
    array = _noise((43, 35, 3), np.float32, low=-0.3, high=1.3)  # 35x43 -> 32x40
    name = _store_precise(world, array, RasterChannels.RGB)

    latents = _encode(world, vae, name)

    resized = F.interpolate(_chw(array)[None], size=(40, 32), mode="bicubic", antialias=True, align_corners=False)[0]
    np.testing.assert_array_equal(latents.numpy(), _reference_latents(vae, resized).numpy())
    assert latents.shape == (1, 4, 20, 16)


def test_color_compensation_applies_to_a_precise_source(world) -> None:
    vae = _tiny_vae()
    array = _noise((16, 16, 3), np.float32, low=-0.2, high=1.2)
    name = _store_precise(world, array, RasterChannels.RGB)

    latents = _encode(world, vae, name, color_compensation="SDXL", base=BaseModelType.StableDiffusionXL)

    scaled = (_chw(array) * 2.0 - 1.0) * 1.015 - 0.002
    torch.manual_seed(11)
    expected = ImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae), upcast=True, tiled=False, image_tensor=scaled[None], tile_size=0
    )
    np.testing.assert_array_equal(latents.numpy(), expected.numpy())


@pytest.mark.parametrize(
    "overrides",
    [
        {"color": ColorEncoding.LINEAR_SRGB},
        {"value_range": ValueRange.UNBOUNDED},
        {"alpha": AlphaMode.PREMULTIPLIED},
    ],
    ids=["linear", "unbounded", "premultiplied"],
)
def test_a_source_without_a_display_policy_is_refused(overrides) -> None:
    channels = RasterChannels.RGBA if "alpha" in overrides else RasterChannels.RGB
    raster = RasterImage.from_array(
        np.zeros((8, 8, len(channels.value)), np.float32), _descriptor(channels, **overrides)
    )

    with pytest.raises(ValueError):
        raster_resized_to_grid_as_tensor(raster)


def test_a_data_source_is_refused() -> None:
    descriptor = RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNBOUNDED, AlphaMode.NONE, 3)
    raster = RasterImage.from_array(np.zeros((8, 8, 3), np.float32), descriptor)

    with pytest.raises(ValueError):
        raster_resized_to_grid_as_tensor(raster)


@pytest.mark.parametrize("source", ["png", "precise"])
def test_tiled_encode_matches_the_reference_for_both_source_kinds(world, source: str) -> None:
    vae = _tiny_vae()
    if source == "png":
        pixels = np.random.default_rng(6).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        name = _store_png(world, pixels, "RGB")
        unit = _chw(pixels) / 255.0
    else:
        array = _noise((32, 32, 3), np.float32, low=-0.4, high=1.4)
        name = _store_precise(world, array, RasterChannels.RGB)
        unit = _chw(array)

    latents = _encode(world, vae, name, tiled=True, tile_size=16)

    expected = _reference_latents(vae, unit, tiled=True, tile_size=16)
    assert latents.shape == (1, 4, 16, 16)
    np.testing.assert_array_equal(latents.numpy(), expected.numpy())


@pytest.mark.parametrize("transparency", [None, 3000])
def test_a_sixteen_bit_gray_png_upload_encodes_exactly_as_the_old_path(world, transparency) -> None:
    """16-bit gray (with or without a color key) has no precise asset, so it never changes latents."""
    vae = _tiny_vae()
    pixels = np.random.default_rng(8).integers(0, 65536, size=(16, 24), dtype=np.uint16)
    source = Image.fromarray(pixels)
    if transparency is not None:
        source.info["transparency"] = transparency
    name = world.service.create(source, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name
    assert not world.images.has_precise_asset(name)

    with world.service.get_pil_image(name).copy() as stored:
        old = image_resized_to_grid_as_tensor(stored.convert("RGB"))
    torch.manual_seed(11)
    expected = ImageToLatentsInvocation.vae_encode(
        vae_info=_loaded(vae), upcast=True, tiled=False, image_tensor=old[None], tile_size=0
    )

    np.testing.assert_array_equal(_encode(world, vae, name).numpy(), expected.numpy())


def test_precise_decode_then_encode_round_trips_through_the_gallery(world) -> None:
    """A precise decode output feeds the next node's encode with its unclamped values, not the 8-bit PNG."""
    vae = _tiny_vae(overshoot=8.0)
    unit, _ = _legacy_decode(vae, _latents())
    name = _run_decode(world, vae, _decode_node(fp32=True), _latents())

    latents = _encode(world, vae, name)

    expected = _reference_latents(vae, unit[0])
    np.testing.assert_array_equal(latents.numpy(), expected.numpy())


@pytest.mark.parametrize("precise", [False, True])
def test_no_decoder_output_tensor_outlives_the_model_block(world, monkeypatch, precise: bool) -> None:
    """The cache is emptied and the image saved after the VAE block; the decoder output and its unit-range image must
    be gone by then in both modes, or they stay on the device through the save. Tracked by weak references (no heap
    scan): the tensors die by reference count the moment the node drops its last name for them."""
    vae = _tiny_vae(overshoot=8.0)
    tracked: list[weakref.ref] = []
    real_decode = vae.decode

    def spy_decode(*args, **kwargs):
        result = real_decode(*args, **kwargs)
        tracked.append(weakref.ref(result[0]))
        return result

    monkeypatch.setattr(vae, "decode", spy_decode)
    real_raster = l2i_module.precise_decode_raster

    def spy_raster(context, unit_image):
        tracked.append(weakref.ref(unit_image))
        return real_raster(context, unit_image)

    monkeypatch.setattr(l2i_module, "precise_decode_raster", spy_raster)
    real_save = l2i_module.save_decoded_image
    alive: dict[str, list[bool]] = {"empty_cache": [], "save": []}

    def spy_save(*args, **kwargs):
        alive["save"] = [ref() is not None for ref in tracked]
        return real_save(*args, **kwargs)

    monkeypatch.setattr(l2i_module, "save_decoded_image", spy_save)
    monkeypatch.setattr(
        l2i_module.TorchDevice,
        "empty_cache",
        staticmethod(lambda: alive.__setitem__("empty_cache", [ref() is not None for ref in tracked])),
    )
    world.configuration.precise_image_storage = precise

    _run_decode(world, vae, _decode_node(fp32=True), _latents())

    assert len(tracked) == 2
    assert alive["empty_cache"] == [False, False]
    assert alive["save"] == [False, False]
