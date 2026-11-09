"""ImageChannelInvocation reads and writes rasters; its output must stay what PIL's `getchannel` produced."""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.image import ImageChannelInvocation
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.startup import init_database
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.backend.image_util.raster.convert import raster_to_pil
from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.precise_asset import (
    PRECISE_ASSET_FORMAT,
    PRECISE_ASSET_VERSION,
    write_precise_asset,
)
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def stack(tmp_path: Path):
    """Real image service, records and PNG storage; the node only talks to `context.images`."""
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    config._root = tmp_path
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    invoker.services.configuration.image_subfolder_strategy = "flat"
    invoker.services.configuration.multiuser = False
    invoker.services.configuration.precise_image_storage = False
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
    return SimpleNamespace(
        service=service,
        storage=storage,
        database=db,
        images=images,
        context=SimpleNamespace(images=images),
        configuration=invoker.services.configuration,
    )


def _noise(mode: str, seed: int) -> Image.Image:
    channels = len(mode)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(7, 5, channels), dtype=np.uint8)
    return Image.fromarray(array[:, :, 0] if channels == 1 else array, mode)


def _run(stack, mode: str, channel: str) -> tuple[str, str] | type[ValueError]:
    """Stored output file names of the node and of the PIL implementation it replaced, or the error type."""
    with _noise(mode, seed=3) as source:
        name = stack.service.create(source, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name
    node = ImageChannelInvocation(id="channel", image={"image_name": name}, channel=channel)

    with stack.service.get_pil_image(name).copy() as pil:
        try:
            expected = pil.getchannel(channel)
        except ValueError:
            with pytest.raises(ValueError):
                node.invoke(stack.context)
            return ValueError
    old_name = stack.images.save(expected).image_name
    new_name = node.invoke(stack.context).image.image_name
    return old_name, new_name


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("channel", ["A", "R", "G", "B"])
def test_channel_output_is_byte_identical_to_pil_getchannel(stack, mode: str, channel: str) -> None:
    outcome = _run(stack, mode, channel)

    if outcome is ValueError:
        assert (mode, channel) in {
            ("L", "A"),
            ("L", "R"),
            ("L", "G"),
            ("L", "B"),
            ("LA", "R"),
            ("LA", "G"),
            ("LA", "B"),
            ("RGB", "A"),
        }
        return
    old_name, new_name = outcome
    assert stack.storage.get_path(new_name).read_bytes() == stack.storage.get_path(old_name).read_bytes()
    with Image.open(stack.storage.get_path(new_name)) as result:
        assert result.mode == "L"


def test_channel_output_does_not_inherit_the_sources_icc_profile() -> None:
    """The gallery writer drops ICC today, so the stored file cannot show it; the raster handed over must be clean."""
    descriptor = RasterDescriptor(
        RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3, icc_profile=b"rgb-profile"
    )
    source = RasterImage.from_array(np.arange(12, dtype=np.uint8).reshape(2, 2, 3), descriptor)
    saved: list[RasterImage] = []
    images = SimpleNamespace(
        get_raster=lambda name: source,
        save_raster=lambda raster: saved.append(raster) or SimpleNamespace(image_name="out.png", width=2, height=2),
    )

    ImageChannelInvocation(id="channel", image={"image_name": "in.png"}, channel="G").invoke(
        SimpleNamespace(images=images)
    )

    assert saved[0].descriptor.icc_profile is None
    assert saved[0].view()[:, :, 0].tolist() == [[1, 4], [7, 10]]
    assert "icc_profile" not in raster_to_pil(saved[0]).info


def _store_precise(stack, raster: RasterImage) -> str:
    """A gallery image whose PNG derivative differs from `raster`, plus the precise asset and its row."""
    with Image.new("RGB", (raster.width, raster.height), (5, 5, 5)) as derivative:
        name = stack.service.create(derivative, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name
    path = stack.storage.get_precise_path(name)
    write_precise_asset(raster, path)
    with stack.database.queries.transaction() as q:
        q.image_precise_assets.insert(
            image_name=name, asset_format=PRECISE_ASSET_FORMAT, format_version=PRECISE_ASSET_VERSION
        )
    return name


def test_channel_of_a_precise_float_image_is_explicitly_quantized_to_8_bit(stack) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    red = np.array([[0.5, -0.3, 1.7], [0.3, 1.0, 0.25]], np.float32)
    pixels = np.stack([red, np.zeros_like(red), np.ones_like(red)], axis=2)
    name = _store_precise(stack, RasterImage.from_array(pixels, descriptor))

    output = ImageChannelInvocation(id="channel", image={"image_name": name}, channel="R").invoke(stack.context)

    # Clip to [0, 1], scale to 255, round to nearest: 127.5 -> 128 (not floored), 76.50000305 -> 77, 63.75 -> 64.
    expected = [[round(min(max(float(v), 0.0), 1.0) * 255) for v in row] for row in red]
    assert expected == [[128, 0, 255], [77, 255, 64]]
    with Image.open(stack.storage.get_path(output.image.image_name)) as result:
        assert result.mode == "L"
        assert np.asarray(result).tolist() == expected


@pytest.mark.parametrize(
    ("color", "value_range"),
    [
        (ColorEncoding.LINEAR_SRGB, ValueRange.UNIT),
        (ColorEncoding.SRGB, ValueRange.UNBOUNDED),
    ],
    ids=["linear", "unbounded"],
)
def test_channel_of_a_float_image_without_display_meaning_is_refused(color, value_range) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, color, value_range, AlphaMode.NONE, 3)
    source = RasterImage.from_array(np.full((2, 2, 3), 0.5, np.float32), descriptor)
    images = SimpleNamespace(get_raster=lambda name: source, save_raster=MagicMock())

    with pytest.raises(ValueError, match="Cannot take a displayable channel"):
        ImageChannelInvocation(id="channel", image={"image_name": "in.png"}, channel="G").invoke(
            SimpleNamespace(images=images)
        )

    images.save_raster.assert_not_called()


def test_with_precise_storage_the_channel_of_a_precise_float_image_stays_float(stack) -> None:
    stack.configuration.precise_image_storage = True
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    red = np.array([[0.5, -0.3, 1.7], [0.3, 1.0, 0.25]], np.float32)
    pixels = np.stack([red, np.zeros_like(red), np.ones_like(red)], axis=2)
    name = _store_precise(stack, RasterImage.from_array(pixels, descriptor))

    output = ImageChannelInvocation(id="channel", image={"image_name": name}, channel="R").invoke(stack.context)

    channel = stack.service.get_raster(output.image.image_name)
    assert channel.dtype == np.float32
    assert channel.descriptor.channels is RasterChannels.L
    assert channel.view()[:, :, 0].tobytes() == red.tobytes()
    with Image.open(stack.storage.get_path(output.image.image_name)) as derivative:
        assert np.asarray(derivative).tolist() == [[128, 0, 255], [77, 255, 64]]


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_with_precise_storage_a_png_only_source_gives_the_same_output_as_without(stack, mode: str) -> None:
    stack.configuration.precise_image_storage = True

    old_name, new_name = _run(stack, mode, "G")

    assert stack.storage.get_path(new_name).read_bytes() == stack.storage.get_path(old_name).read_bytes()
    assert stack.database.queries.image_precise_assets.get(new_name) is None


def test_channel_of_a_precise_uint16_rgba_image_keeps_all_16_bits(stack) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGBA, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.STRAIGHT, 4)
    pixels = np.array([[[0, 1, 2, 65535], [40000, 3, 4, 5]]], np.uint16)
    name = _store_precise(stack, RasterImage.from_array(pixels, descriptor))

    output = ImageChannelInvocation(id="channel", image={"image_name": name}, channel="R").invoke(stack.context)

    with Image.open(stack.storage.get_path(output.image.image_name)) as result:
        assert result.mode == "I;16"
        assert np.asarray(result).tolist() == [[0, 40000]]


def _premultiplied(dtype, values) -> RasterImage:
    descriptor = RasterDescriptor(RasterChannels.RGBA, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.PREMULTIPLIED, 4)
    return RasterImage.from_array(np.array([[values]], dtype), descriptor)


def _run_with(source: RasterImage, channel: str) -> list[RasterImage]:
    saved: list[RasterImage] = []
    images = SimpleNamespace(
        precise_storage_enabled=False,
        get_raster=lambda name: source,
        save_raster=lambda raster: saved.append(raster) or SimpleNamespace(image_name="out.png", width=1, height=1),
    )
    ImageChannelInvocation(id="channel", image={"image_name": "in.png"}, channel=channel).invoke(
        SimpleNamespace(images=images)
    )
    return saved


@pytest.mark.parametrize(
    ("source", "expected_alpha"),
    [
        (_premultiplied(np.float32, [0.25, 0.0, 0.0, 0.5]), 128),
        (_premultiplied(np.uint16, [16384, 0, 0, 32768]), 32768),
    ],
    ids=["float32", "uint16"],
)
def test_premultiplied_color_channels_are_refused_and_alpha_is_allowed(source, expected_alpha) -> None:
    for channel in "RGB":
        with pytest.raises(ValueError, match="premultiplied"):
            _run_with(source, channel)

    (saved,) = _run_with(source, "A")

    assert saved.view().tolist() == [[[expected_alpha]]]


def test_alpha_of_a_linear_float_image_is_taken_as_srgb_and_its_color_channels_are_refused() -> None:
    descriptor = RasterDescriptor(
        RasterChannels.RGBA, ColorEncoding.LINEAR_SRGB, ValueRange.UNIT, AlphaMode.STRAIGHT, 4
    )
    source = RasterImage.from_array(np.array([[[0.1, 0.2, 0.3, 0.5]]], np.float32), descriptor)

    (saved,) = _run_with(source, "A")

    assert saved.view().tolist() == [[[128]]]
    with pytest.raises(ValueError, match="linear_srgb"):
        _run_with(source, "R")


def test_channel_of_a_linear_uint16_image_is_refused_with_the_conversion_reason() -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.LINEAR_SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    source = RasterImage.from_array(np.array([[[1, 2, 3]]], np.uint16), descriptor)

    with pytest.raises(ValueError, match="linear_srgb"):
        _run_with(source, "G")
