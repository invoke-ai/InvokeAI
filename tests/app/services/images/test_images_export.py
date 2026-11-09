"""`ImageService.get_export_formats` / `export_png16` over a real database and files."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.services.images.images_common import ImageExportBusyError, ImageExportUnavailableError
from invokeai.app.services.shared.database.database import Database
from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.convert import quantize_unit
from invokeai.backend.image_util.raster.png_codec import decode_png
from invokeai.backend.image_util.raster.precise_asset import (
    PRECISE_ASSET_FORMAT,
    PRECISE_ASSET_VERSION,
    write_precise_asset,
)
from tests.app.services.images.test_images_precise_write import Stack


@pytest.fixture
def stack(database: Database, tmp_path: Path) -> Stack:
    return Stack(database, tmp_path / "images")


def _float_raster(channels: RasterChannels = RasterChannels.RGB) -> RasterImage:
    count = len(channels.value)
    pixels = np.random.default_rng(5).uniform(-0.25, 1.25, size=(4, 6, count)).astype(np.float32)
    alpha = AlphaMode.STRAIGHT if channels is RasterChannels.LA else AlphaMode.NONE
    descriptor = RasterDescriptor(channels, ColorEncoding.SRGB, ValueRange.UNIT, alpha, count)
    return RasterImage.from_array(pixels, descriptor)


def test_the_export_carries_the_stored_text_strings_verbatim(stack: Stack) -> None:
    raster = _float_raster(RasterChannels.L)
    text = {
        "metadata": '{ "seed":  1 }',  # odd whitespace and non-Latin-1 text must survive untouched
        "workflow": '{"name": "ünicode ☃"}',
        "graph": '{"nodes": {}, "edges": []}',
        "resolved_inputs": '{"nodes": {}, "edges": []}',
    }
    name = stack.save(raster, **text)

    export = stack.service.export_png16(name)
    export.release()

    decoded = decode_png(export.content)
    assert decoded.raster.view().tolist() == quantize_unit(raster, np.uint16).view().tolist()
    assert (
        decoded.text.invokeai_metadata,
        decoded.text.invokeai_workflow,
        decoded.text.invokeai_graph,
        decoded.text.invokeai_resolved_inputs,
    ) == (text["metadata"], text["workflow"], text["graph"], text["resolved_inputs"])


def test_the_slot_is_held_by_an_export_until_released_and_never_waited_for(stack: Stack) -> None:
    first, second = (stack.save(_float_raster()) for _ in range(2))

    held = stack.service.export_png16(first)
    with pytest.raises(ImageExportBusyError):  # returns at once; a waiting export would hang this test
        stack.service.export_png16(second)
    held.release()
    held.release()  # idempotent: must not free a slot a later export holds
    next_export = stack.service.export_png16(second)
    with pytest.raises(ImageExportBusyError):
        stack.service.export_png16(first)
    next_export.release()


def test_an_error_after_the_claim_releases_the_slot_explicitly(stack: Stack) -> None:
    # The refusal is raised inside the claimed section. Keeping the exception (and so its traceback and the claim
    # object it references) alive stops the claim's finalizer from hiding a missing explicit release.
    ordinary = stack.service.create(Image.new("RGB", (4, 4)), ResourceOrigin.INTERNAL, ImageCategory.GENERAL).image_name
    precise = stack.save(_float_raster())

    with pytest.raises(ImageExportUnavailableError) as excinfo:
        stack.service.export_png16(ordinary)
    stack.service.export_png16(precise).release()

    assert excinfo.value is not None


def test_an_abandoned_export_frees_the_slot_when_dropped(stack: Stack) -> None:
    name = stack.save(_float_raster())

    stack.service.export_png16(name)  # never released: a disconnected client's response may not run its task

    stack.service.export_png16(name).release()


def _ordinary_with_precise_row(stack: Stack, raster: RasterImage | None, content: bytes | None = None) -> str:
    """An image whose row claims a precise asset: a written one, a corrupt file, or none at all."""
    name = stack.service.create(Image.new("RGB", (4, 4)), ResourceOrigin.INTERNAL, ImageCategory.GENERAL).image_name
    path = stack.storage.get_precise_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raster is not None:
        write_precise_asset(raster, path)
    elif content is not None:
        path.write_bytes(content)
    with stack.database.queries.transaction() as q:
        q.image_precise_assets.insert(
            image_name=name, asset_format=PRECISE_ASSET_FORMAT, format_version=PRECISE_ASSET_VERSION
        )
    return name


def test_a_precise_gray_alpha_asset_lists_no_format_because_the_png_writer_cannot_encode_it(stack: Stack) -> None:
    name = _ordinary_with_precise_row(stack, _float_raster(RasterChannels.LA))

    assert stack.service.get_export_formats(name) == []


@pytest.mark.parametrize("content", [None, b"", b"not a safetensors file"], ids=["missing", "empty", "corrupt"])
def test_an_unreadable_precise_asset_lists_no_format_instead_of_failing(stack: Stack, content: bytes | None) -> None:
    name = _ordinary_with_precise_row(stack, None, content)

    assert stack.service.get_export_formats(name) == []
    assert "unreadable" in stack.invoker.services.logger.warning.call_args.args[0]
