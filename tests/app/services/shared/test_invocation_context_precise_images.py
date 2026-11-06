"""`ImagesInterface.save_raster` and `get_raster` across nodes, with the real image service, storage and database."""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.backend.image_util.raster.convert import quantize_unit, raster_to_pil
from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterConversionError,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage


class Stack:
    """One shared database and output folder; each `node()` is a context over its own service and storage."""

    def __init__(self, database: Database, output: Path, enabled: bool | None) -> None:
        self.database = database
        self.output = output
        self.enabled = enabled
        self.records = ImageRecordStorage(database)
        self.names = itertools.count()

    def node(self, node_id: str = "node") -> ImagesInterface:
        storage = DiskImageFileStorage(self.output)
        invoker = MagicMock()
        services = invoker.services
        # A real configuration, so the field name and its default are exercised.
        services.configuration = (
            InvokeAIAppConfig(use_memory_db=True)
            if self.enabled is None
            else InvokeAIAppConfig(use_memory_db=True, precise_image_storage=self.enabled)
        )
        services.database = self.database
        services.image_records = self.records
        services.image_files = storage
        services.logger = MagicMock()
        services.names.create_image_name.side_effect = lambda: f"image-{next(self.names)}.png"
        services.urls.get_image_url.side_effect = lambda name, thumbnail=False: f"/{name}"
        services.board_image_records.get_board_for_image.return_value = None
        storage.start(invoker)
        service = ImageService()
        service.start(invoker)
        services.images = service
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
            invocation=SimpleNamespace(is_intermediate=False, id=node_id),
        )
        self.storage = storage
        return ImagesInterface(services, data, MagicMock())

    def files(self) -> list[Path]:
        return [path for path in self.output.rglob("*") if path.is_file()]


@pytest.fixture
def stack(database: Database, tmp_path: Path) -> Stack:
    return Stack(database, tmp_path / "images", enabled=True)


@pytest.fixture
def default_stack(database: Database, tmp_path: Path) -> Stack:
    return Stack(database, tmp_path / "images", enabled=None)


def _rasters() -> dict[str, RasterImage]:
    rgb = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    rgba = RasterDescriptor(RasterChannels.RGBA, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.STRAIGHT, 4)
    rng = np.random.default_rng(5)
    float32 = rng.uniform(-0.5, 2.0, size=(6, 9, 3)).astype(np.float32)
    float32[0, 0] = [7.0, -3.0, 0.5]
    return {
        "float32-rgb": RasterImage.from_array(float32, rgb),
        "float16-rgba": RasterImage.from_array(rng.uniform(-0.5, 2.0, size=(6, 9, 4)).astype(np.float16), rgba),
        "uint16-rgba": RasterImage.from_array(
            rng.integers(0, 65536, size=(6, 9, 4), dtype=np.uint16),
            rgba,
        ),
    }


@pytest.mark.parametrize("case", ["float32-rgb", "float16-rgba", "uint16-rgba"])
def test_another_node_and_a_restarted_process_read_back_the_exact_raster(stack: Stack, case: str) -> None:
    raster = _rasters()[case]

    saved = stack.node("writer").save_raster(raster)
    from_other_node = stack.node("reader").get_raster(saved.image_name)
    # Fresh storage and service instances over the same folder and database, as after a restart.
    after_restart = stack.node("restarted").get_raster(saved.image_name)

    for read in (from_other_node, after_restart):
        assert (read.dtype, read.descriptor) == (raster.dtype, raster.descriptor)
        assert read.view().tobytes() == raster.view().tobytes()
    expected = raster_to_pil(quantize_unit(raster, np.uint8))
    with Image.open(stack.storage.get_path(saved.image_name)) as derivative:
        assert derivative.tobytes() == expected.tobytes()


def test_a_precise_save_carries_the_same_metadata_as_an_ordinary_save(stack: Stack) -> None:
    node = stack.node()
    raster = _rasters()["float32-rgb"]
    metadata = MetadataField({"prompt": "a fox", "seed": 3})

    precise = node.save_raster(raster, board_id="board-1", metadata=metadata)
    ordinary = node.save(raster_to_pil(quantize_unit(raster, np.uint8)), board_id="board-1", metadata=metadata)

    assert stack.records.get_metadata(precise.image_name) == stack.records.get_metadata(ordinary.image_name)
    with (
        Image.open(stack.storage.get_path(precise.image_name)) as a,
        Image.open(stack.storage.get_path(ordinary.image_name)) as b,
    ):
        assert a.info["invokeai_metadata"] == b.info["invokeai_metadata"] == metadata.model_dump_json()
    assert precise.model_dump(
        exclude={"image_name", "image_url", "thumbnail_url", "created_at", "updated_at", "file_size_bytes"}
    ) == (
        ordinary.model_dump(
            exclude={"image_name", "image_url", "thumbnail_url", "created_at", "updated_at", "file_size_bytes"}
        )
    )


def test_a_representable_raster_is_an_ordinary_image_even_when_precise_storage_is_enabled(stack: Stack) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    raster = RasterImage.from_array(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), descriptor)
    node = stack.node()

    saved = node.save_raster(raster)

    assert stack.database.queries.image_precise_assets.get(saved.image_name) is None
    assert not any(path.suffix == ".safetensors" for path in stack.files())


def test_precise_storage_is_off_by_default_and_a_float_raster_is_refused_with_nothing_written(
    default_stack: Stack,
) -> None:
    node = default_stack.node()

    assert node.precise_storage_enabled is False
    with pytest.raises(RasterConversionError, match="precise_image_storage"):
        node.save_raster(_rasters()["float32-rgb"])

    assert default_stack.files() == []
    assert next(default_stack.names) == 0


def _display_image(raster: RasterImage, mode: str = "RGB", size: tuple[int, int] | None = None) -> Image.Image:
    """A gallery PNG source unlike the quantized derivative (inverted), as a family's legacy decode would differ."""
    size = size or (raster.width, raster.height)
    values = np.random.default_rng(9).integers(0, 256, size=(size[1], size[0], len(mode)), dtype=np.uint8)
    return Image.fromarray(values[:, :, 0] if len(mode) == 1 else values, mode)


def test_display_image_is_stored_byte_for_byte_as_the_gallery_png_of_a_precise_save(stack: Stack) -> None:
    raster = _rasters()["float32-rgb"]
    display = _display_image(raster)
    node = stack.node()

    saved = node.save_raster(raster, display_image=display)

    with Image.open(stack.storage.get_path(saved.image_name)) as png:
        assert png.mode == "RGB"
        assert png.tobytes() == display.tobytes()
        assert png.tobytes() != raster_to_pil(quantize_unit(raster, np.uint8)).tobytes()
    assert node.get_raster(saved.image_name).view().tobytes() == raster.view().tobytes()


def test_display_image_of_the_ordinary_derivative_changes_nothing(stack: Stack) -> None:
    raster = _rasters()["float16-rgba"]
    node = stack.node()

    explicit = node.save_raster(raster, display_image=raster_to_pil(quantize_unit(raster, np.uint8)))
    omitted = node.save_raster(raster)

    assert (
        stack.storage.get_path(explicit.image_name).read_bytes()
        == stack.storage.get_path(omitted.image_name).read_bytes()
    )


@pytest.mark.parametrize(
    ("case", "display"),
    [
        ("size", lambda raster: _display_image(raster, "RGB", (raster.width + 1, raster.height))),
        ("mode", lambda raster: _display_image(raster, "RGBA")),
        ("gray", lambda raster: _display_image(raster, "L")),
        ("dtype", lambda raster: Image.fromarray(np.zeros((raster.height, raster.width), np.float32), "F")),
    ],
)
def test_a_mismatched_display_image_is_refused_with_nothing_persisted(stack: Stack, case: str, display) -> None:
    raster = _rasters()["float32-rgb"]

    with pytest.raises(ValueError):
        stack.node().save_raster(raster, display_image=display(raster))

    assert stack.files() == []
    assert next(stack.names) == 0


def test_a_display_image_with_a_non_finite_raster_is_refused_with_nothing_persisted(stack: Stack) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    pixels = np.full((4, 4, 3), 0.5, np.float32)
    pixels[1, 1, 0] = np.nan
    raster = RasterImage.from_array(pixels, descriptor)

    with pytest.raises(ValueError, match="non-finite"):
        stack.node().save_raster(raster, display_image=_display_image(raster))

    assert stack.files() == []
    assert next(stack.names) == 0


def test_a_display_image_for_a_png_representable_raster_is_a_caller_error(stack: Stack) -> None:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    raster = RasterImage.from_array(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), descriptor)

    with pytest.raises(ValueError, match="display_image"):
        stack.node().save_raster(raster, display_image=_display_image(raster))

    assert stack.files() == []
    assert next(stack.names) == 0
