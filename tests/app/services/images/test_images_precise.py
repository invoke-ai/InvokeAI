"""`ImageService.get_raster` resolves precise assets from the record store's row, over a real database and files.

Fixtures are written directly (`write_precise_asset` + a row through `Database.queries`), never by a production writer.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.database import Database
from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.precise_asset import (
    PRECISE_ASSET_FORMAT,
    PRECISE_ASSET_VERSION,
    PreciseAssetError,
    write_precise_asset,
)

_CURRENT = (PRECISE_ASSET_FORMAT, PRECISE_ASSET_VERSION)


class Stack:
    def __init__(self, database: Database, output: Path) -> None:
        self.database = database
        self.output = output
        self.records = ImageRecordStorage(database)
        self.storage, self.service = self._build()

    def _build(self) -> tuple[DiskImageFileStorage, ImageService]:
        storage = DiskImageFileStorage(self.output)
        invoker = MagicMock()
        invoker.services.configuration.pil_compress_level = 6
        invoker.services.database = self.database
        invoker.services.image_records = self.records
        invoker.services.image_files = storage
        invoker.services.logger = MagicMock()
        storage.start(invoker)
        service = ImageService()
        service.start(invoker)
        return storage, service

    def restart(self) -> None:
        """Fresh storage and service over the same folder and database, as after a process restart."""
        self.storage, self.service = self._build()

    def add(
        self,
        name: str,
        png_pixels: tuple[int, int, int] = (9, 9, 9),
        subfolder: str = "",
        precise: RasterImage | None = None,
        asset: tuple[str, int] | None = None,
    ) -> None:
        """An image with a PNG derivative; with `precise` also the asset file and its row (or `asset` as the row)."""
        self.records.save(
            image_name=name,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=2,
            height=1,
            has_workflow=False,
            image_subfolder=subfolder,
        )
        if ".." not in subfolder:
            with Image.new("RGB", (2, 1), png_pixels) as derivative:
                self.storage.save(derivative, name, image_subfolder=subfolder)
            self.storage.evict_cache_paths([self.storage.get_path(name, image_subfolder=subfolder)])
            if precise is not None:
                path = self.storage.get_precise_path(name, subfolder)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_precise_asset(precise, path)
        row = asset or (_CURRENT if precise is not None else None)
        if row is not None:
            with self.database.queries.transaction() as q:
                q.image_precise_assets.insert(image_name=name, asset_format=row[0], format_version=row[1])


@pytest.fixture
def stack(database: Database, tmp_path: Path) -> Stack:
    return Stack(database, tmp_path / "images")


def _float_rgb() -> RasterImage:
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    nan_with_payload = np.array([0x7FC00123], np.uint32).view(np.float32)[0]
    pixels = np.array([[[2.5, -0.25, nan_with_payload], [0.0, 1.0, 7.0]]], np.float32)
    return RasterImage.from_array(pixels, descriptor)


def _uint16_rgba() -> RasterImage:
    descriptor = RasterDescriptor(RasterChannels.RGBA, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.STRAIGHT, 4)
    pixels = np.array([[[0, 1, 257, 65535], [40000, 65535, 3, 12345]]], np.uint16)
    return RasterImage.from_array(pixels, descriptor)


@pytest.mark.parametrize("make", [_float_rgb, _uint16_rgba], ids=["float32-rgb", "uint16-rgba"])
def test_a_record_with_an_asset_returns_the_precise_raster_byte_exactly(stack: Stack, make) -> None:
    expected = make()
    stack.add("precise.png", png_pixels=(1, 2, 3), precise=expected)

    raster = stack.service.get_raster("precise.png")

    assert raster.dtype == expected.dtype
    assert raster.descriptor == expected.descriptor
    assert raster.view().tobytes() == expected.view().tobytes()


def test_old_png_only_records_and_precise_records_coexist(stack: Stack) -> None:
    stack.add("old.png", png_pixels=(10, 20, 30))
    stack.add("new.png", png_pixels=(1, 2, 3), precise=_float_rgb())

    old = stack.service.get_raster("old.png")
    new = stack.service.get_raster("new.png")

    assert old.dtype == np.uint8
    assert old.view().tolist() == [[[10, 20, 30], [10, 20, 30]]]
    assert new.dtype == np.float32
    # The PNG derivative of the precise image keeps its own 8-bit pixels for every compatibility view.
    assert stack.service.get_pil_image("new.png").getpixel((0, 0)) == (1, 2, 3)


def test_a_record_in_a_subfolder_resolves_the_asset_in_the_same_subfolder(stack: Stack) -> None:
    stack.add("flat.png", precise=_uint16_rgba())
    stack.add("nested.png", subfolder="2026/11", precise=_float_rgb())

    assert (stack.storage.precise_root / "2026" / "11" / "nested.safetensors").is_file()
    assert stack.service.get_raster("nested.png").dtype == np.float32
    assert stack.service.get_raster("flat.png").dtype == np.uint16


def test_a_record_subfolder_that_escapes_the_precise_root_is_refused(stack: Stack) -> None:
    stack.add("evil.png", subfolder="../elsewhere", asset=_CURRENT)

    with pytest.raises(ValueError, match="not allowed"):
        stack.service.get_raster("evil.png")


def test_a_missing_asset_file_is_an_image_file_not_found_and_never_the_png(stack: Stack) -> None:
    stack.add("gone.png", png_pixels=(1, 2, 3), precise=_float_rgb())
    stack.storage.get_precise_path("gone.png").unlink()

    with pytest.raises(ImageFileNotFoundException, match="Precise asset"):
        stack.service.get_raster("gone.png")
    assert stack.service.get_pil_image("gone.png").getpixel((0, 0)) == (
        1,
        2,
        3,
    )  # the derivative is there to fall back to


def test_a_corrupt_asset_file_is_a_precise_asset_error_and_never_the_png(stack: Stack) -> None:
    stack.add("bad.png", precise=_float_rgb())
    path = stack.storage.get_precise_path("bad.png")
    path.write_bytes(path.read_bytes()[:40])

    with pytest.raises(PreciseAssetError):
        stack.service.get_raster("bad.png")


@pytest.mark.parametrize(
    "asset",
    [("raster_exr", PRECISE_ASSET_VERSION), (PRECISE_ASSET_FORMAT, PRECISE_ASSET_VERSION + 1)],
    ids=["unknown-format", "unknown-version"],
)
def test_a_row_with_an_unknown_format_or_version_is_a_precise_asset_error(stack: Stack, asset) -> None:
    stack.add("future.png", precise=_float_rgb(), asset=asset)

    with pytest.raises(PreciseAssetError, match="Unsupported precise asset"):
        stack.service.get_raster("future.png")


def test_a_file_without_a_row_is_ignored(stack: Stack) -> None:
    """The row alone decides whether an asset exists; a leftover file never changes what a read returns."""
    stack.add("orphan.png", png_pixels=(10, 20, 30))
    write_precise_asset(_float_rgb(), stack.storage.get_precise_path("orphan.png"))

    assert stack.service.get_raster("orphan.png").dtype == np.uint8


def test_nothing_is_cached_between_reads(stack: Stack) -> None:
    stack.add("p.png", precise=_uint16_rgba())
    first = stack.service.get_raster("p.png")
    replacement = RasterImage.from_array(first.view() // 2, first.descriptor)
    write_precise_asset(replacement, stack.storage.get_precise_path("p.png"))

    second = stack.service.get_raster("p.png")

    assert second.view().tolist() == replacement.view().tolist()
    assert first.view().tolist() != second.view().tolist()


def test_returned_rasters_are_owned_and_independent(stack: Stack) -> None:
    stack.add("p.png", precise=_float_rgb())

    first = stack.service.get_raster("p.png")
    second = stack.service.get_raster("p.png")
    scratch = first.to_array()
    scratch[...] = 0

    assert first.is_owned and second.is_owned
    assert not np.shares_memory(first.view(), second.view())
    assert first.view().tobytes() == second.view().tobytes() == _float_rgb().view().tobytes()
    with pytest.raises(ValueError):
        first.view()[0, 0, 0] = 1.0


def test_a_missing_record_is_still_a_record_not_found(stack: Stack) -> None:
    with pytest.raises(ImageRecordNotFoundException):
        stack.service.get_raster("nothing.png")


def test_an_os_error_other_than_missing_propagates_and_never_returns_the_png(stack: Stack) -> None:
    """A precise path that cannot be opened as a file (here a directory) is an I/O fault, not a missing asset."""
    stack.add("io.png", png_pixels=(1, 2, 3), precise=_float_rgb())
    path = stack.storage.get_precise_path("io.png")
    path.unlink()
    path.mkdir()

    with pytest.raises(OSError) as storage_error:
        stack.storage.read_precise("io.png")
    with pytest.raises(OSError) as service_error:
        stack.service.get_raster("io.png")

    assert not isinstance(storage_error.value, ImageFileNotFoundException | FileNotFoundError)
    assert not isinstance(service_error.value, ImageFileNotFoundException | FileNotFoundError)
