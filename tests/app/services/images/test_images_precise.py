"""`ImageService.get_raster` resolves precise assets from the record store's row, over a real database and files.

Fixtures are written directly (`write_precise_asset` + a row through `Database.queries`), never by a production writer.
"""

import itertools
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException, ImageFileSaveException
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
        self._names = itertools.count()
        self.storage, self.service = self._build()

    def _build(self) -> tuple[DiskImageFileStorage, ImageService]:
        storage = DiskImageFileStorage(self.output)
        invoker = MagicMock()
        invoker.services.configuration.pil_compress_level = 6
        invoker.services.database = self.database
        invoker.services.image_records = self.records
        invoker.services.image_files = storage
        invoker.services.logger = MagicMock()
        invoker.services.configuration.image_subfolder_strategy = "hash"
        invoker.services.names.create_image_name.side_effect = lambda: f"new-{next(self._names)}.png"
        invoker.services.urls.get_image_url.side_effect = lambda name, thumbnail=False: f"/{name}"
        invoker.services.board_image_records.get_board_for_image.return_value = None
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
        is_intermediate: bool = False,
    ) -> None:
        """An image with a PNG derivative; with `precise` also the asset file and its row (or `asset` as the row)."""
        self.records.save(
            image_name=name,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=2,
            height=1,
            has_workflow=False,
            is_intermediate=is_intermediate,
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


# --- Lifecycle: the asset set is deleted, copied and accounted for as one unit -------------------------------------


def _files_under(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _asset_row(stack: Stack, name: str):
    return stack.database.queries.image_precise_assets.get(name)


def _set_paths(stack: Stack, name: str, subfolder: str = "") -> list[Path]:
    return [
        stack.storage.get_path(name, image_subfolder=subfolder),
        stack.storage.get_path(name, thumbnail=True, image_subfolder=subfolder),
        stack.storage.get_precise_path(name, subfolder),
    ]


@pytest.mark.parametrize("subfolder", ["", "2026/11"])
def test_delete_removes_png_thumbnail_precise_file_and_row(stack: Stack, subfolder: str) -> None:
    stack.add("p.png", subfolder=subfolder, precise=_float_rgb())
    paths = _set_paths(stack, "p.png", subfolder)
    paths[1].parent.mkdir(parents=True, exist_ok=True)
    paths[1].write_bytes(b"thumbnail")
    assert all(path.exists() for path in paths)

    stack.service.delete("p.png")

    assert not any(path.exists() for path in paths)
    assert _asset_row(stack, "p.png") is None
    assert not list(stack.output.glob(".delete_*"))


def test_delete_purges_a_precise_file_that_has_no_row(stack: Stack) -> None:
    stack.add("p.png", png_pixels=(1, 2, 3))
    stray = stack.storage.get_precise_path("p.png")
    write_precise_asset(_float_rgb(), stray)

    stack.service.delete("p.png")

    assert not stray.exists()


def test_bulk_delete_removes_the_precise_files_of_the_selected_images_only(stack: Stack) -> None:
    stack.add("a.png", precise=_float_rgb())
    stack.add("b.png", subfolder="x", precise=_uint16_rgba())
    stack.add("keep.png", precise=_float_rgb())

    deleted, failed = stack.service.delete_images_by_names(["a.png", "b.png"])

    assert (sorted(deleted), failed) == (["a.png", "b.png"], [])
    assert _files_under(stack.storage.precise_root) == {"keep.safetensors"}
    assert _asset_row(stack, "keep.png") is not None
    assert _asset_row(stack, "a.png") is None


def test_delete_intermediates_removes_the_precise_file(stack: Stack) -> None:
    stack.add("mid.png", precise=_float_rgb(), is_intermediate=True)
    stack.add("final.png", precise=_float_rgb())

    stack.service.delete_intermediates_by_names(["mid.png"])

    assert _files_under(stack.storage.precise_root) == {"final.safetensors"}
    assert _asset_row(stack, "mid.png") is None


def test_a_failed_save_cleanup_removes_a_precise_file_that_landed(stack: Stack) -> None:
    def save_then_fail(image, image_name, image_subfolder="", **_):
        path = stack.storage.get_precise_path(image_name, image_subfolder)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_precise_asset(_float_rgb(), path)
        raise ImageFileSaveException

    with Image.new("RGB", (2, 1)) as image:
        with (
            patch.object(stack.storage, "save", side_effect=save_then_fail),
            pytest.raises(ImageFileSaveException),
        ):
            stack.service.create(image, ResourceOrigin.INTERNAL, ImageCategory.GENERAL)

    assert _files_under(stack.output) == set()
    assert not stack.records.exists("new-0.png")


class TestCopy:
    @staticmethod
    def _snapshot(stack: Stack) -> dict[str, bytes]:
        return {
            path.relative_to(stack.output).as_posix(): path.read_bytes()
            for path in stack.output.rglob("*")
            if path.is_file()
        }

    def test_a_source_with_a_row_gets_its_own_asset_file_row_and_size(self, stack: Stack) -> None:
        stack.add("src.png", subfolder="old", precise=_uint16_rgba())

        dto = stack.service.copy("src.png")

        destination = stack.records.get(dto.image_name)
        paths = _set_paths(stack, dto.image_name, destination.image_subfolder)
        assert destination.image_subfolder == dto.image_name[:2]
        assert paths[2].name == f"{Path(dto.image_name).stem}.safetensors"
        assert paths[2].read_bytes() == stack.storage.get_precise_path("src.png", "old").read_bytes()
        assert _asset_row(stack, dto.image_name) == _asset_row(stack, "src.png")._replace(image_name=dto.image_name)
        assert destination.file_size_bytes == sum(path.stat().st_size for path in paths)
        assert stack.service.get_raster(dto.image_name).view().tobytes() == _uint16_rgba().view().tobytes()

    def test_a_source_without_a_row_copies_no_asset_even_with_a_stray_file(self, stack: Stack) -> None:
        stack.add("src.png")
        write_precise_asset(_float_rgb(), stack.storage.get_precise_path("src.png"))

        dto = stack.service.copy("src.png")

        subfolder = stack.records.get(dto.image_name).image_subfolder
        assert not stack.storage.get_precise_path(dto.image_name, subfolder).exists()
        assert _asset_row(stack, dto.image_name) is None
        assert _files_under(stack.storage.precise_root) == {"src.safetensors"}

    def test_the_destination_row_exists_before_the_copy_is_announced(self, stack: Stack) -> None:
        stack.add("src.png", precise=_float_rgb())
        announced_with_row: list[bool] = []

        def on_changed(dto) -> None:
            announced_with_row.append(_asset_row(stack, dto.image_name) is not None)

        with patch.object(stack.service, "_on_changed", side_effect=on_changed):
            stack.service.copy("src.png")

        assert announced_with_row == [True]

    @pytest.mark.parametrize("inject", ["row_insert", "dto", "changed_notification"])
    def test_a_failure_after_the_asset_landed_leaves_no_destination_and_an_untouched_source(
        self, stack: Stack, inject: str
    ) -> None:
        stack.add("src.png", subfolder="old", precise=_float_rgb())
        before = self._snapshot(stack)

        if inject == "row_insert":
            failure = patch.object(
                type(stack.database.queries.image_precise_assets), "insert", side_effect=RuntimeError("insert failed")
            )
        elif inject == "dto":
            failure = patch.object(stack.service, "get_dto", side_effect=RuntimeError("dto failed"))
        else:
            failure = patch.object(stack.service, "_on_changed", side_effect=RuntimeError("event failed"))
        with failure, patch.object(stack.service, "_on_changed", wraps=stack.service._on_changed) as announced:
            with pytest.raises(RuntimeError):
                stack.service.copy("src.png")
        if inject == "row_insert":
            announced.assert_not_called()

        assert self._snapshot(stack) == before
        assert not stack.records.exists("new-0.png")
        assert _asset_row(stack, "new-0.png") is None
        assert stack.records.get("src.png").image_subfolder == "old"

    def test_a_file_copy_failure_after_the_png_landed_is_unwound(self, stack: Stack) -> None:
        stack.add("src.png", precise=_float_rgb())
        before = self._snapshot(stack)
        real_copy2 = shutil.copy2

        def copy2(src, dst, *args, **kwargs):
            real_copy2(src, dst, *args, **kwargs)
            if str(dst).endswith(".safetensors"):
                raise OSError("disk full")

        with (
            patch("invokeai.app.services.image_files.image_files_disk.shutil.copy2", copy2),
            pytest.raises(ImageFileSaveException),
        ):
            stack.service.copy("src.png")

        assert self._snapshot(stack) == before
        assert not stack.records.exists("new-0.png")
