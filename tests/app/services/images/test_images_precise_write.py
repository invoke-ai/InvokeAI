"""`ImageService.create_from_raster` writes the complete precise asset set as one unit, over a real database and files."""

import itertools
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_common import ImageFileSaveException
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveService
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.image_precise_assets import ImagePreciseAssetQueries
from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.convert import quantize_unit, raster_to_pil
from invokeai.backend.image_util.raster.descriptor import RasterConversionError
from invokeai.backend.image_util.raster.precise_asset import (
    PRECISE_ASSET_FORMAT,
    PRECISE_ASSET_VERSION,
    PreciseAssetError,
)
from invokeai.backend.util.logging import InvokeAILogger

_STORAGE_MODULE = "invokeai.app.services.image_files.image_files_disk"


class Stack:
    def __init__(self, database: Database, output: Path, strategy: str = "flat") -> None:
        self.database = database
        self.output = output
        self.strategy = strategy
        self.records = ImageRecordStorage(database)
        self.names = itertools.count()
        config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
        self.storage, self.service = self._build()
        self.moves = ImageMoveService(
            database, image_files=self.storage, config=config, logger=InvokeAILogger.get_logger()
        )
        self.invoker.services.image_moves = self.moves

    def _build(self) -> tuple[DiskImageFileStorage, ImageService]:
        storage = DiskImageFileStorage(self.output)
        self.invoker = invoker = MagicMock()
        invoker.services.configuration.pil_compress_level = 6
        invoker.services.configuration.image_subfolder_strategy = self.strategy
        invoker.services.database = self.database
        invoker.services.image_records = self.records
        invoker.services.image_files = storage
        invoker.services.logger = MagicMock()
        invoker.services.names.create_image_name.side_effect = lambda: f"new-{next(self.names)}.png"
        invoker.services.urls.get_image_url.side_effect = lambda name, thumbnail=False: f"/{name}"
        invoker.services.board_image_records.get_board_for_image.return_value = None
        storage.start(invoker)
        service = ImageService()
        service.start(invoker)
        return storage, service

    def restart(self) -> None:
        """Fresh storage and service over the same folder and database, as after a process restart."""
        self.storage, self.service = self._build()
        self.invoker.services.image_moves = self.moves

    def save(self, raster: RasterImage, **kwargs) -> str:
        return self.service.create_from_raster(
            raster, ResourceOrigin.INTERNAL, ImageCategory.GENERAL, **kwargs
        ).image_name

    def files(self) -> list[Path]:
        return sorted(path for path in self.output.rglob("*") if path.is_file())

    def row(self, name: str):
        return self.database.queries.image_precise_assets.get(name)

    def assert_nothing_persisted(self, name: str = "new-0.png") -> None:
        with pytest.raises(ImageRecordNotFoundException):
            self.records.get(name)
        assert self.row(name) is None
        assert self.files() == []


@pytest.fixture
def stack(database: Database, tmp_path: Path) -> Stack:
    return Stack(database, tmp_path / "images")


def _raster(
    dtype, channels: RasterChannels, alpha: AlphaMode, values=None, color=ColorEncoding.SRGB, vr=ValueRange.UNIT
):
    count = len(channels.value)
    descriptor = RasterDescriptor(channels, color, vr, alpha, count)
    rng = np.random.default_rng(7)
    if np.dtype(dtype).kind == "f":
        pixels = rng.uniform(-0.5, 1.75, size=(5, 7, count)).astype(dtype)
    else:
        pixels = rng.integers(0, np.iinfo(dtype).max, size=(5, 7, count), dtype=dtype, endpoint=True)
    if values is not None:
        pixels[0, 0, :] = values
    return RasterImage.from_array(pixels, descriptor)


_EXACT_CASES = {
    "float32-rgb": lambda: _raster(np.float32, RasterChannels.RGB, AlphaMode.NONE, values=[2.5, -0.25, 7.0]),
    "float16-rgba": lambda: _raster(np.float16, RasterChannels.RGBA, AlphaMode.STRAIGHT),
    "uint16-rgba": lambda: _raster(np.uint16, RasterChannels.RGBA, AlphaMode.STRAIGHT),
    "float32-l": lambda: _raster(np.float32, RasterChannels.L, AlphaMode.NONE),
}


@pytest.mark.parametrize("case", list(_EXACT_CASES))
def test_a_precise_save_stores_the_complete_set_and_reads_back_byte_exactly(stack: Stack, case: str) -> None:
    raster = _EXACT_CASES[case]()

    name = stack.save(raster)
    # A second reader on fresh storage and service instances sees the same bytes, as after a restart.
    stack.restart()
    read = stack.service.get_raster(name)

    assert (read.dtype, read.descriptor) == (raster.dtype, raster.descriptor)
    assert read.view().tobytes() == raster.view().tobytes()
    assert stack.row(name).asset_format == PRECISE_ASSET_FORMAT
    assert stack.row(name).format_version == PRECISE_ASSET_VERSION
    expected = raster_to_pil(quantize_unit(raster, np.uint8))
    with Image.open(stack.storage.get_path(name)) as derivative:
        assert derivative.mode == expected.mode
        assert derivative.tobytes() == expected.tobytes()
    assert stack.storage.get_path(name, thumbnail=True).is_file()
    assert stack.storage.get_precise_path(name).is_file()
    record = stack.records.get(name)
    assert (record.width, record.height) == (raster.width, raster.height)
    assert record.file_size_bytes == sum(path.stat().st_size for path in stack.files())


def test_values_outside_the_unit_range_are_clipped_only_in_the_derivative(stack: Stack) -> None:
    name = stack.save(_EXACT_CASES["float32-rgb"]())

    assert stack.service.get_raster(name).view()[0, 0].tolist() == [2.5, -0.25, 7.0]
    with Image.open(stack.storage.get_path(name)) as derivative:
        assert derivative.getpixel((0, 0)) == (255, 0, 255)


def test_a_precise_save_carries_the_same_record_and_png_as_an_ordinary_save(stack: Stack) -> None:
    raster = _EXACT_CASES["float32-rgb"]()
    options = {
        "node_id": "node-1",
        "session_id": "session-1",
        "is_intermediate": True,
        "metadata": '{"a": 1}',
        "workflow": '{"w": 2}',
        "graph": '{"g": 3}',
        "resolved_inputs": '{"r": 4}',
        "user_id": "user-1",
        "project_id": "project-1",
        "board_id": "board-1",
    }
    precise = stack.service.create_from_raster(raster, ResourceOrigin.INTERNAL, ImageCategory.CONTROL, **options)
    ordinary = stack.service.create(
        raster_to_pil(quantize_unit(raster, np.uint8)), ResourceOrigin.INTERNAL, ImageCategory.CONTROL, **options
    )

    ignored = {"image_name", "image_url", "thumbnail_url", "created_at", "updated_at", "file_size_bytes"}
    assert {k: v for k, v in precise.model_dump().items() if k not in ignored} == {
        k: v for k, v in ordinary.model_dump().items() if k not in ignored
    }
    assert stack.records.get_metadata(precise.image_name) == stack.records.get_metadata(ordinary.image_name)
    assert (
        stack.storage.get_path(precise.image_name).read_bytes()
        == stack.storage.get_path(ordinary.image_name).read_bytes()
    )
    assert (
        stack.storage.get_path(precise.image_name, thumbnail=True).read_bytes()
        == stack.storage.get_path(ordinary.image_name, thumbnail=True).read_bytes()
    )
    boards = stack.invoker.services.board_image_records.add_image_to_board.call_args_list
    assert [call.kwargs["board_id"] for call in boards] == ["board-1", "board-1"]
    assert stack.row(ordinary.image_name) is None


def test_a_precise_save_lands_in_the_records_subfolder(database: Database, tmp_path: Path) -> None:
    stack = Stack(database, tmp_path / "images", strategy="hash")

    name = stack.save(_EXACT_CASES["uint16-rgba"]())

    subfolder = stack.records.get(name).image_subfolder
    assert subfolder
    assert stack.storage.get_precise_path(name, subfolder).is_file()
    assert stack.service.get_raster(name).dtype == np.uint16


def _with_value(value: float) -> RasterImage:
    raster = _raster(np.float32, RasterChannels.RGB, AlphaMode.NONE)
    pixels = raster.to_array()
    pixels[3, 4, 1] = value
    return RasterImage.from_array(pixels, raster.descriptor)


def _nan_beyond_the_first_row_block() -> RasterImage:
    descriptor = RasterDescriptor(RasterChannels.L, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 1)
    pixels = np.full((600, 3, 1), 0.5, np.float32)
    pixels[599, 2, 0] = np.nan
    return RasterImage.from_array(pixels, descriptor)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (_nan_beyond_the_first_row_block, "non-finite"),
        (
            lambda: _raster(np.float32, RasterChannels.RGB, AlphaMode.NONE, color=ColorEncoding.LINEAR_SRGB),
            "linear_srgb",
        ),
        (lambda: _raster(np.float32, RasterChannels.RGB, AlphaMode.NONE, vr=ValueRange.UNBOUNDED), "unbounded"),
        (lambda: _raster(np.float32, RasterChannels.RGBA, AlphaMode.PREMULTIPLIED), "premultiplied"),
        (
            lambda: RasterImage.from_array(
                np.zeros((2, 2, 3), np.float32),
                RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNBOUNDED, AlphaMode.NONE, 3),
            ),
            "data",
        ),
        (lambda: _with_value(float("nan")), "non-finite"),
        (lambda: _with_value(float("inf")), "non-finite"),
    ],
    ids=["nan-late", "linear", "unbounded", "premultiplied", "data", "nan", "inf"],
)
def test_rasters_without_a_display_policy_are_refused_and_nothing_is_persisted(stack: Stack, make, reason: str) -> None:
    with pytest.raises(RasterConversionError, match=reason):
        stack.save(make())

    stack.assert_nothing_persisted()
    assert next(stack.names) == 0, "a refused raster must not even consume an image name"


def _icc(color_space: bytes, length: int = 128) -> bytes:
    return bytes(16) + color_space + bytes(length - 20)


def _with_icc(raster: RasterImage, profile: bytes) -> RasterImage:
    return RasterImage.from_array(raster.view(), replace(raster.descriptor, icc_profile=profile))


@pytest.mark.parametrize(
    ("case", "profile"),
    [
        ("float32-l", _icc(b"RGB ")),
        ("float32-rgb", _icc(b"GRAY")),
        ("float16-rgba", _icc(b"GRAY")),
        ("float32-l", b"short"),
    ],
    ids=["l-with-rgb", "rgb-with-gray", "rgba-with-gray", "too-short"],
)
def test_an_icc_profile_that_does_not_match_the_channels_is_refused_and_nothing_is_persisted(
    stack: Stack, case: str, profile: bytes
) -> None:
    with pytest.raises(RasterConversionError, match="ICC"):
        stack.save(_with_icc(_EXACT_CASES[case](), profile))

    stack.assert_nothing_persisted()


@pytest.mark.parametrize(
    ("case", "profile"), [("float32-l", _icc(b"GRAY")), ("float16-rgba", _icc(b"RGB ", 400))], ids=["gray", "rgb"]
)
def test_a_matching_icc_profile_is_accepted_and_round_trips(stack: Stack, case: str, profile: bytes) -> None:
    raster = _with_icc(_EXACT_CASES[case](), profile)

    name = stack.save(raster)

    assert stack.service.get_raster(name).descriptor.icc_profile == profile


def test_an_out_of_bounds_raster_is_refused_before_anything_is_persisted(stack: Stack) -> None:
    # A zero-stride view: 20000 x 20000 RGBA float32 without allocating it.
    descriptor = RasterDescriptor(RasterChannels.RGBA, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.STRAIGHT, 4)
    huge = RasterImage.borrow(np.broadcast_to(np.zeros((1, 1, 4), np.float32), (20000, 20000, 4)), descriptor)

    with pytest.raises(PreciseAssetError):
        stack.save(huge)

    stack.assert_nothing_persisted()


def _boom(*args, **kwargs):
    raise OSError("injected")


def _install_fault(stack: Stack, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    if step == "record_save":

        def fail(**kwargs):
            raise ImageRecordSaveException

        monkeypatch.setattr(stack.records, "save", fail)
    elif step == "precise_staging":

        def partial_then_fail(raster, path):
            path.write_bytes(b"partial")
            raise OSError("injected")

        monkeypatch.setattr(f"{_STORAGE_MODULE}.write_precise_asset", partial_then_fail)
    elif step == "publish":
        monkeypatch.setattr(f"{_STORAGE_MODULE}.publish_noreplace", _boom)
    elif step == "publish_directory_sync":
        real_sync = DiskImageFileStorage._DiskImageFileStorage__fsync_directory
        calls = []

        def fail_once(directory, missing_ok=False):
            # Only the publication's own sync fails; the failure cleanup syncs its journal through the same method.
            calls.append(directory)
            if len(calls) == 1:
                raise OSError("injected")
            return real_sync(directory, missing_ok)

        monkeypatch.setattr(DiskImageFileStorage, "_DiskImageFileStorage__fsync_directory", staticmethod(fail_once))
    elif step == "row_insert":
        monkeypatch.setattr(ImagePreciseAssetQueries, "insert", _boom)
    elif step == "png":
        monkeypatch.setattr(stack.storage, "save", _boom)
    elif step == "thumbnail":
        monkeypatch.setattr(f"{_STORAGE_MODULE}.make_thumbnail", _boom)
    elif step == "thumbnail_save":
        broken = MagicMock()
        broken.save.side_effect = OSError("injected")
        monkeypatch.setattr(f"{_STORAGE_MODULE}.make_thumbnail", lambda image, size: broken)
    elif step == "notify":
        stack.service.on_changed(_boom)
    else:
        raise AssertionError(step)


@pytest.mark.parametrize(
    "step",
    [
        "record_save",
        "precise_staging",
        "publish",
        "publish_directory_sync",
        "row_insert",
        "png",
        "thumbnail",
        "thumbnail_save",
        "notify",
    ],
)
def test_a_failure_at_any_step_leaves_no_record_row_or_file(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    source = _EXACT_CASES["float32-rgb"]()
    before = source.view().tobytes()
    _install_fault(stack, monkeypatch, step)

    with pytest.raises((OSError, ImageFileSaveException, ImageRecordSaveException)):
        stack.save(source)

    stack.assert_nothing_persisted()
    assert source.view().tobytes() == before


def test_a_failed_size_measurement_does_not_discard_the_finished_image(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stack.storage, "get_file_size_bytes", _boom)

    name = stack.save(_EXACT_CASES["float32-rgb"]())

    assert stack.records.get(name).file_size_bytes is None
    assert stack.service.get_raster(name).dtype == np.float32
    assert len(stack.files()) == 3


def test_an_existing_asset_at_the_derived_path_is_refused_not_adopted(stack: Stack) -> None:
    stray = stack.storage.get_precise_path("new-0.png")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"stray")

    with pytest.raises(ImageFileSaveException):
        stack.save(_EXACT_CASES["float32-rgb"]())

    with pytest.raises(ImageRecordNotFoundException):
        stack.records.get("new-0.png")
    assert stack.row("new-0.png") is None


def test_the_mutation_lock_is_held_across_every_step_of_a_precise_save(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, bool] = {}
    holds: dict[str, int] = {}
    outermost_holds = [0]
    depth = [0]
    real_lock = stack.moves.image_mutation_lock

    @contextmanager
    def counting_lock(**kwargs):
        # An outermost entry is a separate hold: a release and re-take in the middle shows up as a second one.
        if depth[0] == 0:
            outermost_holds[0] += 1
        depth[0] += 1
        try:
            with real_lock(**kwargs) as acquired:
                yield acquired
        finally:
            depth[0] -= 1

    monkeypatch.setattr(stack.moves, "image_mutation_lock", counting_lock)

    def lock_is_held_for_other_threads() -> bool:
        result: list[bool] = []

        def probe() -> None:
            acquired = stack.moves._image_mutation_lock.acquire(blocking=False)
            if acquired:
                stack.moves._image_mutation_lock.release()
            result.append(acquired)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        return not result[0]

    def wrap(owner, attribute: str, step: str) -> None:
        real = getattr(owner, attribute)

        def probed(*args, **kwargs):
            observed[step] = lock_is_held_for_other_threads()
            holds[step] = outermost_holds[0]
            return real(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, probed)

    wrap(stack.records, "save", "record")
    wrap(stack.storage, "save_precise", "precise")
    wrap(ImagePreciseAssetQueries, "insert", "row")
    wrap(stack.storage, "save", "png_and_thumbnail")
    wrap(stack.storage, "get_file_size_bytes", "size")

    stack.save(_EXACT_CASES["float32-rgb"]())

    steps = ["record", "precise", "row", "png_and_thumbnail", "size"]
    assert observed == dict.fromkeys(steps, True)
    assert holds == dict.fromkeys(steps, 1), "the steps must share one hold of the lock"
    assert outermost_holds[0] == 1
