"""Service-level coverage for the account-scoped font library."""

import io
import logging
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables.TupleVariation import TupleVariation

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.fonts.fonts_common import FontScope, FontSource, FontUploadResult
from invokeai.app.services.fonts.fonts_default import (
    MAX_INDEXED_FONT_BYTES,
    FontChangedError,
    FontDeleteForbiddenError,
    FontNotFoundError,
    FontQuotaExceededError,
    FontService,
    FontValidationError,
)
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService


@pytest.fixture
def font_service(tmp_path: Path) -> FontService:
    config = InvokeAIAppConfig(use_memory_db=True)
    db = init_db(config, logging.getLogger("font-tests"), Mock(spec=ImageFileStorageBase))
    service = FontService(
        db=db,
        fonts_dir=tmp_path / "fonts",
        storage_dir=tmp_path / "uploaded-fonts",
        logger=logging.getLogger("font-tests"),
    )
    yield service
    service.stop()


@pytest.fixture
def font_bytes() -> bytes:
    return (Path(__file__).parents[4] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf").read_bytes()


def _create_user(service: FontService, email: str) -> str:
    users = UserService(service._db)
    return users.create(UserCreateRequest(email=email, password="TestPass123", display_name=email)).user_id


def _variable_font_bytes() -> bytes:
    """Build a tiny variable TTF so the worker test has no external fixture dependency."""
    glyph_order = [".notdef", "space", "A"]
    glyphs = {}
    for glyph_name in glyph_order:
        pen = TTGlyphPen(None)
        pen.moveTo((100, 100))
        pen.lineTo((100, 800))
        pen.lineTo((500, 800))
        pen.lineTo((500, 100))
        pen.closePath()
        glyphs[glyph_name] = pen.glyph()

    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(glyph_order)
    builder.setupCharacterMap({32: "space", 65: "A"})
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({".notdef": (600, 0), "space": (500, 0), "A": (500, 100)})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(
        {
            "familyName": "Fixture Variable",
            "styleName": "Regular",
            "uniqueFontIdentifier": "Fixture Variable Regular",
            "fullName": "Fixture Variable Regular",
            "psName": "FixtureVariable-Regular",
            "version": "Version 1.0",
        }
    )
    builder.setupOS2(sTypoAscender=800, usWinAscent=800, usWinDescent=200, usWeightClass=400)
    builder.setupPost()
    builder.setupFvar(
        [("wght", 100, 400, 900, "Weight")],
        [{"location": {"wght": 700}, "stylename": "Bold"}],
    )
    # Four outline points plus four phantom points for A. The right edge widens as wght rises.
    builder.setupGvar(
        {
            "A": [
                TupleVariation(
                    {"wght": (0.0, 1.0, 1.0)},
                    [(0, 0), (0, 0), (100, 0), (100, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                )
            ]
        }
    )
    output = io.BytesIO()
    builder.save(output)
    return output.getvalue()


def test_private_uploads_are_deduplicated_and_isolated(font_service: FontService, font_bytes: bytes) -> None:
    alice = _create_user(font_service, "alice-fonts@test.com")
    bob = _create_user(font_service, "bob-fonts@test.com")

    first = font_service.upload(user_id=alice, filename="Inter-Regular.ttf", data=font_bytes)
    duplicate = font_service.upload(user_id=alice, filename="renamed.ttf", data=font_bytes)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.font.id == first.font.id
    assert font_service.list(user_id=alice)[1] == 1
    assert font_service.list(user_id=bob)[1] == 0
    with pytest.raises(FontNotFoundError):
        font_service.get_accessible(user_id=bob, font_id=first.font.id)


def test_user_cleanup_removes_private_files_but_preserves_shared_files(
    font_service: FontService, font_bytes: bytes
) -> None:
    users = UserService(font_service._db)
    user_id = _create_user(font_service, "deleted-font-owner@test.com")

    private = font_service.upload(user_id=user_id, filename="Private.ttf", data=font_bytes).font
    shared = font_service.upload(user_id="system", filename="Shared.ttf", data=font_bytes, scope=FontScope.SHARED).font
    assert private.storage_path is not None
    assert shared.storage_path is not None
    private_path = font_service._storage_dir / private.storage_path
    shared_path = font_service._storage_dir / shared.storage_path
    assert private_path.is_file()
    assert shared_path.is_file()

    font_storage_paths = font_service.prepare_user_cleanup(user_id)
    users.delete(user_id)
    font_service.cleanup_user(user_id, font_storage_paths)

    assert not private_path.exists()
    assert shared_path.is_file()
    assert font_service.get(private.id) is None
    assert font_service.get(shared.id) is not None


def test_shared_uploads_are_visible_and_directory_fonts_are_read_only(
    font_service: FontService, font_bytes: bytes, tmp_path: Path
) -> None:
    shared = font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes, scope=FontScope.SHARED)
    assert shared.font.scope == FontScope.SHARED
    assert font_service.list(user_id="another-user", scope=FontScope.SHARED)[1] == 1

    directory_file = tmp_path / "fonts" / "nested" / "Inter.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1
    directory = font_service.list(user_id="another-user", scope=FontScope.SHARED)[0]
    assert directory[0].source == FontSource.DIRECTORY

    with pytest.raises(FontDeleteForbiddenError):
        font_service.delete(user_id="system", font_id=directory[0].id, is_admin=True)


def test_validation_and_quota_do_not_leave_files(font_service: FontService, font_bytes: bytes) -> None:
    validated = font_service.validate("Inter-Regular.ttf", font_bytes)
    assert validated.content_hash
    assert font_service.list(user_id="system")[1] == 0

    tiny_quota = FontService(
        db=font_service._db,
        fonts_dir=font_service._fonts_dir,
        storage_dir=font_service._storage_dir / "quota",
        max_library_bytes=len(font_bytes) - 1,
    )
    try:
        with pytest.raises(FontQuotaExceededError):
            tiny_quota.upload(user_id="system", filename="Inter.ttf", data=font_bytes)
        assert not list((font_service._storage_dir / "quota").glob("*"))
    finally:
        tiny_quota.stop()


def test_rescan_removes_missing_directory_rows_and_detects_replacement(
    font_service: FontService, font_bytes: bytes, tmp_path: Path
) -> None:
    directory_file = tmp_path / "fonts" / "Inter.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1
    record = font_service.list(user_id="system")[0][0]

    directory_file.write_bytes(b"changed")
    with pytest.raises(FontChangedError):
        font_service.read_file(user_id="system", font_id=record.id, expected_hash=record.content_hash)

    directory_file.unlink()
    assert font_service.rescan_directory() == 0
    assert font_service.list(user_id="system")[1] == 0


def test_rescan_clears_index_when_directory_is_removed(
    font_service: FontService, font_bytes: bytes, tmp_path: Path
) -> None:
    directory_file = tmp_path / "fonts" / "Inter.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1
    assert font_service.list(user_id="system")[1] == 1

    directory_file.unlink()
    directory_file.parent.rmdir()
    assert font_service.rescan_directory() == 0
    assert font_service.list(user_id="system")[1] == 0


def test_rescan_rejects_oversized_directory_files_before_open(
    font_service: FontService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "fonts" / "oversized.ttf"
    oversized.parent.mkdir(parents=True)
    with oversized.open("wb") as file:
        file.truncate(MAX_INDEXED_FONT_BYTES + 1)

    original_open = Path.open
    opened = False

    def guarded_open(path: Path, *args: object, **kwargs: object):
        nonlocal opened
        if path == oversized:
            opened = True
            raise AssertionError("oversized directory file must be rejected before opening")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert font_service.rescan_directory() == 0
    assert opened is False
    assert font_service.list(user_id="system")[1] == 0


def test_cleanup_cannot_delete_upload_that_starts_during_sweep(
    font_service: FontService, font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup snapshot and upload publish sequence share one mutation lock."""
    validation = font_service.validate("Inter.ttf", font_bytes)
    font_service._storage_dir.mkdir(parents=True)
    validation_started = threading.Event()
    write_finished = threading.Event()
    sweep_started = threading.Event()
    allow_sweep = threading.Event()
    cleanup_errors: list[BaseException] = []
    upload_result: list[FontUploadResult | BaseException] = []

    def validate(_filename: str, _data: bytes):
        validation_started.set()
        return validation

    original_write = font_service._write_atomic

    def write_atomic(target: Path, data: bytes) -> None:
        original_write(target, data)
        write_finished.set()

    original_iterdir = Path.iterdir

    def blocked_iterdir(path: Path):
        if path == font_service._storage_dir and not sweep_started.is_set():
            sweep_started.set()
            assert allow_sweep.wait(timeout=5)
        return original_iterdir(path)

    monkeypatch.setattr(font_service, "validate", validate)
    monkeypatch.setattr(font_service, "_write_atomic", write_atomic)
    monkeypatch.setattr(Path, "iterdir", blocked_iterdir)

    def cleanup() -> None:
        try:
            font_service.cleanup_orphaned_files()
        except BaseException as error:  # pragma: no cover - surfaced below
            cleanup_errors.append(error)

    def upload() -> None:
        try:
            upload_result.append(font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes))
        except BaseException as error:  # pragma: no cover - surfaced below
            upload_result.append(error)

    cleanup_thread = threading.Thread(target=cleanup)
    upload_thread = threading.Thread(target=upload)
    cleanup_thread.start()
    assert sweep_started.wait(timeout=5)
    upload_thread.start()
    try:
        assert validation_started.wait(timeout=5)
        assert not write_finished.wait(timeout=0.25)
    finally:
        allow_sweep.set()
        cleanup_thread.join(timeout=5)
        upload_thread.join(timeout=5)

    assert cleanup_errors == []
    assert upload_result and not isinstance(upload_result[0], BaseException)
    assert write_finished.is_set()
    uploaded = upload_result[0]
    assert isinstance(uploaded, FontUploadResult)
    assert uploaded.font.storage_path is not None
    assert (font_service._storage_dir / uploaded.font.storage_path).is_file()


def test_variable_instances_are_pinned_and_cached(font_service: FontService, tmp_path: Path) -> None:
    data = _variable_font_bytes()
    destination = tmp_path / "fonts" / "FixtureVariable.ttf"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data)
    assert font_service.rescan_directory() == 1
    record = font_service.list(user_id="system")[0][0]

    _, default_data = font_service.instance(
        user_id="system", font_id=record.id, content_hash=record.content_hash, coordinates={"wght": 400}
    )
    _, bold_data = font_service.instance(
        user_id="system", font_id=record.id, content_hash=record.content_hash, coordinates={"wght": 850}
    )
    _, cached_bold_data = font_service.instance(
        user_id="system", font_id=record.id, content_hash=record.content_hash, coordinates={"wght": 850}
    )

    assert default_data != data
    assert bold_data != default_data
    assert cached_bold_data == bold_data


def test_deleted_font_cannot_repopulate_instance_cache_after_worker_finishes(
    font_service: FontService, font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = _create_user(font_service, "inflight-font-owner@test.com")
    uploaded = font_service.upload(user_id=user_id, filename="Variable.ttf", data=_variable_font_bytes()).font
    font_service.instance(
        user_id=user_id,
        font_id=uploaded.id,
        content_hash=uploaded.content_hash,
        coordinates={"wght": 400},
    )
    assert font_service._instance_cache
    worker_started = threading.Event()
    allow_worker = threading.Event()
    worker_errors: list[BaseException] = []

    def blocked_worker(operation: str, args: tuple[object, ...], *, error_type: type[FontValidationError]):
        del args, error_type
        assert operation == "instance"
        worker_started.set()
        assert allow_worker.wait(timeout=5)
        return b"derived-instance"

    monkeypatch.setattr(font_service, "_run_worker", blocked_worker)

    def run_instance() -> None:
        try:
            font_service.instance(
                user_id=user_id,
                font_id=uploaded.id,
                content_hash=uploaded.content_hash,
                coordinates={"wght": 700},
            )
        except BaseException as error:  # pragma: no cover - surfaced below
            worker_errors.append(error)

    instance_thread = threading.Thread(target=run_instance)
    instance_thread.start()
    assert worker_started.wait(timeout=5)
    font_service.delete(user_id=user_id, font_id=uploaded.id)
    allow_worker.set()
    instance_thread.join(timeout=5)

    assert not instance_thread.is_alive()
    assert worker_errors and isinstance(worker_errors[0], FontNotFoundError)
    assert not font_service._instance_cache


def test_failed_worker_is_cleaned_up_before_next_job(font_service: FontService, font_bytes: bytes) -> None:
    with pytest.raises(FontValidationError):
        font_service.validate("broken.ttf", b"not-a-font")
    assert not font_service._active_workers

    uploaded = font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes)
    assert uploaded.created is True


def test_timed_out_worker_is_killed_and_slots_are_released(
    font_service: FontService, font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HangingProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        def communicate(self, input: bytes | None = None, timeout: float | None = None):
            del input
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="font-worker", timeout=timeout)
            return b"", b""

        def poll(self) -> int | None:
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True

    hanging = HangingProcess()
    with monkeypatch.context() as patch:
        patch.setattr("invokeai.app.services.fonts.fonts_default.subprocess.Popen", lambda *args, **kwargs: hanging)
        with pytest.raises(FontValidationError, match="timed out"):
            font_service.validate("Inter.ttf", font_bytes)
    assert hanging.killed
    assert not font_service._active_workers

    # This call proves the worker admission and active-slot semaphores were both released.
    uploaded = font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes)
    assert uploaded.created is True
