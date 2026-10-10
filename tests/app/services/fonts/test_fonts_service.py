"""Service-level coverage for the account-scoped font library."""

import io
import logging
import subprocess
import threading
from pathlib import Path

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from sqlalchemy import select

from invokeai.app.services.fonts import fonts_default
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
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import fonts as font_queries
from invokeai.app.services.shared.database.queries.fonts import FontQueries, UploadedFont
from invokeai.app.services.shared.database.schema.fonts import fonts as fonts_table
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from tests.fixtures.races import when_called


@pytest.fixture
def font_service(tmp_path: Path, database: Database) -> FontService:
    service = FontService(
        database,
        fonts_dir=tmp_path / "fonts",
        storage_dir=tmp_path / "uploaded-fonts",
        logger=logging.getLogger("font-tests"),
    )
    yield service
    service.stop()


@pytest.fixture
def font_bytes() -> bytes:
    return (Path(__file__).parents[4] / "invokeai" / "assets" / "fonts" / "inter" / "Inter-Regular.ttf").read_bytes()


def _create_user(database: Database, email: str) -> str:
    users = UserService(database)
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


def test_private_uploads_are_deduplicated_and_isolated(
    font_service: FontService, database: Database, font_bytes: bytes
) -> None:
    alice = _create_user(database, "alice-fonts@test.com")
    bob = _create_user(database, "bob-fonts@test.com")

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
    font_service: FontService, database: Database, font_bytes: bytes
) -> None:
    users = UserService(database)
    user_id = _create_user(database, "deleted-font-owner@test.com")

    other_user_id = _create_user(database, "kept-font-owner@test.com")

    private = font_service.upload(user_id=user_id, filename="Private.ttf", data=font_bytes).font
    shared = font_service.upload(user_id="system", filename="Shared.ttf", data=font_bytes, scope=FontScope.SHARED).font
    other = font_service.upload(user_id=other_user_id, filename="Other.ttf", data=font_bytes).font
    assert private.storage_path is not None
    assert shared.storage_path is not None
    assert other.storage_path is not None
    private_path = font_service._storage_dir / private.storage_path
    shared_path = font_service._storage_dir / shared.storage_path
    other_path = font_service._storage_dir / other.storage_path
    assert private_path.is_file()
    assert shared_path.is_file()

    font_storage_paths = font_service.prepare_user_cleanup(user_id)
    users.delete(user_id)
    font_service.cleanup_user(user_id, font_storage_paths)

    # A private upload is the owner's own copy, even of content shared already.
    assert other.scope == FontScope.PRIVATE and other.id != shared.id
    assert font_storage_paths == (private.storage_path,)
    assert not private_path.exists()
    assert shared_path.is_file()
    assert other_path.is_file()
    assert font_service.get(private.id) is None
    assert font_service.get(shared.id) is not None
    assert font_service.get(other.id) is not None


def test_orphan_sweep_removes_only_unreferenced_managed_files(font_service: FontService, font_bytes: bytes) -> None:
    uploaded = font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes).font
    assert uploaded.storage_path is not None
    orphan = font_service._storage_dir / f"font_{'0' * 32}.ttf"
    interrupted = font_service._storage_dir / f".font_{'1' * 32}.ttf.{'2' * 32}.tmp"
    operator_file = font_service._storage_dir / "notes.txt"
    for path in (orphan, interrupted, operator_file):
        path.write_bytes(b"x")

    font_service.cleanup_orphaned_files()

    assert (font_service._storage_dir / uploaded.storage_path).is_file()
    assert not orphan.exists()
    assert not interrupted.exists()
    assert operator_file.is_file()


def test_shared_uploads_are_visible_and_directory_fonts_are_read_only(
    font_service: FontService, database: Database, font_bytes: bytes, tmp_path: Path
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
    # Deleting an upload never deletes a directory font of the same id.
    assert database.queries.fonts.delete_upload(directory[0].id) is False
    assert font_service.get(directory[0].id) is not None


def test_validation_and_quota_do_not_leave_files(
    font_service: FontService, database: Database, font_bytes: bytes
) -> None:
    validated = font_service.validate("Inter-Regular.ttf", font_bytes)
    assert validated.content_hash
    assert font_service.list(user_id="system")[1] == 0

    tiny_quota = FontService(
        database,
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

    revision = font_service.revision
    directory_file.unlink()
    assert font_service.rescan_directory() == 0
    assert font_service.list(user_id="system")[1] == 0
    assert font_service.revision > revision


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
    font_service: FontService, database: Database, font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = _create_user(database, "inflight-font-owner@test.com")
    uploaded = font_service.upload(user_id=user_id, filename="Variable.ttf", data=_variable_font_bytes()).font
    assert [instance.name for instance in uploaded.instances] == ["Bold"]
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


def _catalog_upload(
    database: Database, font_id: str, family: str, *, owner_id: str | None, label: str = "Regular", digit: str = "a"
) -> None:
    """Catalogs an upload without a file: listing reads the catalog only."""
    database.queries.fonts.insert_upload(
        UploadedFont(
            id=font_id,
            owner_id=owner_id,
            scope=FontScope.PRIVATE if owner_id is not None else FontScope.SHARED,
            filename=f"{font_id}.ttf",
            storage_path=f"{font_id}.ttf",
            family=family,
            label=label,
            style="normal",
            weight=400,
            content_hash=digit * 64,
            byte_size=100,
            axes=(),
            instances=(),
        )
    )


def test_listing_filters_by_account_scope_and_hash_and_orders_ignoring_case(
    font_service: FontService, database: Database, font_bytes: bytes, tmp_path: Path
) -> None:
    alice = _create_user(database, "alice-listing@test.com")
    bob = _create_user(database, "bob-listing@test.com")
    _catalog_upload(database, "font_b", "beta", owner_id=alice)
    _catalog_upload(database, "font_a2", "alpha", owner_id=alice, label="Bold", digit="b")
    _catalog_upload(database, "font_a1", "Alpha", owner_id=None, label="bold", digit="c")
    _catalog_upload(database, "font_a0", "ALPHA", owner_id=None, label="Bold", digit="d")
    _catalog_upload(database, "font_bob", "Aardvark", owner_id=bob, digit="e")
    directory_file = tmp_path / "fonts" / "Inter.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1
    inter = font_service.list(user_id=bob, scope=FontScope.SHARED, search="inter")[0][0]

    def ids(**kwargs: object) -> tuple[list[str], int]:
        records, total = font_service.list(user_id=alice, **kwargs)  # type: ignore[arg-type]
        return [record.id for record in records], total

    # Family, then label, ignoring case; then id. Bob's private font is not Alice's to see.
    assert ids() == (["font_a0", "font_a1", "font_a2", "font_b", inter.id], 5)
    assert ids(offset=1, limit=2) == (["font_a1", "font_a2"], 5)
    assert ids(offset=4, limit=10) == ([inter.id], 5)
    assert ids(offset=9, limit=10) == ([], 5)
    assert ids(limit=0) == ([], 5)
    assert ids(scope=FontScope.PRIVATE) == (["font_a2", "font_b"], 2)
    assert ids(scope=FontScope.SHARED) == (["font_a0", "font_a1", inter.id], 3)
    assert ids(content_hash="c" * 64) == (["font_a1"], 1)
    assert ids(content_hash="e" * 64) == ([], 0)
    # Full pages, which count the fonts with a statement of their own.
    assert ids(scope=FontScope.PRIVATE, limit=1) == (["font_a2"], 2)
    assert ids(scope=FontScope.SHARED, offset=1, limit=1) == (["font_a1"], 3)
    assert ids(content_hash="c" * 64, limit=0) == ([], 1)
    assert font_service.list(user_id=bob, scope=FontScope.PRIVATE)[1] == 1


def test_listing_search_matches_text_ignoring_case(font_service: FontService, database: Database) -> None:
    _catalog_upload(database, "font_percent", "Grotesk 100%", owner_id=None, digit="a")
    _catalog_upload(database, "font_thousand", "Grotesk 1000", owner_id=None, digit="b")
    _catalog_upload(database, "font_label", "Serif", owner_id=None, label="Condensed_Bold", digit="c")
    _catalog_upload(database, "font_space", "Serif", owner_id=None, label="Condensed Bold", digit="d")

    def found(search: str) -> list[str]:
        return [record.id for record in font_service.list(user_id="system", search=search)[0]]

    assert found("grotesk") == ["font_percent", "font_thousand"]
    # `%` and `_` match only themselves.
    assert found("100%") == ["font_percent"]
    assert found("condensed_") == ["font_label"]
    assert found("FONT_SPACE.TTF") == ["font_space"]
    first, total = font_service.list(user_id="system", search="grotesk", limit=1)
    assert ([record.id for record in first], total) == (["font_percent"], 2)


def test_quota_counts_each_place_of_uploads_on_its_own(
    font_service: FontService, database: Database, font_bytes: bytes
) -> None:
    alice = _create_user(database, "alice-quota@test.com")
    bob = _create_user(database, "bob-quota@test.com")
    one_font = FontService(
        database,
        fonts_dir=font_service._fonts_dir,
        storage_dir=font_service._storage_dir,
        max_library_bytes=len(font_bytes),
    )
    try:
        assert one_font.upload(user_id=alice, filename="Inter.ttf", data=font_bytes).created
        # Neither Alice's upload counts toward the shared fonts, nor do they count toward Bob's.
        assert one_font.upload(user_id=alice, filename="Inter.ttf", data=font_bytes, scope=FontScope.SHARED).created
        assert one_font.upload(user_id=bob, filename="Inter.ttf", data=font_bytes).created
        with pytest.raises(FontQuotaExceededError):
            one_font.upload(user_id=alice, filename="Variable.ttf", data=_variable_font_bytes())
        with pytest.raises(FontQuotaExceededError):
            one_font.upload(user_id=bob, filename="Variable.ttf", data=_variable_font_bytes(), scope=FontScope.SHARED)
    finally:
        one_font.stop()


def test_rescan_changes_the_catalog_only_when_the_directory_changed(
    font_service: FontService, font_bytes: bytes, tmp_path: Path
) -> None:
    directory_file = tmp_path / "fonts" / "Face.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1
    indexed = font_service.list(user_id="system")[0]
    revision = font_service.revision

    assert font_service.rescan_directory() == 1
    assert font_service.revision == revision
    assert font_service.list(user_id="system")[0] == indexed

    directory_file.write_bytes(_variable_font_bytes())
    assert font_service.rescan_directory() == 1
    assert font_service.revision > revision
    [replaced] = font_service.list(user_id="system")[0]
    assert replaced.id == indexed[0].id
    assert (replaced.family, replaced.source_path) == ("Fixture Variable", "Face.ttf")
    assert replaced.content_hash != indexed[0].content_hash
    assert [axis.tag for axis in replaced.axes] == ["wght"]
    assert [instance.name for instance in replaced.instances] == ["Bold"]
    # The catalog holds the variable font as found, so another scan changes nothing.
    revision = font_service.revision
    assert font_service.rescan_directory() == 1
    assert font_service.revision == revision


def test_rescan_updates_changed_fonts_and_deletes_only_missing_ones(
    font_service: FontService, database: Database, font_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fonts"
    root.mkdir()
    for name in ("Missing.ttf", "Changed.ttf", "Kept.ttf"):
        (root / name).write_bytes(font_bytes)
    assert font_service.rescan_directory() == 3
    before = _directory_rows(database)

    monkeypatch.setattr(font_queries, "now_text", lambda: "2031-02-03 04:05:06.789")
    (root / "Missing.ttf").unlink()
    (root / "Changed.ttf").write_bytes(_variable_font_bytes())
    assert font_service.rescan_directory() == 2

    after = _directory_rows(database)
    assert set(after) == {"Changed.ttf", "Kept.ttf"}
    assert after["Kept.ttf"] == before["Kept.ttf"]
    assert after["Changed.ttf"][0] == "Fixture Variable"
    assert after["Changed.ttf"][1] == "2031-02-03 04:05:06.789"


def _directory_rows(database: Database) -> dict[str, tuple[str, str]]:
    columns = fonts_table.c
    statement = select(columns.source_path, columns.family, columns.updated_at).where(columns.source == "directory")
    with database.begin(write=False) as conn:
        return {path: (family, updated_at) for path, family, updated_at in conn.execute(statement).all()}


def test_rescan_skips_a_path_longer_than_the_catalog_holds(
    font_service: FontService, font_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fonts"
    (root / "nested").mkdir(parents=True)
    (root / "Short.ttf").write_bytes(font_bytes)
    (root / "nested" / "Long.ttf").write_bytes(font_bytes)
    monkeypatch.setattr(fonts_default, "MAX_SOURCE_PATH_LENGTH", len("nested/Long.tt"))

    assert font_service.rescan_directory() == 1
    assert [record.source_path for record in font_service.list(user_id="system")[0]] == ["Short.ttf"]


def test_upload_removes_its_file_only_when_its_row_is_missing(
    font_service: FontService, font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_insert = FontQueries.insert_upload

    def refused(self: FontQueries, font: UploadedFont) -> None:
        raise RuntimeError("refused")

    monkeypatch.setattr(FontQueries, "insert_upload", refused)
    with pytest.raises(RuntimeError, match="refused"):
        font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes)
    assert not list(font_service._storage_dir.glob("font_*"))

    # A commit that took effect but reported a failure, as when the connection drops during it.
    def lost_reply(self: FontQueries, font: UploadedFont) -> None:
        real_insert(self, font)
        raise ConnectionError("lost")

    monkeypatch.setattr(FontQueries, "insert_upload", lost_reply)
    with pytest.raises(ConnectionError):
        font_service.upload(user_id="system", filename="Inter.ttf", data=font_bytes)
    [record] = font_service.list(user_id="system")[0]
    assert record.storage_path is not None
    assert (font_service._storage_dir / record.storage_path).is_file()


def test_rescans_run_one_at_a_time(
    font_service: FontService, font_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_file = tmp_path / "fonts" / "Inter.ttf"
    directory_file.parent.mkdir(parents=True)
    directory_file.write_bytes(font_bytes)
    assert font_service.rescan_directory() == 1

    # The second scan starts once the first has read the catalog, and waits for it to finish.
    ended = when_called(monkeypatch, FontQueries, "directory_fonts", font_service.rescan_directory)
    assert font_service.rescan_directory() == 1
    assert ended() == []
    assert font_service.list(user_id="system")[1] == 1
