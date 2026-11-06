"""16-bit PNG bulk download (`export_format: png16`): the real BulkDownloadService over a real image service, database
and disk storage, driven directly and through `POST /api/v1/images/download`."""

import io
import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from invokeai.app.services.bulk_download.bulk_download_common import BulkDownloadStatus
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.images.images_common import ImageExportFormat
from invokeai.app.services.invoker import Invoker
from invokeai.backend.image_util.raster import RasterChannels
from invokeai.backend.image_util.raster.png_codec import encode_png
from tests.app.routers.conftest import _auth, _create_board
from tests.app.routers.test_image_resolved_inputs import persisted_inputs  # noqa: F401
from tests.app.routers.test_images_precise_export import (
    _bit_depth_and_color_type,
    _export,
    _float_raster,
    _ordinary_upload,
    _precise_asset_row,
    _raster,
    _stored_png,
    _upload,
    api,  # noqa: F401
)


@pytest.fixture
def bulk(mock_invoker: Invoker, api: TestClient) -> Iterator[BulkDownloadService]:  # noqa: F811
    service = BulkDownloadService()
    service.start(mock_invoker)
    mock_invoker.services.bulk_download = service
    yield service
    service.stop()


def _zip(service: BulkDownloadService, item_id: str) -> ZipFile:
    return ZipFile(service.get_path(item_id + ".zip"))


def _wait_until_built(service: BulkDownloadService, item_name: str) -> BulkDownloadStatus:
    deadline = time.monotonic() + 20
    while (status := service.get_status(item_name)) is BulkDownloadStatus.BUILDING:
        assert time.monotonic() < deadline, "the job did not finish"
        time.sleep(0.01)
    return status


def _folder(service: BulkDownloadService) -> Path:
    return service._bulk_downloads_folder


def test_png16_archive_swaps_only_the_precise_images_and_keeps_the_gallery_files_of_the_rest(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    precise = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    ordinary = _ordinary_upload(api, user1_token)

    bulk.handler([precise, ordinary], None, "mixed", export_format=ImageExportFormat.PNG16)

    with _zip(bulk, "mixed") as archive:
        assert sorted(archive.namelist()) == sorted([f"general/{Path(precise).stem}-16bit.png", f"general/{ordinary}"])
        exported = archive.read(f"general/{Path(precise).stem}-16bit.png")
        assert exported == _export(api, user1_token, precise).content
        assert _bit_depth_and_color_type(exported) == (16, 6)
        assert archive.read(f"general/{ordinary}") == _stored_png(mock_invoker, ordinary)


def test_without_an_export_format_the_archive_holds_the_gallery_files(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    precise = _upload(api, user1_token, encode_png(_raster()))["image_name"]

    bulk.handler([precise], None, "plain")

    with _zip(bulk, "plain") as archive:
        assert archive.namelist() == [f"general/{precise}"]
        assert archive.read(f"general/{precise}") == _stored_png(mock_invoker, precise)


def _truncate_asset_body(mock_invoker: Invoker, name: str) -> None:
    """Keeps the asset's header (so its descriptor reads) and cuts the pixel data."""
    services = mock_invoker.services
    path = services.image_files.get_precise_path(name, services.image_records.get(name).image_subfolder)
    path.write_bytes(path.read_bytes()[:-20])


def _user_id(mock_invoker: Invoker) -> str:
    return mock_invoker.services.users.get_by_email("user1@test.com").user_id


def _warnings_naming(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and name in r.getMessage()]


def test_an_asset_with_a_corrupt_body_falls_back_to_the_gallery_file_with_one_warning(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    _truncate_asset_body(mock_invoker, name)
    assert _export(api, user1_token, name).status_code == 500  # the descriptor reads, the pixels do not

    with caplog.at_level(logging.WARNING):
        bulk.handler([name], None, "corrupt", export_format=ImageExportFormat.PNG16)

    with _zip(bulk, "corrupt") as archive:
        assert archive.namelist() == [f"general/{name}"]
        assert archive.read(f"general/{name}") == _stored_png(mock_invoker, name)
    assert len(_warnings_naming(caplog, name)) == 1


def test_non_finite_pixels_fall_back_to_the_gallery_file_with_one_warning(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = _ordinary_upload(api, user1_token)
    _precise_asset_row(mock_invoker, name, _float_raster(RasterChannels.RGB, float("nan")))

    with caplog.at_level(logging.WARNING):
        bulk.handler([name], None, "nan", export_format=ImageExportFormat.PNG16)

    with _zip(bulk, "nan") as archive:
        assert archive.namelist() == [f"general/{name}"]
    assert [m for m in _warnings_naming(caplog, name) if "non-finite" in m]


def test_the_archive_is_in_progress_until_complete_and_a_single_export_works_meanwhile(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    first = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    second = _upload(api, user1_token, encode_png(_raster(RasterChannels.RGB)))["image_name"]
    images = mock_invoker.services.images
    real_encode = images.encode_png16
    seen: list[tuple[int, str | None, int, bool, list[str]]] = []

    def encode_and_probe(image_name: str) -> bytes:
        # Mid-archive: the fetch endpoint is polled, a single export runs, and no final zip exists yet.
        poll = api.get("/api/v1/images/download/inflight.zip", headers=_auth(user1_token))
        single = _export(api, user1_token, first)
        seen.append(
            (
                poll.status_code,
                poll.headers.get("retry-after"),
                single.status_code,
                (_folder(bulk) / "inflight.zip").exists(),
                sorted(p.name for p in _folder(bulk).iterdir()),
            )
        )
        return real_encode(image_name)

    images.encode_png16 = encode_and_probe  # type: ignore[method-assign]
    try:
        bulk.handler(
            [first, second], None, "inflight", user_id=_user_id(mock_invoker), export_format=ImageExportFormat.PNG16
        )
    finally:
        del images.encode_png16

    assert [(code, retry, single, final) for code, retry, single, final, _ in seen] == [(202, "2", 200, False)] * 2
    assert all(files == ["inflight.zip.part"] for *_, files in seen)
    done = api.get("/api/v1/images/download/inflight.zip", headers=_auth(user1_token))
    assert done.status_code == 200
    with ZipFile(io.BytesIO(done.content)) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 2


def test_a_job_failing_mid_archive_leaves_no_file_and_is_reported_once(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    first = _ordinary_upload(api, user1_token)
    second = _ordinary_upload(api, user1_token)
    services = mock_invoker.services
    services.image_files.get_path(second, services.image_records.get(second).image_subfolder).unlink()

    with pytest.raises(FileNotFoundError):
        bulk.handler([first, second], None, "broken", user_id=_user_id(mock_invoker))

    assert list(_folder(bulk).iterdir()) == []
    first_get = api.get("/api/v1/images/download/broken.zip", headers=_auth(user1_token))
    second_get = api.get("/api/v1/images/download/broken.zip", headers=_auth(user1_token))
    assert first_get.status_code == 500
    assert second_get.status_code == 404


def _block_encodes_until_stopped(mock_invoker: Invoker, bulk: BulkDownloadService) -> threading.Event:
    """Makes every 16-bit encode wait until the service is stopped (or the job is released). Returns the event that
    is set once an encode is running."""
    images = mock_invoker.services.images
    real_encode = images.encode_png16
    entered = threading.Event()

    def blocked_encode(image_name: str) -> bytes:
        entered.set()
        bulk._stop_event.wait(10)
        return real_encode(image_name)

    images.encode_png16 = blocked_encode  # type: ignore[method-assign]
    return entered


def test_a_second_16_bit_job_is_refused_while_one_is_building(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    entered = _block_encodes_until_stopped(mock_invoker, bulk)
    body = {"image_names": [name]}
    headers = _auth(user1_token)
    try:
        running = api.post("/api/v1/images/download", json={**body, "export_format": "png16"}, headers=headers)
        assert entered.wait(10)

        refused = api.post("/api/v1/images/download", json={**body, "export_format": "png16"}, headers=headers)
        plain = api.post("/api/v1/images/download", json=body, headers=headers)
    finally:
        bulk.stop()  # ends the blocked job
        del mock_invoker.services.images.encode_png16

    assert running.status_code == 202
    assert refused.status_code == 503
    assert refused.headers["retry-after"]
    assert plain.status_code == 202


def test_stop_waits_for_the_image_being_encoded_then_leaves_no_part_file_or_directory(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    names = [_upload(api, user1_token, encode_png(_raster()))["image_name"] for _ in range(2)]
    images = mock_invoker.services.images
    real_encode = images.encode_png16
    entered = threading.Event()
    release = threading.Event()
    encode_ends_after = 2.5  # longer than a fixed join cap could cover

    def slow_encode(image_name: str) -> bytes:
        entered.set()
        release.wait(30)
        return real_encode(image_name)

    images.encode_png16 = slow_encode  # type: ignore[method-assign]
    timer = threading.Timer(encode_ends_after, release.set)
    try:
        bulk.start_job(names, None, "stoppable", _user_id(mock_invoker), ImageExportFormat.PNG16)
        assert entered.wait(10)
        folder = _folder(bulk)
        assert (folder / "stoppable.zip.part").exists()

        timer.start()
        started = time.monotonic()
        bulk.stop()
        elapsed = time.monotonic() - started
    finally:
        timer.cancel()
        release.set()
        del images.encode_png16

    assert elapsed >= encode_ends_after - 0.5  # stop waited for the encode instead of giving up
    assert not any(t.is_alive() for t in threading.enumerate() if t.name == "bulk-download-stoppable")
    assert not folder.parent.exists()  # the part file was gone, so the temp directory could be removed
    assert bulk.get_status("stoppable.zip") is BulkDownloadStatus.FAILED
    assert "stopped" in (bulk.pop_failure("stoppable.zip") or "")


def test_a_new_job_deletes_archives_nobody_fetched_for_an_hour(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    name = _ordinary_upload(api, user1_token)
    for item in ("old", "recent"):
        bulk.handler([name], None, item, user_id="someone")
    folder = _folder(bulk)
    two_hours_ago = time.time() - 7200
    os.utime(folder / "old.zip", (two_hours_ago, two_hours_ago))

    bulk.handler([name], None, "next")

    assert sorted(p.name for p in folder.iterdir()) == ["next.zip", "recent.zip"]
    assert bulk.get_owner("old.zip") is None
    assert bulk.get_owner("recent.zip") == "someone"


def test_a_board_download_applies_the_format(
    api: TestClient,  # noqa: F811
    mock_invoker: Invoker,
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    precise = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    ordinary = _ordinary_upload(api, user1_token)
    board_id = _create_board(api, user1_token)
    for name in (precise, ordinary):
        mock_invoker.services.board_image_records.add_image_to_board(board_id, name)

    bulk.handler(None, board_id, "board", export_format=ImageExportFormat.PNG16)

    with _zip(bulk, "board") as archive:
        assert sorted(archive.namelist()) == sorted([f"general/{Path(precise).stem}-16bit.png", f"general/{ordinary}"])


def test_the_download_route_passes_the_export_format_to_the_archive(
    api: TestClient,  # noqa: F811
    user1_token: str,
    bulk: BulkDownloadService,
) -> None:
    precise = _upload(api, user1_token, encode_png(_raster()))["image_name"]

    response = api.post(
        "/api/v1/images/download",
        json={"image_names": [precise], "export_format": "png16"},
        headers=_auth(user1_token),
    )
    rejected = api.post(
        "/api/v1/images/download",
        json={"image_names": [precise], "export_format": "jxl"},
        headers=_auth(user1_token),
    )

    assert response.status_code == 202
    item_name = response.json()["bulk_download_item_name"]
    assert _wait_until_built(bulk, item_name) is BulkDownloadStatus.READY
    with ZipFile(bulk.get_path(item_name)) as archive:
        assert archive.namelist() == [f"general/{Path(precise).stem}-16bit.png"]
    assert rejected.status_code == 422
