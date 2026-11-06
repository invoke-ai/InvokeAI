"""Precise PNG export (`/export-formats`, `/export`) and the precision-preserving 16-bit PNG upload, over a real image
service, database and disk storage."""

import io
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.api.routers import images as images_router
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageExportBusyError
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.convert import quantize_unit
from invokeai.backend.image_util.raster.png_codec import PngText, decode_png, encode_png
from invokeai.backend.image_util.raster.precise_asset import (
    PRECISE_ASSET_FORMAT,
    PRECISE_ASSET_VERSION,
    write_precise_asset,
)
from tests.app.routers.conftest import _auth
from tests.app.routers.test_image_resolved_inputs import _graph_strings, persisted_inputs  # noqa: F401

_WORKFLOW = json.dumps(
    {
        "name": "Round trip",
        "author": "",
        "description": "",
        "version": "1.0.0",
        "contact": "",
        "tags": "",
        "notes": "",
        "exposedFields": [],
        "meta": {"version": "3.0.0", "category": "user"},
        "nodes": [],
        "edges": [],
        "form": {"elements": {}, "rootElementId": "root"},
    }
)
_METADATA = json.dumps({"seed": 18, "positive_prompt": "a fox"})


def _icc(color_space: bytes = b"RGB ") -> bytes:
    profile = bytearray(256)
    profile[16:20] = color_space
    return bytes(profile)


def _raster(channels: RasterChannels = RasterChannels.RGBA, icc: bytes | None = None) -> RasterImage:
    count = len(channels.value)
    pixels = np.random.default_rng(3).integers(0, 65536, size=(6, 9, count), dtype=np.uint16)
    # Not a multiple of 257, so no 8-bit value represents it.
    pixels[0, 0] = np.arange(1, count + 1) * 1001
    alpha = AlphaMode.STRAIGHT if channels is RasterChannels.RGBA else AlphaMode.NONE
    descriptor = RasterDescriptor(channels, ColorEncoding.SRGB, ValueRange.UNIT, alpha, count, icc)
    return RasterImage.from_array(pixels, descriptor)


@pytest.fixture
def api(client: TestClient, mock_invoker: Invoker, user1_token: str, persisted_inputs: DiskImageFileStorage):  # noqa: F811
    services = mock_invoker.services
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    services.configuration.precise_image_storage = True
    return client


def _upload(client: TestClient, token: str, data: bytes, content_type: str = "image/png", **params) -> dict:
    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": "false", **params},
        files={"file": ("upload", data, content_type)},
        headers=_auth(token),
    )
    assert response.status_code == 201, response.text
    return response.json()


def _stored_png_path(invoker: Invoker, image_name: str) -> Path:
    services = invoker.services
    record = services.image_records.get(image_name)
    return services.image_files.get_path(image_name, image_subfolder=record.image_subfolder)


def _stored_png(invoker: Invoker, image_name: str) -> bytes:
    return _stored_png_path(invoker, image_name).read_bytes()


def _bit_depth_and_color_type(png: bytes) -> tuple[int, int]:
    assert png[12:16] == b"IHDR"
    return png[24], png[25]


def _upload_text() -> PngText:
    source, resolved = _graph_strings(True)
    return PngText(
        invokeai_metadata=_METADATA,
        invokeai_workflow=_WORKFLOW,
        invokeai_graph=source,
        invokeai_resolved_inputs=resolved,
    )


def _export(client: TestClient, token: str, name: str, fmt: str = "png16"):
    return client.get(f"/api/v1/images/i/{name}/export", params={"format": fmt}, headers=_auth(token))


def test_a_16_bit_png_upload_keeps_its_exact_pixels_and_records(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    services = mock_invoker.services
    raster = _raster(icc=_icc())
    text = _upload_text()

    dto = _upload(api, user1_token, encode_png(raster, text))

    name = dto["image_name"]
    assert services.images.has_precise_asset(name)
    stored = services.images.get_raster(name)
    assert stored.dtype == np.uint16
    assert stored.view().tolist() == raster.view().tolist()
    assert stored.descriptor.icc_profile == raster.descriptor.icc_profile
    with Image.open(io.BytesIO(_stored_png(mock_invoker, name))) as derivative:
        assert derivative.mode == "RGBA"
        assert np.array_equal(np.asarray(derivative), quantize_unit(raster, np.uint8).view())
    recalled = api.get(f"/api/v1/images/i/{name}/workflow", headers=_auth(user1_token)).json()
    assert recalled == {
        "workflow": _WORKFLOW,
        "graph": text.invokeai_graph,
        "resolved_inputs": text.invokeai_resolved_inputs,
    }
    assert services.images.get_metadata(name).model_dump() == json.loads(_METADATA)


def test_export_after_import_round_trips_the_pixels_and_text(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    raster = _raster(RasterChannels.RGB, icc=_icc())
    text = _upload_text()
    name = _upload(api, user1_token, encode_png(raster, text))["image_name"]

    formats = api.get(f"/api/v1/images/i/{name}/export-formats", headers=_auth(user1_token))
    exported = _export(api, user1_token, name)

    assert formats.status_code == 200
    assert formats.json() == {"formats": ["png16"]}
    assert exported.status_code == 200
    assert exported.headers["content-type"] == "image/png"
    assert exported.headers["cache-control"] == "private, no-store"  # multiuser: the same policy as /full
    assert exported.headers["content-disposition"] == f'attachment; filename="{Path(name).stem}-16bit.png"'
    assert _bit_depth_and_color_type(exported.content) == (16, 2)
    decoded = decode_png(exported.content)
    assert decoded.raster.view().tolist() == quantize_unit(raster, np.uint16).view().tolist()
    assert decoded.raster.descriptor.icc_profile == raster.descriptor.icc_profile
    assert decoded.text == text
    # Pillow reads the same strings, so a later upload of the export recalls them.
    with Image.open(io.BytesIO(exported.content)) as reopened:
        assert reopened.info["invokeai_workflow"] == text.invokeai_workflow
        assert reopened.info["invokeai_metadata"] == text.invokeai_metadata


def test_float_precise_assets_export_as_display_quantized_16_bit(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    services = mock_invoker.services
    values = np.random.default_rng(1).uniform(-0.5, 1.5, size=(5, 4, 3)).astype(np.float32)
    descriptor = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3)
    raster = RasterImage.from_array(values, descriptor)
    user_id = services.users.get_by_email("user1@test.com").user_id
    dto = services.images.create_from_raster(raster, ResourceOrigin.INTERNAL, ImageCategory.GENERAL, user_id=user_id)

    exported = _export(api, user1_token, dto.image_name)

    assert exported.status_code == 200
    assert decode_png(exported.content).raster.view().tolist() == quantize_unit(raster, np.uint16).view().tolist()


def test_an_ordinary_image_has_no_export_formats_and_refuses_export(api: TestClient, user1_token: str) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (9, 8, 7)).save(buffer, format="PNG")
    name = _upload(api, user1_token, buffer.getvalue())["image_name"]

    formats = api.get(f"/api/v1/images/i/{name}/export-formats", headers=_auth(user1_token))

    assert formats.json() == {"formats": []}
    assert _export(api, user1_token, name).status_code == 409


def test_export_works_with_precise_storage_turned_off_for_an_existing_asset(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    raster = _raster()
    name = _upload(api, user1_token, encode_png(raster))["image_name"]
    mock_invoker.services.configuration.precise_image_storage = False

    exported = _export(api, user1_token, name)

    assert exported.status_code == 200
    assert decode_png(exported.content).raster.view().tolist() == raster.view().tolist()


def test_a_second_export_is_refused_as_busy_without_waiting_and_the_slot_is_freed_after_each_response(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]

    held = mock_invoker.services.images.export_png16(name)  # holds the single export slot
    busy = _export(api, user1_token, name)
    held.release()
    first = _export(api, user1_token, name)
    second = _export(api, user1_token, name)

    assert busy.status_code == 503
    assert busy.headers["retry-after"] == "1"
    assert first.status_code == 200
    assert second.status_code == 200  # the background task released the slot after the first body was sent


def test_the_export_streams_the_whole_file_in_chunks_and_frees_the_slot_afterwards(
    api: TestClient, mock_invoker: Invoker, user1_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    reference = mock_invoker.services.images.export_png16(name)
    expected = reference.content
    reference.release()
    monkeypatch.setattr("invokeai.app.api.routers.images._EXPORT_CHUNK_BYTES", 64)
    assert len(expected) > 64 * 3

    response = _export(api, user1_token, name)

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-length"] == str(len(expected))
    mock_invoker.services.images.export_png16(name).release()  # the slot is free again


def test_the_slot_stays_taken_until_the_body_has_been_streamed(
    api: TestClient, mock_invoker: Invoker, user1_token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    monkeypatch.setattr("invokeai.app.api.routers.images._EXPORT_CHUNK_BYTES", 64)
    real_iter_chunks = images_router._iter_chunks
    busy_mid_stream: list[bool] = []

    async def probing_iter_chunks(content: bytes):
        chunks = real_iter_chunks(content)
        yield await anext(chunks)
        try:
            mock_invoker.services.images.export_png16(name).release()
            busy_mid_stream.append(False)
        except ImageExportBusyError:
            busy_mid_stream.append(True)
        async for chunk in chunks:
            yield chunk

    monkeypatch.setattr(images_router, "_iter_chunks", probing_iter_chunks)

    response = _export(api, user1_token, name)

    assert response.status_code == 200
    assert busy_mid_stream == [True]
    mock_invoker.services.images.export_png16(name).release()  # free once the response is complete


def test_the_export_takes_its_text_without_decoding_the_gallery_png(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    text = _upload_text()
    name = _upload(api, user1_token, encode_png(_raster(), text))["image_name"]
    path = _stored_png_path(mock_invoker, name)
    png = path.read_bytes()
    first_idat = png.index(b"IDAT") - 4
    iend = png.index(b"IEND") - 4
    # Valid chunk structure and CRC, but not a zlib stream: any pixel decode of this file fails.
    path.write_bytes(png[:first_idat] + _chunk(b"IDAT", bytes(40)) + png[iend:])
    # The storage cached the image when it saved it; without this, a decoding read would never see the new file.
    mock_invoker.services.image_files.evict_cache_paths([path])
    with pytest.raises((OSError, SyntaxError)):
        Image.open(path).load()

    exported = _export(api, user1_token, name)

    assert exported.status_code == 200
    assert decode_png(exported.content).text == text


def _precise_asset_row(invoker: Invoker, name: str, raster: RasterImage | None, content: bytes | None = None) -> None:
    """Gives an ordinary image a precise asset row, with a written asset, raw file content, or no file at all."""
    services = invoker.services
    subfolder = services.image_records.get(name).image_subfolder
    path = services.image_files.get_precise_path(name, subfolder)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raster is not None:
        write_precise_asset(raster, path)
    elif content is not None:
        path.write_bytes(content)
    with services.database.queries.transaction() as q:
        q.image_precise_assets.insert(
            image_name=name, asset_format=PRECISE_ASSET_FORMAT, format_version=PRECISE_ASSET_VERSION
        )


def _ordinary_upload(api: TestClient, token: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (9, 8, 7)).save(buffer, format="PNG")
    return _upload(api, token, buffer.getvalue())["image_name"]


def _float_raster(channels: RasterChannels = RasterChannels.RGB, value: float = 0.5) -> RasterImage:
    count = len(channels.value)
    alpha = AlphaMode.STRAIGHT if channels in (RasterChannels.LA, RasterChannels.RGBA) else AlphaMode.NONE
    descriptor = RasterDescriptor(channels, ColorEncoding.SRGB, ValueRange.UNIT, alpha, count)
    return RasterImage.from_array(np.full((4, 4, count), value, dtype=np.float32), descriptor)


def test_an_asset_with_non_finite_pixels_is_refused_with_the_reason(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    name = _ordinary_upload(api, user1_token)
    _precise_asset_row(mock_invoker, name, _float_raster(value=float("nan")))

    response = _export(api, user1_token, name)

    assert response.status_code == 409
    assert "non-finite" in response.json()["detail"]


def test_an_asset_the_png_writer_cannot_encode_is_refused_with_the_codec_reason(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    name = _ordinary_upload(api, user1_token)
    _precise_asset_row(mock_invoker, name, _float_raster(RasterChannels.LA))

    response = _export(api, user1_token, name)

    assert response.status_code == 409
    assert "only L, RGB and RGBA" in response.json()["detail"]


def test_a_missing_precise_file_answers_404_and_is_logged(
    api: TestClient, mock_invoker: Invoker, user1_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    name = _ordinary_upload(api, user1_token)
    _precise_asset_row(mock_invoker, name, None)

    with caplog.at_level(logging.ERROR):
        response = _export(api, user1_token, name)

    assert response.status_code == 404
    assert any("Failed to read precise asset" in r.getMessage() for r in caplog.records)
    assert api.get(f"/api/v1/images/i/{name}/export-formats", headers=_auth(user1_token)).json() == {"formats": []}


def test_a_corrupt_precise_file_answers_500_and_is_logged_and_frees_the_slot(
    api: TestClient, mock_invoker: Invoker, user1_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    name = _ordinary_upload(api, user1_token)
    _precise_asset_row(mock_invoker, name, None, b"not a safetensors file")

    with caplog.at_level(logging.ERROR):
        response = _export(api, user1_token, name)
    formats = api.get(f"/api/v1/images/i/{name}/export-formats", headers=_auth(user1_token))

    assert response.status_code == 500
    assert any("Failed to read precise asset" in r.getMessage() for r in caplog.records)
    assert formats.json() == {"formats": []}
    mock_invoker.services.images.export_png16(_upload(api, user1_token, encode_png(_raster()))["image_name"]).release()


def test_export_reports_unknown_images_and_unknown_formats(api: TestClient, user1_token: str) -> None:
    assert api.get("/api/v1/images/i/missing.png/export-formats", headers=_auth(user1_token)).status_code == 404
    assert _export(api, user1_token, "missing.png").status_code == 404
    assert _export(api, user1_token, "missing.png", "jxl").status_code == 422


def test_another_user_cannot_list_or_export_a_private_image(
    api: TestClient, user1_token: str, user2_token: str
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]

    formats = api.get(f"/api/v1/images/i/{name}/export-formats", headers=_auth(user2_token))

    assert formats.status_code == 403
    assert _export(api, user2_token, name).status_code == 403


def test_export_is_refused_during_image_storage_maintenance(
    api: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    name = _upload(api, user1_token, encode_png(_raster()))["image_name"]
    mock_invoker.services.image_moves.is_maintenance_active = lambda: True

    response = _export(api, user1_token, name)

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"


def _plain(fmt: str) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (8, 8))
    image.putdata([(i * 3 % 256, i, 255 - i) for i in range(64)])
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("make", "content_type", "params", "precise_on"),
    [
        (lambda: encode_png(_raster()), "image/png", {}, False),
        (lambda: _plain("PNG"), "image/png", {}, True),
        (lambda: encode_png(_raster(RasterChannels.L)), "image/png", {}, True),
        (lambda: _plain("JPEG"), "image/jpeg", {}, True),
        (lambda: _plain("WEBP"), "image/webp", {}, True),
        (lambda: encode_png(_raster()), "image/png", {"crop_visible": "true"}, True),
    ],
    ids=["precise-off", "png8", "gray16", "jpeg", "webp", "crop-visible"],
)
def test_other_uploads_take_the_ordinary_path(
    api: TestClient, mock_invoker: Invoker, user1_token: str, make, content_type: str, params: dict, precise_on: bool
) -> None:
    services = mock_invoker.services
    services.configuration.precise_image_storage = precise_on
    data = make()

    dto = _upload(api, user1_token, data, content_type, **params)

    assert not services.images.has_precise_asset(dto["image_name"])
    # The stored PNG is exactly what an ordinary save of the opened image writes.
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        if params:
            source = source.crop(source.getbbox())
        reference = services.images.create(source, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL)
    assert _stored_png(mock_invoker, dto["image_name"]) == _stored_png(mock_invoker, reference.image_name)


def test_resize_to_takes_the_ordinary_path(api: TestClient, mock_invoker: Invoker, user1_token: str) -> None:
    response = api.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": "false"},
        files={"file": ("upload.png", encode_png(_raster()), "image/png")},
        data={"resize_to": '{"width": 8, "height": 8}'},
        headers=_auth(user1_token),
    )

    assert response.status_code == 201, response.text
    assert not mock_invoker.services.images.has_precise_asset(response.json()["image_name"])


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def test_a_16_bit_png_the_codec_rejects_is_stored_without_precision_and_warns_once(
    api: TestClient, mock_invoker: Invoker, user1_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    png = encode_png(_raster())
    # Pillow skips ancillary chunks it does not know; the codec bounds their number. 33 = signature + IHDR chunk.
    data = png[:33] + b"".join(_chunk(b"prVt", b"x") for _ in range(300)) + png[33:]
    with Image.open(io.BytesIO(data)) as check:
        check.load()

    with caplog.at_level(logging.WARNING):
        dto = _upload(api, user1_token, data)

    assert not mock_invoker.services.images.has_precise_asset(dto["image_name"])
    with Image.open(io.BytesIO(_stored_png(mock_invoker, dto["image_name"]))) as stored:
        assert stored.mode == "RGBA"
    assert len([r for r in caplog.records if "without precision" in r.getMessage()]) == 1


def test_a_16_bit_gray_alpha_png_is_stored_as_an_ordinary_image_with_one_warning(
    api: TestClient, mock_invoker: Invoker, user1_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    width, height = 3, 2
    rows = b"".join(bytes(1) + struct.pack(">6H", *range(1000, 7000, 1000)) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 4, 0, 0, 0)
    signature = encode_png(_raster())[:8]
    data = signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b"")

    with caplog.at_level(logging.WARNING):
        dto = _upload(api, user1_token, data)

    assert not mock_invoker.services.images.has_precise_asset(dto["image_name"])
    assert len([r for r in caplog.records if "gray+alpha" in r.getMessage()]) == 1


def test_a_16_bit_png_with_a_mismatching_icc_profile_is_stored_without_precision(
    api: TestClient, mock_invoker: Invoker, user1_token: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = encode_png(_raster(RasterChannels.RGB, icc=_icc(b"GRAY")))

    with caplog.at_level(logging.WARNING):
        dto = _upload(api, user1_token, data)

    assert not mock_invoker.services.images.has_precise_asset(dto["image_name"])
    assert len([r for r in caplog.records if "without precision" in r.getMessage()]) == 1
