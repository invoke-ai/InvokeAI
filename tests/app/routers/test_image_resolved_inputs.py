"""Persisted PNG input records reach image inspection without changing editor restoration."""

import io
import json
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invokeai.app.api.routers.images import WorkflowAndGraphResponse
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.noise import NoiseInvocation
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveService
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutIDValidator
from tests.app.routers.conftest import MockApiDependencies, _auth


@pytest.fixture
def persisted_inputs(
    enable_multiuser: None, mock_invoker: Invoker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DiskImageFileStorage]:
    services = mock_invoker.services
    storage = DiskImageFileStorage(tmp_path / "images")
    services.image_files = storage
    storage.start(mock_invoker)
    services.images.start(mock_invoker)
    moves = ImageMoveService(services.database, storage, services.configuration, services.logger)
    services.image_moves = moves
    monkeypatch.setattr(
        "invokeai.app.api.routers.image_move_maintenance.ApiDependencies", MockApiDependencies(mock_invoker)
    )
    try:
        yield storage
    finally:
        moves.stop()


def _graph_strings(resolved: bool) -> tuple[str, str | None]:
    graph = Graph(
        nodes={"math": AddInvocation(id="math", a=0, b=1), "noise": NoiseInvocation(id="noise", seed=0)},
        edges=[
            Edge(
                source=EdgeConnection(node_id="math", field="value"),
                destination=EdgeConnection(node_id="noise", field="seed"),
            )
        ],
    )
    source_json = graph.model_dump_json()
    runtime = graph.model_copy(deep=True)
    runtime.nodes["math"].a = 17
    runtime.nodes["noise"].seed = 18
    resolved_json = runtime.model_dump_json() if resolved else None
    return source_json, resolved_json


def _save_png(
    invoker: Invoker, storage: DiskImageFileStorage, *, workflow: str | None, resolved: bool
) -> tuple[str, str | None]:
    source_json, resolved_json = _graph_strings(resolved)
    services = invoker.services
    user_id = services.users.get_by_email("user1@test.com").user_id
    services.image_records.save(
        image_name="image.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=2,
        height=2,
        has_workflow=True,
        user_id=user_id,
        image_subfolder="nested/record-path",
    )
    with Image.new("RGB", (2, 2), (12, 34, 56)) as image:
        storage.save(
            image,
            "image.png",
            workflow=workflow,
            graph=source_json,
            resolved_inputs=resolved_json,
            image_subfolder="nested/record-path",
        )
    # A fresh reader must recover from disk, using the record's non-flat location.
    services.image_files = DiskImageFileStorage(storage.image_root)
    return source_json, resolved_json


@pytest.mark.parametrize("crop,resize", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("override_metadata", [False, True], ids=["embedded-metadata", "form-metadata"])
@pytest.mark.parametrize("diagnostic", ["resolved", "legacy", "malformed", "wrong-envelope"])
def test_uploaded_png_preserves_generation_records(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    persisted_inputs: DiskImageFileStorage,
    crop: bool,
    resize: bool,
    override_metadata: bool,
    diagnostic: str,
) -> None:
    services = mock_invoker.services
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    source_json, resolved_json = _graph_strings(True)
    workflow_json = (
        json.dumps(
            {
                "name": "Original authoring workflow",
                "author": "",
                "description": "",
                "version": "1.0.0",
                "contact": "",
                "tags": "",
                "notes": "Preserve whitespace as well as content",
                "exposedFields": [],
                "meta": {"version": "3.0.0", "category": "user"},
                "nodes": [],
                "edges": [],
                "form": {"elements": {}, "rootElementId": "root"},
            },
            indent=2,
        )
        + "\n"
    )
    WorkflowWithoutIDValidator.validate_json(workflow_json)
    embedded_metadata = {"seed": 18, "resolved_inputs": "user-owned", "invokeai_resolved_inputs": {"note": "keep"}}
    form_metadata = {"seed": 99, "resolved_inputs": {"custom": True}, "invokeai_resolved_inputs": "also user-owned"}
    incoming_inputs = {
        "resolved": resolved_json,
        "legacy": None,
        "malformed": "not JSON",
        "wrong-envelope": '{"nodes": [], "edges": []}',
    }[diagnostic]
    # Produce the input PNG through the real image service and writer, then upload its actual bytes.
    user_id = services.users.get_by_email("user1@test.com").user_id
    with Image.new("RGBA", (4, 4)) as image:
        image.paste((12, 34, 56, 255), (1, 1, 3, 3))
        original = services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            user_id=user_id,
            metadata=json.dumps(embedded_metadata),
            workflow=workflow_json,
            graph=source_json,
            resolved_inputs=incoming_inputs,
        )
    original_record = services.image_records.get(original.image_name)
    original_path = persisted_inputs.get_path(original.image_name, image_subfolder=original_record.image_subfolder)
    original_png = original_path.read_bytes()
    form = {"metadata": json.dumps(form_metadata)} if override_metadata else {}
    if resize:
        form["resize_to"] = '{"width": 8, "height": 6}'
    uploaded = client.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": "false", "crop_visible": str(crop).lower()},
        headers=_auth(user1_token),
        files={"file": (original.image_name, original_png, "image/png")},
        data=form,
    )
    assert uploaded.status_code == 201, uploaded.text
    image_name = uploaded.json()["image_name"]
    assert image_name != original.image_name
    record = services.image_records.get(image_name)
    assert services.image_records.get_user_id(image_name) == user_id
    assert record.has_workflow
    expected_metadata = form_metadata if override_metadata else embedded_metadata
    expected_inputs = resolved_json if diagnostic == "resolved" else None
    # A new storage instance cannot serve the writer's populated Image.info cache.
    reader = DiskImageFileStorage(persisted_inputs.image_root)
    services.image_files = reader
    with reader.get(image_name, image_subfolder=record.image_subfolder) as reopened:
        assert reopened.format == "PNG"
        assert reopened.size == ((8, 6) if resize else (2, 2) if crop else (4, 4))
        assert json.loads(reopened.info["invokeai_metadata"]) == expected_metadata
        assert reopened.info["invokeai_graph"] == source_json
        assert reopened.info["invokeai_workflow"] == workflow_json
        disk_inputs = reopened.info.get("invokeai_resolved_inputs")
    response = client.get(f"/api/v1/images/i/{image_name}/workflow", headers=_auth(user1_token))
    assert response.status_code == 200, response.text
    assert response.json() == {"workflow": workflow_json, "graph": source_json, "resolved_inputs": expected_inputs}
    assert disk_inputs == expected_inputs
    if expected_inputs is not None:
        inputs = json.loads(disk_inputs)
        assert inputs["nodes"]["noise"]["seed"] == 18
        assert inputs["nodes"]["math"]["a"] == 17
    assert original_path.read_bytes() == original_png


def _png_chunks(png: bytes) -> list[bytes]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = []
    offset = 8
    while offset < len(png):
        end = offset + 12 + int.from_bytes(png[offset : offset + 4], "big")
        chunks.append(png[offset:end])
        offset = end
    assert offset == len(png)
    assert chunks[-1][4:8] == b"IEND"
    return chunks


@pytest.mark.parametrize("crop", [False, True], ids=["plain", "crop"])
def test_upload_preserves_post_idat_png_records(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    persisted_inputs: DiskImageFileStorage,
    crop: bool,
) -> None:
    services = mock_invoker.services
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    source, resolved = _graph_strings(True)
    workflow = (
        '{ "name": "Trailing text workflow", "author": "", "description": "", "version": "1.0.0", '
        '"contact": "", "tags": "", "notes": "", "exposedFields": [], '
        '"meta": {"version": "3.0.0", "category": "user"}, "nodes": [], "edges": [] }\n'
    )
    WorkflowWithoutIDValidator.validate_json(workflow)
    metadata = '{ "seed": 18, "resolved_inputs": "user-owned" }'
    records = {
        "invokeai_metadata": metadata,
        "invokeai_workflow": workflow,
        "invokeai_graph": source,
        "invokeai_resolved_inputs": resolved,
    }
    with Image.new("RGBA", (4, 4)) as image:
        image.paste((12, 34, 56, 255), (1, 1, 3, 3))
        persisted_inputs.save(
            image, "post-idat.png", metadata=metadata, workflow=workflow, graph=source, resolved_inputs=resolved
        )
    original_path = persisted_inputs.get_path("post-idat.png")
    chunks = _png_chunks(original_path.read_bytes())
    text_chunks = [chunk for chunk in chunks if chunk[4:8] == b"tEXt"]
    assert len(text_chunks) == 4
    # Move whole tEXt chunks, retaining their payloads, lengths and CRCs after all IDATs and before IEND.
    body = [chunk for chunk in chunks[:-1] if chunk[4:8] != b"tEXt"]
    assert body[-1][4:8] == b"IDAT"
    original_png = b"\x89PNG\r\n\x1a\n" + b"".join(body + text_chunks + chunks[-1:])
    original_path.write_bytes(original_png)
    with Image.open(io.BytesIO(original_png)) as image:
        assert not records.keys() & image.info.keys()
        image.verify()
    with Image.open(io.BytesIO(original_png)) as image:
        image.load()
        assert image.info == records
        assert image.getpixel((1, 1)) == (12, 34, 56, 255)
    uploaded = client.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": "false", "crop_visible": str(crop).lower()},
        headers=_auth(user1_token),
        files={"file": (original_path.name, original_png, "image/png")},
    )
    assert uploaded.status_code == 201, uploaded.text
    name = uploaded.json()["image_name"]
    assert name != original_path.name
    record = services.image_records.get(name)
    reader = DiskImageFileStorage(persisted_inputs.image_root)
    services.image_files = reader
    with reader.get(name, image_subfolder=record.image_subfolder) as image:
        assert image.size == ((2, 2) if crop else (4, 4))
        assert image.getpixel((0, 0) if crop else (1, 1)) == (12, 34, 56, 255)
        disk_records = dict(image.info)
    response = client.get(f"/api/v1/images/i/{name}/workflow", headers=_auth(user1_token))
    assert response.status_code == 200, response.text
    assert response.json() == {"workflow": workflow, "graph": source, "resolved_inputs": resolved}
    assert disk_records == records
    assert json.loads(disk_records["invokeai_resolved_inputs"])["nodes"]["noise"]["seed"] == 18
    assert original_path.read_bytes() == original_png


def test_upload_rejects_undecodable_idat_as_failed_image_read(
    client: TestClient, mock_invoker: Invoker, user1_token: str, persisted_inputs: DiskImageFileStorage
) -> None:
    mock_invoker.services.names = SimpleNameService()
    mock_invoker.services.urls = LocalUrlService()
    png = io.BytesIO()
    with Image.new("RGB", (2, 2), (12, 34, 56)) as image:
        image.save(png, "PNG")
    chunks = _png_chunks(png.getvalue())
    for index, chunk in enumerate(chunks):
        if chunk[4:8] == b"IDAT":
            payload = b"\x00" * (len(chunk) - 12)
            data = b"IDAT" + payload
            chunks[index] = len(payload).to_bytes(4, "big") + data + zlib.crc32(data).to_bytes(4, "big")
    broken = b"\x89PNG\r\n\x1a\n" + b"".join(chunks)
    # Valid container/CRCs and header, but invalid compressed pixels: Image.open alone cannot catch this.
    with Image.open(io.BytesIO(broken)) as image:
        image.verify()
    with Image.open(io.BytesIO(broken)) as image, pytest.raises(OSError):
        image.load()
    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": "false"},
        headers=_auth(user1_token),
        files={"file": ("broken.png", broken, "image/png")},
    )
    assert response.status_code == 415, response.text
    assert response.json() == {"detail": "Failed to read image"}
    assert list(persisted_inputs.image_root.rglob("*.png")) == []


@pytest.mark.parametrize("workflow", [None, '{ "name": "original workflow", "nodes": [], "edges": [] }'])
@pytest.mark.parametrize("resolved", [True, False], ids=["resolved", "legacy"])
def test_workflow_http_preserves_source_and_exposes_persisted_inputs(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    persisted_inputs: DiskImageFileStorage,
    workflow: str | None,
    resolved: bool,
) -> None:
    source_json, resolved_json = _save_png(mock_invoker, persisted_inputs, workflow=workflow, resolved=resolved)
    path = persisted_inputs.get_path("image.png", image_subfolder="nested/record-path")
    before = path.read_bytes()
    response = client.get("/api/v1/images/i/image.png/workflow", headers=_auth(user1_token))
    assert response.status_code == 200, response.text
    assert response.json() == {"workflow": workflow, "graph": source_json, "resolved_inputs": resolved_json}
    source = Graph.model_validate_json(response.json()["graph"])
    assert source.nodes["noise"].seed == 0
    assert source.nodes["math"].a == 0
    assert len(source.edges) == 1
    if resolved:
        inputs = Graph.model_validate_json(response.json()["resolved_inputs"])
        assert inputs.nodes["noise"].seed == 18
        assert inputs.nodes["math"].a == 17
    else:
        # Responses predating the optional channel (including video callers) still deserialize.
        old_response = WorkflowAndGraphResponse.model_validate({"workflow": workflow, "graph": source_json})
        assert old_response.model_dump() == response.json()
    assert path.read_bytes() == before


@pytest.mark.parametrize("resolved", [True, False], ids=["resolved", "legacy"])
def test_image_service_reads_inputs_from_record_subfolder(
    mock_invoker: Invoker, user1_token: str, persisted_inputs: DiskImageFileStorage, resolved: bool
) -> None:
    source_json, resolved_json = _save_png(mock_invoker, persisted_inputs, workflow=None, resolved=resolved)
    images = mock_invoker.services.images
    assert images.get_resolved_inputs("image.png") == resolved_json
    assert images.get_graph("image.png") == source_json
    assert images.get_workflow("image.png") is None


def test_input_inspection_preserves_read_access_and_maintenance_guards(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    user2_token: str,
    persisted_inputs: DiskImageFileStorage,
) -> None:
    _save_png(mock_invoker, persisted_inputs, workflow=None, resolved=True)
    endpoint = "/api/v1/images/i/image.png/workflow"
    assert client.get(endpoint).status_code == 401
    assert client.get(endpoint, headers=_auth(user2_token)).status_code == 403
    with mock_invoker.services.image_moves.reserve_gallery_maintenance():
        # Read denial must still precede the maintenance check.
        assert client.get(endpoint, headers=_auth(user2_token)).status_code == 403
        response = client.get(endpoint, headers=_auth(user1_token))
        assert response.status_code == 409
        assert response.json()["detail"] == "Image storage maintenance is active"
    response = client.get(endpoint, headers=_auth(user1_token))
    assert response.status_code == 200
    assert Graph.model_validate_json(response.json()["resolved_inputs"]).nodes["noise"].seed == 18


def test_missing_record_or_file_keeps_service_errors_and_http_404(
    client: TestClient, mock_invoker: Invoker, user1_token: str, persisted_inputs: DiskImageFileStorage
) -> None:
    images = mock_invoker.services.images
    with pytest.raises(ImageRecordNotFoundException):
        images.get_resolved_inputs("missing.png")
    assert client.get("/api/v1/images/i/missing.png/workflow", headers=_auth(user1_token)).status_code == 404
    _save_png(mock_invoker, persisted_inputs, workflow=None, resolved=True)
    persisted_inputs.get_path("image.png", image_subfolder="nested/record-path").unlink()
    with pytest.raises(ImageFileNotFoundException):
        images.get_resolved_inputs("image.png")
    assert client.get("/api/v1/images/i/image.png/workflow", headers=_auth(user1_token)).status_code == 404
