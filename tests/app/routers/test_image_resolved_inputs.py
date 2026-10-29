"""Persisted PNG input records reach image inspection without changing editor restoration."""

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
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph
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


def _save_png(
    invoker: Invoker, storage: DiskImageFileStorage, *, workflow: str | None, resolved: bool
) -> tuple[str, str | None]:
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
