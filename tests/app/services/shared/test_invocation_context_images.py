from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.api.extract_metadata import extract_metadata_from_image
from invokeai.app.invocations.math import AddInvocation, RandomIntInvocation
from invokeai.app.services.board_access.board_access_common import (
    BoardAccessDeniedError,
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID


def _make_interface(allowed: bool) -> tuple[ImagesInterface, MagicMock]:
    """The board access service decides; these tests check how generation asks it and acts on the answer."""
    services = MagicMock()
    services.configuration.multiuser = True
    services.users.get.return_value = MagicMock(is_admin=False)
    if not allowed:
        services.board_access.require.side_effect = BoardAccessDeniedError("board", BoardAction.AddMedia)
        services.board_access.require_image.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
        services.board_access.require_video.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.session.graph = None
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation.is_intermediate = False
    return ImagesInterface(services, data, MagicMock()), services


def test_image_save_rejects_a_board_the_user_cannot_add_to() -> None:
    images, services = _make_interface(allowed=False)

    with pytest.raises(PermissionError, match="not authorized"):
        images.save(MagicMock(), board_id="foreign-board")

    services.images.create.assert_not_called()


def test_image_save_allows_a_board_the_user_can_add_to() -> None:
    images, services = _make_interface(allowed=True)

    images.save(MagicMock(), board_id="public-board")

    services.images.create.assert_called_once()
    services.board_access.require.assert_called_once_with("public-board", "queue-user", False, BoardAction.AddMedia)


def test_image_save_allows_any_board_in_single_user_mode() -> None:
    images, services = _make_interface(allowed=False)
    services.configuration.multiuser = False
    services.users.get.return_value = None

    images.save(MagicMock(), board_id="foreign-board")

    services.images.create.assert_called_once()


@pytest.mark.parametrize(
    ("method_name", "service_method"),
    [
        ("get_pil", "get_pil_image"),
        ("get_metadata", "get_metadata"),
        ("get_dto", "get_dto"),
        ("get_path", "get_path"),
    ],
)
def test_image_reads_reject_foreign_private_image(method_name: str, service_method: str) -> None:
    images, services = _make_interface(allowed=False)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "foreign-board"

    with pytest.raises(PermissionError, match="not authorized"):
        getattr(images, method_name)("private-image")

    getattr(services.images, service_method).assert_not_called()


def test_image_dto_allows_foreign_shared_image() -> None:
    images, services = _make_interface(allowed=True)
    services.image_records.get_user_id.return_value = "owner"
    services.board_image_records.get_board_for_image.return_value = "shared-board"

    images.get_dto("shared-image")

    services.images.get_dto.assert_called_once_with("shared-image")
    services.board_access.require_image.assert_called_once_with("shared-image", "queue-user", False, ItemAction.Read)


def test_image_read_allows_any_image_in_single_user_mode() -> None:
    images, services = _make_interface(allowed=False)
    services.configuration.multiuser = False
    services.users.get.return_value = None

    images.get_dto("foreign-image")

    services.images.get_dto.assert_called_once_with("foreign-image")


def test_image_read_rejects_deactivated_queue_user() -> None:
    """A deactivated account keeps no queue-time privileges, even for its own images."""
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = MagicMock(is_admin=False, is_active=False)
    services.image_records.get_user_id.return_value = "queue-user"

    with pytest.raises(PermissionError, match="not authorized"):
        images.get_dto("own-image")

    services.images.get_dto.assert_not_called()


def test_image_read_rejects_deleted_queue_user() -> None:
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = None

    with pytest.raises(PermissionError, match="not authorized"):
        images.get_dto("any-image")

    services.images.get_dto.assert_not_called()


@pytest.mark.parametrize(
    "account",
    [pytest.param(MagicMock(is_admin=False, is_active=False), id="deactivated"), pytest.param(None, id="deleted")],
)
def test_image_save_rejects_a_deactivated_or_deleted_queue_user(account: MagicMock | None) -> None:
    """No output may be saved on behalf of such an account, even uncategorized."""
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = account

    with pytest.raises(PermissionError, match="not authorized"):
        images.save(MagicMock())

    services.images.create.assert_not_called()


def test_image_save_allows_active_queue_user_without_board() -> None:
    images, services = _make_interface(allowed=False)
    services.users.get.return_value = MagicMock(is_admin=False, is_active=True)

    images.save(MagicMock())

    services.images.create.assert_called_once()


def test_images_save_attributes_the_output_to_the_queue_items_project() -> None:
    images, services = _make_interface(allowed=True)

    images.save(MagicMock())

    assert services.images.create.call_args.kwargs["project_id"] == "project-1"


def test_image_save_graph_records_resolved_runtime_inputs() -> None:
    graph = Graph()
    graph.add_node(RandomIntInvocation(id="random", low=17, high=18))
    graph.add_node(AddInvocation(id="add", b=1))
    graph.add_node(AddInvocation(id="downstream", b=2))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="random", field="value"),
            destination=EdgeConnection(node_id="add", field="a"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="add", field="value"),
            destination=EdgeConnection(node_id="downstream", field="a"),
        )
    )
    source_graph_json = graph.model_dump_json()
    session = GraphExecutionState(graph=graph)

    random_node = session.next()
    assert isinstance(random_node, RandomIntInvocation)
    session.complete(random_node.id, random_node.invoke(MagicMock()))
    runtime_add_node = session.next()
    assert isinstance(runtime_add_node, AddInvocation)
    assert runtime_add_node.a == 17

    services = MagicMock()
    services.configuration.multiuser = False
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.root_item_id = None
    data.queue_item._workflow_json_loaded = False
    data.queue_item._workflow_json_snapshot = None
    data.queue_item.session = session
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation = runtime_add_node
    images = ImagesInterface(services, data, MagicMock())

    images.save(MagicMock())

    saved_kwargs = services.images.create.call_args.kwargs
    saved_graph = Graph.model_validate_json(saved_kwargs["graph"])
    saved_add_node = saved_graph.nodes[next(iter(session.source_prepared_mapping["add"]))]
    assert isinstance(saved_add_node, AddInvocation)
    assert saved_add_node.a == 17
    assert saved_kwargs["node_id"] == saved_add_node.id
    downstream_id = next(iter(session.source_prepared_mapping["downstream"]))
    assert set(saved_graph.nodes) == {random_node.id, runtime_add_node.id, downstream_id}
    saved_downstream_node = saved_graph.nodes[downstream_id]
    assert isinstance(saved_downstream_node, AddInvocation)
    assert saved_downstream_node.b == 2
    assert sorted(
        (edge.source.node_id, edge.source.field, edge.destination.node_id, edge.destination.field)
        for edge in saved_graph.edges
    ) == sorted(
        [
            (random_node.id, "value", runtime_add_node.id, "a"),
            (runtime_add_node.id, "value", downstream_id, "a"),
        ]
    )
    assert graph.nodes["add"].a == 0
    assert session.graph.model_dump_json() == source_graph_json
    assert graph.model_dump_json() == source_graph_json
    assert saved_kwargs["workflow"] is None
    assert data.queue_item.workflow is None


def test_image_save_graph_falls_back_to_source_graph_before_materialization() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=2, b=3))
    session = GraphExecutionState(graph=graph)
    assert not session.execution_graph.nodes

    services = MagicMock()
    services.configuration.multiuser = False
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.root_item_id = None
    data.queue_item._workflow_json_loaded = False
    data.queue_item._workflow_json_snapshot = None
    data.queue_item.session = session
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation = graph.nodes["add"]
    images = ImagesInterface(services, data, MagicMock())

    images.save(MagicMock())

    saved_graph_json = services.images.create.call_args.kwargs["graph"]
    saved_graph = Graph.model_validate_json(saved_graph_json)
    saved_add_node = saved_graph.nodes["add"]
    assert isinstance(saved_add_node, AddInvocation)
    assert (saved_add_node.a, saved_add_node.b) == (2, 3)
    assert saved_graph == graph
    assert not session.execution_graph.nodes


def test_image_save_graph_png_round_trip(tmp_path: Path) -> None:
    session = GraphExecutionState(
        graph=Graph(
            nodes={
                "random": RandomIntInvocation(id="random", low=17, high=18),
                "add": AddInvocation(id="add", b=1),
                "downstream": AddInvocation(id="downstream", b=2),
            },
            edges=[
                Edge(
                    source=EdgeConnection(node_id="random", field="value"),
                    destination=EdgeConnection(node_id="add", field="a"),
                ),
                Edge(
                    source=EdgeConnection(node_id="add", field="value"),
                    destination=EdgeConnection(node_id="downstream", field="a"),
                ),
            ],
        )
    )
    random_node = session.next()
    assert isinstance(random_node, RandomIntInvocation)
    session.complete(random_node.id, random_node.invoke(MagicMock()))
    runtime_add_node = session.next()
    assert isinstance(runtime_add_node, AddInvocation)
    assert runtime_add_node.a == 17
    assert runtime_add_node.id != "add"

    workflow = WorkflowWithoutID.model_validate(
        {
            "name": "Reusable image workflow",
            "author": "",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "",
            "notes": "Keep the authoring workflow separate from the resolved runtime graph.",
            "exposedFields": [],
            "meta": {"category": "user", "version": "3.0.0"},
            "nodes": [],
            "edges": [],
        }
    )
    workflow_json = workflow.model_dump_json()
    services = MagicMock()
    services.configuration.multiuser = False
    services.configuration.pil_compress_level = 6
    storage = DiskImageFileStorage(tmp_path / "outputs")
    invoker = MagicMock()
    invoker.services = services
    storage.start(invoker)

    # Replace only ImageService's record/DTO boundary; PNG persistence and extraction are real.
    def create_image(
        image: Image.Image, metadata: str | None, workflow: str | None, graph: str | None, **kwargs: object
    ) -> None:
        storage.save(image=image, image_name="runtime.png", metadata=metadata, workflow=workflow, graph=graph)

    services.images.create.side_effect = create_image
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = workflow
    data.queue_item.root_item_id = None
    data.queue_item._workflow_json_loaded = False
    data.queue_item._workflow_json_snapshot = None
    data.queue_item.session = session
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation = runtime_add_node
    images = ImagesInterface(services, data, MagicMock())

    with Image.new("RGB", (2, 2), color=(12, 34, 56)) as image:
        images.save(image)

    # A fresh reader cannot return the writer's cached image with its populated image.info.
    reader = DiskImageFileStorage(storage.image_root)
    with reader.get("runtime.png") as reopened:
        assert (reopened.format, reopened.size, reopened.getpixel((0, 0))) == ("PNG", (2, 2), (12, 34, 56))
        extracted = extract_metadata_from_image(
            pil_image=reopened,
            invokeai_metadata_override=None,
            invokeai_workflow_override=None,
            invokeai_graph_override=None,
            logger=services.logger,
        )

    assert extracted.invokeai_graph is not None
    saved_graph = Graph.model_validate_json(extracted.invokeai_graph)
    saved_node_id = services.images.create.call_args.kwargs["node_id"]
    assert saved_node_id == runtime_add_node.id
    saved_add_node = saved_graph.nodes[saved_node_id]
    assert isinstance(saved_add_node, AddInvocation)
    assert (saved_add_node.id, saved_add_node.a, saved_add_node.b) == (runtime_add_node.id, 17, 1)
    downstream_id = next(iter(session.source_prepared_mapping["downstream"]))
    assert set(saved_graph.nodes) == {random_node.id, runtime_add_node.id, downstream_id}
    assert sorted(
        (edge.source.node_id, edge.source.field, edge.destination.node_id, edge.destination.field)
        for edge in saved_graph.edges
    ) == sorted(
        [
            (random_node.id, "value", runtime_add_node.id, "a"),
            (runtime_add_node.id, "value", downstream_id, "a"),
        ]
    )
    assert extracted.invokeai_workflow == workflow_json
    assert data.queue_item.workflow.model_dump_json() == workflow_json
