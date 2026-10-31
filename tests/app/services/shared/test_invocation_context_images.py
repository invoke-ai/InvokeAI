import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.api.extract_metadata import extract_metadata_from_image
from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.fields import LatentsField
from invokeai.app.invocations.image import BlankImageInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.math import AddInvocation, RandomIntInvocation
from invokeai.app.invocations.noise import NoiseInvocation, NoiseOutput
from invokeai.app.services.board_access.board_access_common import (
    BoardAccessDeniedError,
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Edge,
    EdgeConnection,
    Graph,
    GraphExecutionState,
    IterateInvocation,
)
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from tests.test_nodes import AnyTypeTestInvocation


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
    execution_graph_json = session.execution_graph.model_dump_json()

    images.save(MagicMock())

    saved_kwargs = services.images.create.call_args.kwargs
    assert saved_kwargs["graph"] == source_graph_json
    saved_graph = Graph.model_validate_json(saved_kwargs["resolved_inputs"])
    saved_add_node = saved_graph.nodes[next(iter(session.source_prepared_mapping["add"]))]
    assert isinstance(saved_add_node, AddInvocation)
    assert saved_add_node.a == 17
    assert saved_kwargs["node_id"] == saved_add_node.id
    assert set(saved_graph.nodes) == {random_node.id, runtime_add_node.id}
    assert sorted(
        (edge.source.node_id, edge.source.field, edge.destination.node_id, edge.destination.field)
        for edge in saved_graph.edges
    ) == sorted(
        [
            (random_node.id, "value", runtime_add_node.id, "a"),
        ]
    )
    assert graph.nodes["add"].a == 0
    assert session.graph.model_dump_json() == source_graph_json
    assert graph.model_dump_json() == source_graph_json
    assert session.execution_graph.model_dump_json() == execution_graph_json
    assert saved_kwargs["workflow"] is None
    assert data.queue_item.workflow is None


def test_image_save_graph_uses_active_inputs_before_materialization() -> None:
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
    data.invocation = AddInvocation(id="resolved-add", a=17, b=1)
    images = ImagesInterface(services, data, MagicMock())
    source_json = graph.model_dump_json()

    images.save(MagicMock())

    saved_kwargs = services.images.create.call_args.kwargs
    assert saved_kwargs["graph"] == source_json
    saved_graph_json = saved_kwargs["resolved_inputs"]
    saved_graph = Graph.model_validate_json(saved_graph_json)
    saved_add_node = saved_graph.nodes["resolved-add"]
    assert isinstance(saved_add_node, AddInvocation)
    assert (saved_add_node.a, saved_add_node.b) == (17, 1)
    assert set(saved_graph.nodes) == {"resolved-add"}
    assert saved_graph.edges == []
    assert graph.model_dump_json() == source_json
    assert not session.execution_graph.nodes


@pytest.mark.parametrize("cached_workflow", [False, True])
def test_image_save_graph_png_round_trip(tmp_path: Path, cached_workflow: bool) -> None:
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
            "notes": "Keep the authoring workflow separate from the resolved input record.",
            "exposedFields": [],
            "meta": {"category": "user", "version": "3.0.0"},
            "nodes": [
                {
                    "id": "random",
                    "type": "invocation",
                    "position": {"x": 0, "y": 0},
                    "data": {"id": "random", "type": "rand_int", "inputs": {"low": {"value": 17}}},
                },
                {
                    "id": "add",
                    "type": "invocation",
                    "position": {"x": 100, "y": 0},
                    "data": {"id": "add", "type": "add", "inputs": {"a": {"value": 0}, "b": {"value": 1}}},
                },
            ],
            "edges": [
                {
                    "id": "random-to-add",
                    "source": "random",
                    "sourceHandle": "value",
                    "target": "add",
                    "targetHandle": "a",
                    "type": "default",
                }
            ],
        }
    )
    source_json = session.graph.model_dump_json()
    workflow_model_json = workflow.model_dump_json()
    # A recovered queue item's original workflow JSON can have different whitespace from model_dump_json().
    workflow_json = (
        json.dumps(json.loads(workflow_model_json), indent=2) + "\n" if cached_workflow else workflow_model_json
    )
    services = MagicMock()
    services.configuration.multiuser = False
    services.configuration.pil_compress_level = 6
    storage = DiskImageFileStorage(tmp_path / "outputs")
    invoker = MagicMock()
    invoker.services = services
    storage.start(invoker)

    # PNG persistence and extraction are real; the owning image-service path is covered in its SQLite suite.
    def create_image(
        image: Image.Image,
        metadata: str | None,
        workflow: str | None,
        graph: str | None,
        resolved_inputs: str | None,
        **kwargs: object,
    ) -> None:
        storage.save(
            image=image,
            image_name="runtime.png",
            metadata=metadata,
            workflow=workflow,
            graph=graph,
            resolved_inputs=resolved_inputs,
        )

    services.images.create.side_effect = create_image
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = workflow
    data.queue_item.root_item_id = None
    data.queue_item._workflow_json_loaded = cached_workflow
    data.queue_item._workflow_json_snapshot = workflow_json if cached_workflow else None
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
        saved_graph = Graph.model_validate_json(reopened.info["invokeai_resolved_inputs"])

    assert extracted.invokeai_graph == source_json
    saved_node_id = services.images.create.call_args.kwargs["node_id"]
    assert saved_node_id == runtime_add_node.id
    saved_add_node = saved_graph.nodes[saved_node_id]
    assert isinstance(saved_add_node, AddInvocation)
    assert (saved_add_node.id, saved_add_node.a, saved_add_node.b) == (runtime_add_node.id, 17, 1)
    assert set(saved_graph.nodes) == {random_node.id, runtime_add_node.id}
    assert sorted(
        (edge.source.node_id, edge.source.field, edge.destination.node_id, edge.destination.field)
        for edge in saved_graph.edges
    ) == sorted(
        [
            (random_node.id, "value", runtime_add_node.id, "a"),
        ]
    )
    assert extracted.invokeai_workflow == workflow_json
    assert data.queue_item.workflow.model_dump_json() == workflow_model_json
    assert session.graph.model_dump_json() == source_json


def test_image_save_graph_retains_resolved_noise_seed_and_math_ancestors() -> None:
    graph = Graph(
        nodes={
            "random": RandomIntInvocation(id="random", low=17, high=18),
            "add": AddInvocation(id="add", b=1),
            "noise": NoiseInvocation(id="noise", width=8, height=8),
            "image": BlankImageInvocation(id="image", width=1, height=2),
            "future": AddInvocation(id="future", b=2),
        },
        edges=[
            Edge(
                source=EdgeConnection(node_id=source, field=output),
                destination=EdgeConnection(node_id=destination, field=input_),
            )
            for source, output, destination, input_ in [
                ("random", "value", "add", "a"),
                ("add", "value", "noise", "seed"),
                ("noise", "width", "image", "width"),
                ("image", "width", "future", "a"),
            ]
        ],
    )
    source_json = graph.model_dump_json()
    session = GraphExecutionState(graph=graph)
    images, services = _make_interface(allowed=True)
    services.configuration.multiuser = False
    data = images._data
    data.queue_item.session = session
    data.queue_item.root_item_id = None
    data.queue_item._workflow_json_loaded = False
    data.queue_item._workflow_json_snapshot = None
    services.images.create.return_value = SimpleNamespace(image_name="resolved.png", width=8, height=2)
    for _ in range(5):
        node = session.next()
        assert node is not None
        data.invocation = node
        if isinstance(node, BlankImageInvocation):
            execution_json = session.execution_graph.model_dump_json()
            node.invoke(MagicMock(images=images))
            assert session.execution_graph.model_dump_json() == execution_json
            break
        if isinstance(node, NoiseInvocation):
            assert node.seed == 18
            # Substitute only the tensor-generation boundary. The downstream leaf has no seed input.
            result = NoiseOutput(noise=LatentsField(latents_name="opaque-noise"), width=8, height=8)
        else:
            result = node.invoke(MagicMock())
        session.complete(node.id, result)
    else:
        pytest.fail("Image invocation was not scheduled")

    saved_kwargs = services.images.create.call_args.kwargs
    assert saved_kwargs["graph"] == source_json
    saved_graph = Graph.model_validate_json(saved_kwargs["resolved_inputs"])
    ids = {name: next(iter(session.source_prepared_mapping[name])) for name in ("random", "add", "noise", "image")}
    assert set(saved_graph.nodes) == set(ids.values())
    assert saved_kwargs["node_id"] == ids["image"]
    assert saved_graph.nodes[ids["add"]].a == 17
    assert saved_graph.nodes[ids["noise"]].seed == 18
    assert saved_graph.nodes[ids["image"]].width == 8
    assert {
        (e.source.node_id, e.source.field, e.destination.node_id, e.destination.field) for e in saved_graph.edges
    } == {
        (ids["random"], "value", ids["add"], "a"),
        (ids["add"], "value", ids["noise"], "seed"),
        (ids["noise"], "width", ids["image"], "width"),
    }
    assert graph.nodes["add"].a == graph.nodes["noise"].seed == 0
    assert graph.model_dump_json() == source_json


@pytest.mark.parametrize(
    ("boundary", "output", "resolved_value"),
    [
        (IterateInvocation(id="boundary", collection=[17, 18]), "item", 17),
        (CollectInvocation(id="boundary", collection=[17, 18]), "collection", [17, 18]),
        (IfInvocation(id="boundary", condition=True, true_input=17, false_input=90), "value", 17),
        (ForInvocation(id="boundary", collection=[17, 18], index=0), "item", 17),
        (ForReturnInvocation(id="boundary", output=17), "output", 17),
        (CallSavedWorkflowInvocation(id="boundary", workflow_id="saved"), "values", {"seed": 17}),
        (CollectionConcatInvocation(id="boundary", first=[17, 18]), "collection", [17, 18]),
        (AnyTypeTestInvocation(id="boundary", value=[17, 18]), "value", [17, 18]),
    ],
)
def test_image_graph_projection_keeps_resolved_inputs_across_boundaries(
    boundary: BaseInvocation, output: str, resolved_value: object
) -> None:
    graph = Graph(
        nodes={"boundary": boundary, "active": AnyTypeTestInvocation(id="active")},
        edges=[
            Edge(
                source=EdgeConnection(node_id="boundary", field=output),
                destination=EdgeConnection(node_id="active", field="value"),
            )
        ],
    )
    before = graph.model_dump_json()
    active = AnyTypeTestInvocation(id="active", value=resolved_value)

    record = graph._get_invocation_input_graph(active)

    assert set(record.nodes) == {active.id}
    assert record.nodes[active.id].value == resolved_value
    assert record.edges == []
    assert graph.model_dump_json() == before


def test_image_save_graph_nested_iterate_keeps_one_resolved_record_per_image(tmp_path: Path) -> None:
    measurements = []
    saved_runs = []
    for outer_size, inner_size in [(2, 3), (4, 6)]:
        values = [[10 + i * inner_size + j for j in range(inner_size)] for i in range(outer_size)]
        graph = Graph(
            nodes={
                "source": CollectionConcatInvocation(id="source", first=values),
                "outer": IterateInvocation(id="outer"),
                "prepare": CollectionConcatInvocation(id="prepare"),
                "inner": IterateInvocation(id="inner"),
                "image": BlankImageInvocation(id="image", width=1, height=2),
            },
            edges=[
                Edge(
                    source=EdgeConnection(node_id=source, field=output),
                    destination=EdgeConnection(node_id=destination, field=input_),
                )
                for source, output, destination, input_ in [
                    ("source", "collection", "outer", "collection"),
                    ("outer", "item", "prepare", "first"),
                    ("prepare", "collection", "inner", "collection"),
                    ("inner", "item", "image", "width"),
                ]
            ],
        )
        source_json = graph.model_dump_json()
        session = GraphExecutionState(graph=graph)
        images, services = _make_interface(allowed=True)
        services.configuration.multiuser = False
        services.configuration.pil_compress_level = 6
        data = images._data
        data.queue_item.session = session
        data.queue_item.root_item_id = None
        data.queue_item._workflow_json_loaded = False
        data.queue_item._workflow_json_snapshot = None
        storage = DiskImageFileStorage(tmp_path / f"{outer_size}x{inner_size}")
        storage.start(SimpleNamespace(services=services))
        saved: list[tuple[str, str, str, str, int]] = []

        def create_image(
            image: Image.Image,
            graph: str,
            resolved_inputs: str,
            *,
            _storage: DiskImageFileStorage = storage,
            _saved: list[tuple[str, str, str, str, int]] = saved,
            **kwargs: object,
        ) -> SimpleNamespace:
            image_name = f"{len(_saved)}.png"
            _storage.save(image=image, image_name=image_name, graph=graph, resolved_inputs=resolved_inputs)
            node_id = kwargs["node_id"]
            assert isinstance(node_id, str)
            _saved.append((image_name, graph, resolved_inputs, node_id, image.width))
            return SimpleNamespace(image_name=image_name, width=image.width, height=image.height)

        services.images.create.side_effect = create_image
        context = MagicMock(images=images)
        # Bound scheduler calls to catch a stalled loop without asserting execution time.
        for _ in range(2 + 2 * outer_size + 2 * outer_size * inner_size):
            node = session.next()
            if node is None:
                break
            data.invocation = node
            execution_json = session.execution_graph.model_dump_json()
            result = node.invoke(context)
            assert session.execution_graph.model_dump_json() == execution_json
            session.complete(node.id, result)
        else:
            pytest.fail("Nested Iterate graph did not finish")

        assert session.is_complete()
        assert [width for _, _, _, _, width in saved] == [value for group in values for value in group]
        assert graph.model_dump_json() == source_json
        reader = DiskImageFileStorage(storage.image_root)
        for image_name, graph_json, resolved_json, _, _ in saved:
            assert reader.get_graph(image_name) == graph_json
            with reader.get(image_name) as reopened:
                assert reopened.info["invokeai_resolved_inputs"] == resolved_json
            assert graph_json == source_json
        payloads = [json.loads(resolved_json) for _, _, resolved_json, _, _ in saved]
        source_byte_counts = [len(graph_json.encode("utf-8")) for _, graph_json, _, _, _ in saved]
        resolved_byte_counts = [len(resolved_json.encode("utf-8")) for _, _, resolved_json, _, _ in saved]
        byte_counts = [
            source_bytes + resolved_bytes
            for source_bytes, resolved_bytes in zip(source_byte_counts, resolved_byte_counts, strict=True)
        ]
        measurements.append(
            {
                "shape": [outer_size, inner_size],
                "images": len(saved),
                "per_image_bytes": byte_counts,
                "total_bytes": sum(byte_counts),
                "source_bytes": len(source_json.encode("utf-8")),
                "per_image_resolved_bytes": resolved_byte_counts,
                "total_resolved_bytes": sum(resolved_byte_counts),
                "resolved_node_counts": [len(payload["nodes"]) for payload in payloads],
                "source_node_counts": [len(json.loads(graph_json)["nodes"]) for _, graph_json, _, _, _ in saved],
            }
        )
        saved_runs.append(saved)

    # Count both persisted channels: the complete source graph and the compact input record.
    print("NESTED_IMAGE_GRAPH_MEASUREMENTS=" + json.dumps(measurements))
    for saved in saved_runs:
        for _, _, resolved_json, node_id, width in saved:
            saved_graph = Graph.model_validate_json(resolved_json)
            assert set(saved_graph.nodes) == {node_id}
            assert saved_graph.edges == []
            saved_node = saved_graph.nodes[node_id]
            assert isinstance(saved_node, BlankImageInvocation)
            assert (saved_node.id, saved_node.width, saved_node.height) == (node_id, width, 2)
            assert node_id != "image"
            assert len(node_id) == len(saved_graph.id) == 36
    # Fixed-width resolved inputs cost no extra bytes as runtime copies grow. The authoring collection grows.
    assert len(set(measurements[0]["per_image_resolved_bytes"] + measurements[1]["per_image_resolved_bytes"])) == 1
    assert measurements[1]["total_resolved_bytes"] == 4 * measurements[0]["total_resolved_bytes"]
    for measurement in measurements:
        assert measurement["total_bytes"] == (
            measurement["images"] * measurement["source_bytes"] + measurement["total_resolved_bytes"]
        )
