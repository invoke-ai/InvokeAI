import subprocess
import sys
import textwrap

import pytest

from invokeai.app.services.shared import graph as graph_facade
from invokeai.app.services.shared import graph_runtime_records


@pytest.mark.platform_independent
def test_runtime_records_import_without_graph_cycle() -> None:
    script = textwrap.dedent(
        """
        import sys

        from invokeai.app.services.shared import graph_runtime_records

        assert graph_runtime_records._PreparedExecRegistry
        assert "invokeai.app.services.shared.graph" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_graph_facade_reexports_runtime_record_aliases() -> None:
    assert graph_facade._ApplyTransaction is graph_runtime_records._ApplyTransaction
    assert graph_facade._PreparedExecNodeMetadata is graph_runtime_records._PreparedExecNodeMetadata
    assert graph_facade._PreparedExecRegistry is graph_runtime_records._PreparedExecRegistry


def test_graph_facade_metadata_alias_controls_registry_metadata(monkeypatch) -> None:
    class PatchedMetadata(graph_facade._PreparedExecNodeMetadata):
        pass

    monkeypatch.setattr(graph_facade, "_PreparedExecNodeMetadata", PatchedMetadata)
    prepared_source_mapping = {}
    source_prepared_mapping = {}
    registry = graph_runtime_records._PreparedExecRegistry(
        prepared_source_mapping,
        source_prepared_mapping,
        {},
        {},
    )

    registry.register("registered-exec-node", "source-node")
    assert type(registry.get_metadata("registered-exec-node")) is PatchedMetadata

    prepared_source_mapping["lazy-exec-node"] = "source-node"
    assert type(registry.get_metadata("lazy-exec-node")) is PatchedMetadata


def test_standalone_registry_lifecycle_and_lazy_metadata() -> None:
    prepared_source_mapping = {"exec-node": "source-node"}
    source_prepared_mapping = {"source-node": {"exec-node"}}
    prepared_iteration_paths = {"exec-node": (2, 3)}
    metadata = {}
    changed: list[str] = []
    registry = graph_runtime_records._PreparedExecRegistry(
        prepared_source_mapping,
        source_prepared_mapping,
        prepared_iteration_paths,
        metadata,
        on_iteration_path_change=changed.append,
    )

    assert registry.get_source_node_id("exec-node") == "source-node"
    assert "exec-node" not in metadata
    assert registry.get_iteration_path("exec-node") == (2, 3)
    assert metadata["exec-node"].iteration_path == (2, 3)

    registry.set_state("exec-node", "ready")
    registry.set_iteration_path("exec-node", (4,))
    assert metadata["exec-node"].state == "ready"
    assert prepared_iteration_paths["exec-node"] == (4,)
    assert changed == ["exec-node"]

    registry.register("new-exec-node", "source-node")
    assert registry.get_prepared_ids("source-node") == {"exec-node", "new-exec-node"}
    assert registry.get_metadata("new-exec-node").source_node_id == "source-node"
    assert "new-exec-node" not in prepared_iteration_paths


def test_prepared_path_overlap_index_tracks_registration_and_path_changes() -> None:
    prepared_source_mapping = {"first": "source"}
    source_prepared_mapping = {"source": {"first"}}
    prepared_iteration_paths = {"first": (2, 3)}
    registry = graph_runtime_records._PreparedExecRegistry(
        prepared_source_mapping,
        source_prepared_mapping,
        prepared_iteration_paths,
        {},
    )

    assert set(registry.get_prepared_ids_with_overlapping_path("source", (2,))) == {"first"}
    assert set(registry.get_prepared_ids_with_overlapping_path("source", (2, 3, 4))) == {"first"}
    assert set(registry.get_prepared_ids_with_overlapping_path("source", (1,))) == set()

    registry.register("second", "source")
    registry.set_iteration_path("second", (4,))
    assert set(registry.get_prepared_ids_with_overlapping_path("source", (4, 5))) == {"second"}

    registry.set_iteration_path("first", (5,))
    assert set(registry.get_prepared_ids_with_overlapping_path("source", (2,))) == set()
    assert set(registry.get_prepared_ids_with_overlapping_path("source", (5, 6))) == {"first"}


def test_prepared_path_overlap_index_rolls_back_with_registration_and_path() -> None:
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())
    registry = state._prepared_registry()

    def add_then_fail() -> None:
        registry.register("exec-node", "source-node")
        registry.set_iteration_path("exec-node", (2, 3))
        assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (2,))) == {"exec-node"}
        raise RuntimeError("rollback test")

    with pytest.raises(RuntimeError, match="rollback test"):
        state._run_materialization_transaction(add_then_fail)

    assert registry.get_prepared_ids("source-node") == set()
    assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (2,))) == set()


def test_prepared_path_overlap_index_incremental_updates_roll_back() -> None:
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())
    registry = state._prepared_registry()
    registry.register("existing", "source-node")
    registry.set_iteration_path("existing", (1, 2))
    assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (1,))) == {"existing"}

    def mutate_then_fail() -> None:
        registry.register("new", "source-node")
        registry.set_iteration_path("new", (2,))
        registry.set_iteration_path("existing", (3, 4))
        assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (3,))) == {"existing"}
        raise RuntimeError("rollback cached index")

    with pytest.raises(RuntimeError, match="rollback cached index"):
        state._run_materialization_transaction(mutate_then_fail)

    assert registry.get_prepared_ids("source-node") == {"existing"}
    assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (1,))) == {"existing"}
    assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (2,))) == set()
    assert set(registry.get_prepared_ids_with_overlapping_path("source-node", (3,))) == set()


def test_apply_transaction_record_once_rolls_back_in_reverse_order() -> None:
    transaction = graph_runtime_records._ApplyTransaction()
    undone: list[str] = []

    transaction.record_once(("first",), lambda: undone.append("first"))
    transaction.record_once(("first",), lambda: undone.append("duplicate"))
    transaction.record(lambda: undone.append("second"))

    transaction.rollback()

    assert undone == ["second", "first"]
