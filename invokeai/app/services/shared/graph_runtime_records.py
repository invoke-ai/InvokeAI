import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from invokeai.app.services.shared.graph_models import PreparedExecState

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _ApplyTransaction:
    """Small undo journal for scheduler mutations made by GraphExecutionState.apply()."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []
        self._recorded: set[tuple[Any, ...]] = set()

    def record_once(self, key: tuple[Any, ...], undo: Callable[[], None]) -> None:
        if key in self._recorded:
            return
        self._recorded.add(key)
        self._undo.append(undo)

    def is_recorded(self, key: tuple[Any, ...]) -> bool:
        return key in self._recorded

    def record(self, undo: Callable[[], None]) -> None:
        self._undo.append(undo)

    def rollback(self) -> None:
        for undo in reversed(self._undo):
            undo()


@dataclass
class _PreparedExecNodeMetadata:
    """Cached metadata for a materialized execution node."""

    source_node_id: str
    iteration_path: Optional[tuple[int, ...]] = None
    state: PreparedExecState = "pending"


def _new_prepared_exec_node_metadata(source_node_id: str) -> _PreparedExecNodeMetadata:
    graph_facade = sys.modules.get("invokeai.app.services.shared.graph")
    metadata_type = getattr(graph_facade, "_PreparedExecNodeMetadata", _PreparedExecNodeMetadata)
    return metadata_type(source_node_id=source_node_id)


class _PreparedExecRegistry:
    """Tracks prepared execution nodes and their relationship to source graph nodes."""

    def __init__(
        self,
        prepared_source_mapping: dict[str, str],
        source_prepared_mapping: dict[str, set[str]],
        prepared_iteration_paths: dict[str, tuple[int, ...]],
        metadata: dict[str, _PreparedExecNodeMetadata],
        on_iteration_path_change: Callable[[str], None] | None = None,
        state: Optional["GraphExecutionState"] = None,
    ) -> None:
        self._prepared_source_mapping = prepared_source_mapping
        self._source_prepared_mapping = source_prepared_mapping
        self._prepared_iteration_paths = prepared_iteration_paths
        self._metadata = metadata
        self._on_iteration_path_change = on_iteration_path_change
        self._state = state
        self._prepared_ids_by_source_and_path: dict[str, dict[tuple[int, ...], set[str]]] = {}

    def _invalidate_source_path_index(self, source_node_id: str) -> None:
        previous = self._prepared_ids_by_source_and_path.pop(source_node_id, None)
        if self._state is not None and self._state._apply_transaction is not None:
            self._state._tx_record_once(
                ("prepared_path_index", id(self), source_node_id),
                lambda: (
                    self._prepared_ids_by_source_and_path.__setitem__(source_node_id, previous)
                    if previous is not None
                    else self._prepared_ids_by_source_and_path.pop(source_node_id, None)
                ),
            )

    def _set_mapping(self, mapping: dict[Any, Any], key: Any, value: Any) -> None:
        if self._state is not None:
            self._state._tx_set_mapping(mapping, key, value)
        else:
            mapping[key] = value

    def _pop_mapping(self, mapping: dict[Any, Any], key: Any) -> None:
        if self._state is not None:
            self._state._tx_pop_mapping(mapping, key)
        else:
            mapping.pop(key, None)

    def _set_attr(self, obj: Any, name: str, value: Any) -> None:
        if self._state is not None:
            self._state._tx_set_attr(obj, name, value)
        else:
            setattr(obj, name, value)

    def register(self, exec_node_id: str, source_node_id: str) -> None:
        self._invalidate_source_path_index(source_node_id)
        self._set_mapping(self._prepared_source_mapping, exec_node_id, source_node_id)
        self._pop_mapping(self._prepared_iteration_paths, exec_node_id)
        self._set_mapping(self._metadata, exec_node_id, _new_prepared_exec_node_metadata(source_node_id))
        if source_node_id not in self._source_prepared_mapping:
            self._set_mapping(self._source_prepared_mapping, source_node_id, set())
        if self._state is not None:
            self._state._tx_add_set(self._source_prepared_mapping[source_node_id], exec_node_id)
        else:
            self._source_prepared_mapping[source_node_id].add(exec_node_id)

    def get_metadata(self, exec_node_id: str) -> _PreparedExecNodeMetadata:
        metadata = self._metadata.get(exec_node_id)
        if metadata is None:
            metadata = _new_prepared_exec_node_metadata(self._prepared_source_mapping[exec_node_id])
            self._set_mapping(self._metadata, exec_node_id, metadata)
        return metadata

    def get_source_node_id(self, exec_node_id: str) -> str:
        metadata = self._metadata.get(exec_node_id)
        if metadata is not None:
            return metadata.source_node_id
        return self._prepared_source_mapping[exec_node_id]

    def get_prepared_ids(self, source_node_id: str) -> set[str]:
        return self._source_prepared_mapping.get(source_node_id, set())

    def get_prepared_id_at_path(self, source_node_id: str, iteration_path: tuple[int, ...]) -> str | None:
        """Return the only prepared execution for a source/path without rescanning siblings."""
        paths = self._prepared_ids_by_source_and_path.get(source_node_id)
        if paths is None:
            paths = {}
            for exec_node_id in self.get_prepared_ids(source_node_id):
                path = self.get_iteration_path(exec_node_id)
                if path is not None:
                    paths.setdefault(path, set()).add(exec_node_id)
            self._prepared_ids_by_source_and_path[source_node_id] = paths
            if self._state is not None and self._state._apply_transaction is not None:
                self._state._tx_record_once(
                    ("prepared_path_index", id(self), source_node_id),
                    lambda: self._prepared_ids_by_source_and_path.pop(source_node_id, None),
                )
        matches = paths.get(iteration_path, set())
        if len(matches) > 1:
            raise RuntimeError(f"Multiple prepared nested nodes exist for {source_node_id} at {iteration_path}")
        return next(iter(matches), None)

    def set_state(self, exec_node_id: str, state: PreparedExecState) -> None:
        self._set_attr(self.get_metadata(exec_node_id), "state", state)

    def get_iteration_path(self, exec_node_id: str) -> Optional[tuple[int, ...]]:
        metadata = self._metadata.get(exec_node_id)
        if metadata is not None and metadata.iteration_path is not None:
            return metadata.iteration_path
        iteration_path = self._prepared_iteration_paths.get(exec_node_id)
        if iteration_path is not None:
            self._set_attr(self.get_metadata(exec_node_id), "iteration_path", iteration_path)
        return iteration_path

    def set_iteration_path(self, exec_node_id: str, iteration_path: tuple[int, ...]) -> None:
        self._invalidate_source_path_index(self.get_source_node_id(exec_node_id))
        self._set_mapping(self._prepared_iteration_paths, exec_node_id, iteration_path)
        self._set_attr(self.get_metadata(exec_node_id), "iteration_path", iteration_path)
        if self._on_iteration_path_change is not None:
            self._on_iteration_path_change(exec_node_id)
