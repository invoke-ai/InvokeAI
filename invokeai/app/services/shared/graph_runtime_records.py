import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from invokeai.app.services.shared.graph_models import PreparedExecState

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _ApplyTransaction:
    """Small undo journal for scheduler and materialization state mutations."""

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
    iteration_axes: Optional[tuple[str, ...]] = None
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
        prepared_iteration_axes: Optional[dict[str, tuple[str, ...]]] = None,
    ) -> None:
        self._prepared_source_mapping = prepared_source_mapping
        self._source_prepared_mapping = source_prepared_mapping
        self._prepared_iteration_paths = prepared_iteration_paths
        self._prepared_iteration_axes = prepared_iteration_axes if prepared_iteration_axes is not None else {}
        self._metadata = metadata
        self._on_iteration_path_change = on_iteration_path_change
        self._state = state
        self._prepared_ids_by_source_and_path: dict[
            str, tuple[dict[tuple[int, ...], set[str]], dict[tuple[int, ...], set[str]]]
        ] = {}

    def _record_index_undo(self, undo: Callable[[], None]) -> None:
        if self._state is not None:
            self._state._tx_record(undo)

    @staticmethod
    def _change_path_index(
        index: tuple[dict[tuple[int, ...], set[str]], dict[tuple[int, ...], set[str]]],
        exec_node_id: str,
        iteration_path: tuple[int, ...],
        add: bool,
    ) -> None:
        ids_by_path, ids_by_prefix = index
        if add:
            ids_by_path.setdefault(iteration_path, set()).add(exec_node_id)
            for prefix_length in range(len(iteration_path) + 1):
                ids_by_prefix.setdefault(iteration_path[:prefix_length], set()).add(exec_node_id)
            return
        path_ids = ids_by_path.get(iteration_path)
        if path_ids is not None:
            path_ids.discard(exec_node_id)
            if not path_ids:
                del ids_by_path[iteration_path]
        for prefix_length in range(len(iteration_path) + 1):
            prefix = iteration_path[:prefix_length]
            prefix_ids = ids_by_prefix.get(prefix)
            if prefix_ids is not None:
                prefix_ids.discard(exec_node_id)
                if not prefix_ids:
                    del ids_by_prefix[prefix]

    def _get_source_path_index(
        self, source_node_id: str
    ) -> tuple[dict[tuple[int, ...], set[str]], dict[tuple[int, ...], set[str]]]:
        index = self._prepared_ids_by_source_and_path.get(source_node_id)
        if index is None:
            ids_by_path: dict[tuple[int, ...], set[str]] = {}
            ids_by_prefix: dict[tuple[int, ...], set[str]] = {}
            index = (ids_by_path, ids_by_prefix)
            self._prepared_ids_by_source_and_path[source_node_id] = index
            if self._state is not None:
                self._state._tx_record_once(
                    ("prepared_path_index", id(self), source_node_id),
                    lambda: self._prepared_ids_by_source_and_path.pop(source_node_id, None),
                )
            for exec_node_id in self.get_prepared_ids(source_node_id):
                path = self.get_iteration_path(exec_node_id)
                if path is not None:
                    self._change_path_index(index, exec_node_id, path, True)
        return index

    def _update_source_path_index(
        self, source_node_id: str, exec_node_id: str, iteration_path: tuple[int, ...], add: bool
    ) -> None:
        index = self._prepared_ids_by_source_and_path.get(source_node_id)
        if index is None:
            return
        self._change_path_index(index, exec_node_id, iteration_path, add)
        self._record_index_undo(lambda: self._change_path_index(index, exec_node_id, iteration_path, not add))

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
        previous_source = self._prepared_source_mapping.get(exec_node_id)
        previous_path = self.get_iteration_path(exec_node_id)
        if previous_source is not None and previous_path is not None:
            self._update_source_path_index(previous_source, exec_node_id, previous_path, False)
        self._set_mapping(self._prepared_source_mapping, exec_node_id, source_node_id)
        self._pop_mapping(self._prepared_iteration_paths, exec_node_id)
        self._pop_mapping(self._prepared_iteration_axes, exec_node_id)
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
        paths, _ = self._get_source_path_index(source_node_id)
        matches = paths.get(iteration_path, set())
        if len(matches) > 1:
            raise RuntimeError(f"Multiple prepared nested nodes exist for {source_node_id} at {iteration_path}")
        return next(iter(matches), None)

    def get_prepared_ids_with_overlapping_path(
        self, source_node_id: str, iteration_path: tuple[int, ...]
    ) -> Iterable[str]:
        """Yield prepared IDs whose paths are ancestors or descendants of ``iteration_path``."""
        ids_by_path, ids_by_prefix = self._get_source_path_index(source_node_id)
        seen: set[str] = set()
        for prefix_length in range(len(iteration_path) + 1):
            for exec_node_id in ids_by_path.get(iteration_path[:prefix_length], ()):
                if exec_node_id not in seen:
                    seen.add(exec_node_id)
                    yield exec_node_id
        for exec_node_id in ids_by_prefix.get(iteration_path, ()):
            if exec_node_id not in seen:
                yield exec_node_id

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
        source_node_id = self.get_source_node_id(exec_node_id)
        previous_path = self.get_iteration_path(exec_node_id)
        if previous_path is not None:
            self._update_source_path_index(source_node_id, exec_node_id, previous_path, False)
        self._set_mapping(self._prepared_iteration_paths, exec_node_id, iteration_path)
        self._set_attr(self.get_metadata(exec_node_id), "iteration_path", iteration_path)
        self._update_source_path_index(source_node_id, exec_node_id, iteration_path, True)
        if self._on_iteration_path_change is not None:
            self._on_iteration_path_change(exec_node_id)

    def get_iteration_axes(self, exec_node_id: str) -> Optional[tuple[str, ...]]:
        metadata = self._metadata.get(exec_node_id)
        if metadata is not None and metadata.iteration_axes is not None:
            return metadata.iteration_axes
        iteration_axes = self._prepared_iteration_axes.get(exec_node_id)
        if iteration_axes is not None:
            self._set_attr(self.get_metadata(exec_node_id), "iteration_axes", iteration_axes)
        return iteration_axes

    def set_iteration_axes(self, exec_node_id: str, iteration_axes: tuple[str, ...]) -> None:
        self._set_mapping(self._prepared_iteration_axes, exec_node_id, iteration_axes)
        self._set_attr(self.get_metadata(exec_node_id), "iteration_axes", iteration_axes)
