import networkx as nx

from abc import ABC, abstractmethod
from itertools import product
from pydantic import BaseModel, Field
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event

from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.graph import Graph, GraphExecutionState
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.batch_manager_storage import (
    BatchProcessStorageBase,
    BatchSessionNotFoundException,
    Batch,
    BatchProcess,
    BatchSession,
    BatchSessionChanges,
)


class BatchProcessResponse(BaseModel):
    batch_id: str = Field(description="ID for the batch")
    session_ids: list[str] = Field(description="List of session IDs created for this batch")


class BatchManagerBase(ABC):
    @abstractmethod
    def start(self, invoker: Invoker) -> None:
        """Starts the BatchManager service"""
        pass

    @abstractmethod
    def create_batch_process(self, batch: Batch, graph: Graph) -> BatchProcessResponse:
        """Creates a batch process"""
        pass

    @abstractmethod
    def run_batch_process(self, batch_id: str) -> None:
        """Runs a batch process"""
        pass

    @abstractmethod
    def cancel_batch_process(self, batch_process_id: str) -> None:
        """Cancels a batch process"""
        pass

    @abstractmethod
    def get_batch(self, batch_id: str) -> BatchProcessResponse:
        """Gets a batch process"""
        pass

    @abstractmethod
    def get_batch_processes(self) -> list[BatchProcessResponse]:
        """Gets all batch processes"""
        pass

    @abstractmethod
    def get_incomplete_batch_processes(self) -> list[BatchProcessResponse]:
        """Gets all incomplete batch processes"""
        pass

    @abstractmethod
    def get_sessions(self, batch_id: str) -> list[BatchSession]:
        """Gets the sessions associated with a batch"""
        pass


class BatchManager(BatchManagerBase):
    """Responsible for managing currently running and scheduled batch jobs"""

    __invoker: Invoker
    __batch_process_storage: BatchProcessStorageBase

    def __init__(self, batch_process_storage: BatchProcessStorageBase) -> None:
        super().__init__()
        self.__batch_process_storage = batch_process_storage

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        local_handler.register(event_name=EventServiceBase.session_event, _func=self.on_event)

    async def on_event(self, event: Event):
        event_name = event[1]["event"]

        match event_name:
            case "graph_execution_state_complete":
                await self._process(event, False)
            case "invocation_error":
                await self._process(event, True)

        return event

    async def _process(self, event: Event, err: bool) -> None:
        data = event[1]["data"]
        try:
            batch_session = self.__batch_process_storage.get_session_by_session_id(data["graph_execution_state_id"])
        except BatchSessionNotFoundException:
            return None
        changes = BatchSessionChanges(state="error" if err else "completed")
        batch_session = self.__batch_process_storage.update_session_state(
            batch_session.batch_id,
            batch_session.session_id,
            changes,
        )
        batch_process = self.__batch_process_storage.get(batch_session.batch_id)
        if not batch_process.canceled:
            self.run_batch_process(batch_process.batch_id)

    def _create_batch_session(self, batch_process: BatchProcess, batch_indices: tuple[int]) -> GraphExecutionState:
        graph = batch_process.graph.copy(deep=True)
        batch = batch_process.batch
        g = graph.nx_graph_flat()
        sorted_nodes = nx.topological_sort(g)
        for npath in sorted_nodes:
            node = graph.get_node(npath)
            for index, bdl in enumerate(batch.data):
                relevant_bd = [bd for bd in bdl if bd.node_id in node.id]
                if not relevant_bd:
                    continue
                for bd in relevant_bd:
                    batch_index = batch_indices[index]
                    datum = bd.items[batch_index]
                    key = bd.field_name
                    node.__dict__[key] = datum
                graph.update_node(npath, node)

        return GraphExecutionState(graph=graph)

    def run_batch_process(self, batch_id: str) -> None:
        self.__batch_process_storage.start(batch_id)
        try:
            next_session = self.__batch_process_storage.get_next_session(batch_id)
        except BatchSessionNotFoundException:
            return
        batch_process = self.__batch_process_storage.get(batch_id)
        ges = self._create_batch_session(batch_process=batch_process, batch_indices=tuple(next_session.batch_index))
        ges.id = next_session.session_id
        self.__invoker.services.graph_execution_manager.set(ges)
        self.__batch_process_storage.update_session_state(
            batch_id=next_session.batch_id,
            session_id=next_session.session_id,
            changes=BatchSessionChanges(state="in_progress"),
        )
        self.__invoker.invoke(ges, invoke_all=True)

    def create_batch_process(self, batch: Batch, graph: Graph) -> BatchProcessResponse:
        batch_process = BatchProcess(
            batch=batch,
            graph=graph,
        )
        batch_process = self.__batch_process_storage.save(batch_process)
        sessions = self._create_sessions(batch_process)
        return BatchProcessResponse(
            batch_id=batch_process.batch_id,
            session_ids=[session.session_id for session in sessions],
        )

    def _create_sessions(self, batch_process: BatchProcess) -> list[BatchSession]:
        batch_indices = list()
        sessions_to_create: list[BatchSession] = list()
        for batchdata in batch_process.batch.data:
            batch_indices.append(list(range(len(batchdata[0].items))))
        all_batch_indices = product(*batch_indices)
        for bi in all_batch_indices:
            for _ in range(batch_process.batch.runs):
                sessions_to_create.append(BatchSession(batch_id=batch_process.batch_id, batch_index=list(bi)))
            if not sessions_to_create:
                sessions_to_create.append(BatchSession(batch_id=batch_process.batch_id, batch_index=list(bi)))
        created_sessions = self.__batch_process_storage.create_sessions(sessions_to_create)
        return created_sessions

    def get_sessions(self, batch_id: str) -> list[BatchSession]:
        return self.__batch_process_storage.get_sessions_by_batch_id(batch_id)

    def get_batch(self, batch_id: str) -> BatchProcess:
        return self.__batch_process_storage.get(batch_id)

    def get_batch_processes(self) -> list[BatchProcessResponse]:
        bps = self.__batch_process_storage.get_all()
        return self._get_batch_process_responses(bps)

    def get_incomplete_batch_processes(self) -> list[BatchProcessResponse]:
        bps = self.__batch_process_storage.get_incomplete()
        return self._get_batch_process_responses(bps)

    def cancel_batch_process(self, batch_process_id: str) -> None:
        self.__batch_process_storage.cancel(batch_process_id)

    def _get_batch_process_responses(self, batch_processes: list[BatchProcess]) -> list[BatchProcessResponse]:
        sessions = list()
        res: list[BatchProcessResponse] = list()
        for bp in batch_processes:
            sessions = self.__batch_process_storage.get_sessions_by_batch_id(bp.batch_id)
            res.append(
                BatchProcessResponse(
                    batch_id=bp.batch_id,
                    session_ids=[session.session_id for session in sessions],
                )
            )
        return res
