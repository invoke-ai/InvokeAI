import networkx as nx
import copy

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
    def start(self, invoker: Invoker):
        pass

    @abstractmethod
    def create_batch_process(self, batches: list[Batch], graph: Graph) -> BatchProcessResponse:
        pass

    @abstractmethod
    def run_batch_process(self, batch_id: str):
        pass

    @abstractmethod
    def cancel_batch_process(self, batch_process_id: str):
        pass


class BatchManager(BatchManagerBase):
    """Responsible for managing currently running and scheduled batch jobs"""

    __invoker: Invoker
    __batches: list[BatchProcess]
    __batch_process_storage: BatchProcessStorageBase

    def __init__(self, batch_process_storage: BatchProcessStorageBase) -> None:
        super().__init__()
        self.__batch_process_storage = batch_process_storage

    def start(self, invoker: Invoker) -> None:
        # if we do want multithreading at some point, we could make this configurable
        self.__invoker = invoker
        self.__batches = list()
        local_handler.register(event_name=EventServiceBase.session_event, _func=self.on_event)

    async def on_event(self, event: Event):
        event_name = event[1]["event"]

        match event_name:
            case "graph_execution_state_complete":
                await self.process(event, False)
            case "invocation_error":
                await self.process(event, True)

        return event

    async def process(self, event: Event, err: bool):
        data = event[1]["data"]
        batch_session = self.__batch_process_storage.get_session(data["graph_execution_state_id"])
        if not batch_session:
            return
        updateSession = BatchSessionChanges(state="error" if err else "completed")
        batch_session = self.__batch_process_storage.update_session_state(
            batch_session.batch_id,
            batch_session.session_id,
            updateSession,
        )
        batch_process = self.__batch_process_storage.get(batch_session.batch_id)
        if not batch_process.canceled:
            self.run_batch_process(batch_process.batch_id)

    def _create_batch_session(self, batch_process: BatchProcess, batch_indices: list[int]) -> GraphExecutionState:
        graph = copy.deepcopy(batch_process.graph)
        batches = batch_process.batches
        g = graph.nx_graph_flat()
        sorted_nodes = nx.topological_sort(g)
        for npath in sorted_nodes:
            node = graph.get_node(npath)
            (index, batch) = next(((i, b) for i, b in enumerate(batches) if b.node_id in node.id), (None, None))
            if batch:
                batch_index = batch_indices[index]
                datum = batch.data[batch_index]
                for key in datum:
                    node.__dict__[key] = datum[key]
                graph.update_node(npath, node)

        return GraphExecutionState(graph=graph)

    def run_batch_process(self, batch_id: str):
        try:
            created_session = self.__batch_process_storage.get_created_session(batch_id)
        except BatchSessionNotFoundException:
            return
        ges = self.__invoker.services.graph_execution_manager.get(created_session.session_id)
        self.__invoker.invoke(ges, invoke_all=True)

    def _valid_batch_config(self, batch_process: BatchProcess) -> bool:
        # TODO: Check that the node_ids in the batches are unique
        # TODO: Validate data types are correct for each batch data
        return True

    def create_batch_process(self, batches: list[Batch], graph: Graph) -> BatchProcessResponse:
        batch_process = BatchProcess(
            batches=batches,
            graph=graph,
        )
        if not self._valid_batch_config(batch_process):
            return None
        batch_process = self.__batch_process_storage.save(batch_process)
        sessions = self._create_sessions(batch_process)
        return BatchProcessResponse(
            batch_id=batch_process.batch_id,
            session_ids=[session.session_id for session in sessions],
        )

    def _create_sessions(self, batch_process: BatchProcess) -> list[BatchSession]:
        batch_indices = list()
        sessions = list()
        for batch in batch_process.batches:
            batch_indices.append(list(range(len(batch.data))))
        all_batch_indices = product(*batch_indices)
        for bi in all_batch_indices:
            ges = self._create_batch_session(batch_process, bi)
            self.__invoker.services.graph_execution_manager.set(ges)
            batch_session = BatchSession(batch_id=batch_process.batch_id, session_id=ges.id, state="created")
            sessions.append(self.__batch_process_storage.create_session(batch_session))
        return sessions

    def cancel_batch_process(self, batch_process_id: str):
        self.__batch_process_storage.cancel(batch_process_id)
