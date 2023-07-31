
import networkx as nx
import uuid
import copy

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event
from typing import (
    Optional,
    Union,
)

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
)
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.graph import Graph, GraphExecutionState
from invokeai.app.services.invoker import Invoker


InvocationsUnion = Union[BaseInvocation.get_invocations()]  # type: ignore
class Batch(BaseModel):
    data: list[InvocationsUnion] = Field(description="Mapping of ")
    node_id: str = Field(description="ID of the node to batch")


class BatchProcess(BaseModel):
    batch_id: Optional[str] = Field(default_factory=uuid.uuid4().__str__, description="Identifier for this batch")
    sessions: list[str] = Field(description="Tracker for which batch is currently being processed", default_factory=list)
    batches: list[Batch] = Field(
        description="List of batch configs to apply to this session",
        default_factory=list,
    )
    batch_indices: list[int] = Field(description="Tracker for which batch is currently being processed", default_factory=list)
    graph: Graph = Field(description="The graph being executed")


class BatchManagerBase(ABC):
    @abstractmethod
    def start(
        self,
        invoker: Invoker
    ):
        pass

    @abstractmethod
    def run_batch_process(
        self,
        batches: list[Batch],
        graph: Graph
    ) -> BatchProcess:
        pass

    @abstractmethod
    def cancel_batch_process(
        self,
        batch_process_id: str
    ):
        pass


class BatchManager(BatchManagerBase):
    """Responsible for managing currently running and scheduled batch jobs"""
    __invoker: Invoker
    __batches: list[BatchProcess]


    def start(self, invoker) -> None:
        # if we do want multithreading at some point, we could make this configurable
        self.__invoker = invoker
        self.__batches = list()
        local_handler.register(
            event_name=EventServiceBase.session_event, _func=self.on_event
        )

    async def on_event(self, event: Event):
        event_name = event[1]["event"]

        match event_name:
            case "graph_execution_state_complete":
                await self.process(event)
            case "invocation_error":
                await self.process(event)

        return event
    
    async def process(self, event: Event):
        data = event[1]["data"]
        batchTarget = None
        for batch in self.__batches:
            if data['graph_execution_state_id'] in batch.sessions:
                batchTarget = batch
                break
        
        if batchTarget == None:
            return
        
        if sum(batchTarget.batch_indices) == 0:
            self.__batches = [batch for batch in self.__batches if batch != batchTarget]
            return
        
        batchTarget.batch_indices = self._next_batch_index(batchTarget)
        ges = self._next_batch_session(batchTarget)
        batchTarget.sessions.append(ges.id)
        self.__invoker.services.graph_execution_manager.set(ges)
        self.__invoker.invoke(ges, invoke_all=True)

    def _next_batch_session(self, batch_process: BatchProcess) -> GraphExecutionState:
        graph = copy.deepcopy(batch_process.graph)
        batches = batch_process.batches
        g = graph.nx_graph_flat()
        sorted_nodes = nx.topological_sort(g)
        for npath in sorted_nodes:
            node = graph.get_node(npath)
            (index, batch) = next(((i,b) for i,b in enumerate(batches) if b.node_id in node.id), (None, None))
            if batch:
                batch_index = batch_process.batch_indices[index]
                datum = batch.data[batch_index]
                datum.id = node.id
                graph.update_node(npath, datum)
        
        return GraphExecutionState(graph=graph)


    def _next_batch_index(self, batch_process: BatchProcess):
        batch_indicies = batch_process.batch_indices.copy()
        for index in range(len(batch_indicies)):
            if batch_indicies[index] > 0:
                batch_indicies[index] -= 1
                break
        return batch_indicies


    def run_batch_process(
        self,
        batches: list[Batch],
        graph: Graph
    ) -> BatchProcess:
        batch_indices = list()
        for batch in batches:
            batch_indices.append(len(batch.data)-1)
        batch_process = BatchProcess(
            batches = batches,
            batch_indices = batch_indices,
            graph = graph,
        )
        ges = self._next_batch_session(batch_process)
        batch_process.sessions.append(ges.id)
        self.__batches.append(batch_process)
        self.__invoker.services.graph_execution_manager.set(ges)
        self.__invoker.invoke(ges, invoke_all=True)
        return batch_process

    def cancel_batch_process(
        self,
        batch_process_id: str
    ):
        self.__batches = [batch for batch in self.__batches if batch.id != batch_process_id]
