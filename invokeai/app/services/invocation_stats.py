# Copyright 2023 Lincoln D. Stein <lincoln.stein@gmail.com>
"""Utility to collect execution time and GPU usage stats on invocations in flight"""

"""
Usage:

statistics = InvocationStatsService(graph_execution_manager)
with statistics.collect_stats(invocation, graph_execution_state.id):
      ... execute graphs...
statistics.log_stats()

Typical output:
[2023-08-02 18:03:04,507]::[InvokeAI]::INFO --> Graph stats: c7764585-9c68-4d9d-a199-55e8186790f3
[2023-08-02 18:03:04,507]::[InvokeAI]::INFO --> Node                 Calls  Seconds  VRAM Used
[2023-08-02 18:03:04,507]::[InvokeAI]::INFO --> main_model_loader        1   0.005s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> clip_skip                1   0.004s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> compel                   2   0.512s     0.26G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> rand_int                 1   0.001s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> range_of_size            1   0.001s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> iterate                  1   0.001s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> metadata_accumulator     1   0.002s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> noise                    1   0.002s     0.01G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> t2l                      1   3.541s     1.93G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> l2i                      1   0.679s     0.58G
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> TOTAL GRAPH EXECUTION TIME:  4.749s
[2023-08-02 18:03:04,508]::[InvokeAI]::INFO --> Current VRAM utilization 0.01G

The abstract base class for this class is InvocationStatsServiceBase. An implementing class which
writes to the system log is stored in InvocationServices.performance_statistics.
"""

import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Dict

import torch

import invokeai.backend.util.logging as logger

from ..invocations.baseinvocation import BaseInvocation
from .graph import GraphExecutionState
from .item_storage import ItemStorageABC


class InvocationStatsServiceBase(ABC):
    "Abstract base class for recording node memory/time performance statistics"

    @abstractmethod
    def __init__(self, graph_execution_manager: ItemStorageABC["GraphExecutionState"]):
        """
        Initialize the InvocationStatsService and reset counters to zero
        :param graph_execution_manager: Graph execution manager for this session
        """
        pass

    @abstractmethod
    def collect_stats(
        self,
        invocation: BaseInvocation,
        graph_execution_state_id: str,
    ) -> AbstractContextManager:
        """
        Return a context object that will capture the statistics on the execution
        of invocaation. Use with: to place around the part of the code that executes the invocation.
        :param invocation: BaseInvocation object from the current graph.
        :param graph_execution_state: GraphExecutionState object from the current session.
        """
        pass

    @abstractmethod
    def reset_stats(self, graph_execution_state_id: str):
        """
        Reset all statistics for the indicated graph
        :param graph_execution_state_id
        """
        pass

    @abstractmethod
    def reset_all_stats(self):
        """Zero all statistics"""
        pass

    @abstractmethod
    def update_invocation_stats(
        self,
        graph_id: str,
        invocation_type: str,
        time_used: float,
        vram_used: float,
    ):
        """
        Add timing information on execution of a node. Usually
        used internally.
        :param graph_id: ID of the graph that is currently executing
        :param invocation_type: String literal type of the node
        :param time_used: Time used by node's exection (sec)
        :param vram_used: Maximum VRAM used during exection (GB)
        """
        pass

    @abstractmethod
    def log_stats(self):
        """
        Write out the accumulated statistics to the log or somewhere else.
        """
        pass


@dataclass
class NodeStats:
    """Class for tracking execution stats of an invocation node"""

    calls: int = 0
    time_used: float = 0.0  # seconds
    max_vram: float = 0.0  # GB


@dataclass
class NodeLog:
    """Class for tracking node usage"""

    # {node_type => NodeStats}
    nodes: Dict[str, NodeStats] = field(default_factory=dict)


class InvocationStatsService(InvocationStatsServiceBase):
    """Accumulate performance information about a running graph. Collects time spent in each node,
    as well as the maximum and current VRAM utilisation for CUDA systems"""

    def __init__(self, graph_execution_manager: ItemStorageABC["GraphExecutionState"]):
        self.graph_execution_manager = graph_execution_manager
        # {graph_id => NodeLog}
        self._stats: Dict[str, NodeLog] = {}

    class StatsContext:
        def __init__(self, invocation: BaseInvocation, graph_id: str, collector: "InvocationStatsServiceBase"):
            self.invocation = invocation
            self.collector = collector
            self.graph_id = graph_id
            self.start_time = 0

        def __enter__(self):
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def __exit__(self, *args):
            self.collector.update_invocation_stats(
                self.graph_id,
                self.invocation.type,
                time.time() - self.start_time,
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            )

    def collect_stats(
        self,
        invocation: BaseInvocation,
        graph_execution_state_id: str,
    ) -> StatsContext:
        """
        Return a context object that will capture the statistics.
        :param invocation: BaseInvocation object from the current graph.
        :param graph_execution_state: GraphExecutionState object from the current session.
        """
        if not self._stats.get(graph_execution_state_id):  # first time we're seeing this
            self._stats[graph_execution_state_id] = NodeLog()
        return self.StatsContext(invocation, graph_execution_state_id, self)

    def reset_all_stats(self):
        """Zero all statistics"""
        self._stats = {}

    def reset_stats(self, graph_execution_id: str):
        """Zero the statistics for the indicated graph."""
        try:
            self._stats.pop(graph_execution_id)
        except KeyError:
            logger.warning(f"Attempted to clear statistics for unknown graph {graph_execution_id}")

    def update_invocation_stats(self, graph_id: str, invocation_type: str, time_used: float, vram_used: float):
        """
        Add timing information on execution of a node. Usually
        used internally.
        :param graph_id: ID of the graph that is currently executing
        :param invocation_type: String literal type of the node
        :param time_used: Floating point seconds used by node's exection
        """
        if not self._stats[graph_id].nodes.get(invocation_type):
            self._stats[graph_id].nodes[invocation_type] = NodeStats()
        stats = self._stats[graph_id].nodes[invocation_type]
        stats.calls += 1
        stats.time_used += time_used
        stats.max_vram = max(stats.max_vram, vram_used)

    def log_stats(self):
        """
        Send the statistics to the system logger at the info level.
        Stats will only be printed if when the execution of the graph
        is complete.
        """
        completed = set()
        for graph_id, node_log in self._stats.items():
            current_graph_state = self.graph_execution_manager.get(graph_id)
            if not current_graph_state.is_complete():
                continue

            total_time = 0
            logger.info(f"Graph stats: {graph_id}")
            logger.info("Node                 Calls    Seconds VRAM Used")
            for node_type, stats in self._stats[graph_id].nodes.items():
                logger.info(f"{node_type:<20} {stats.calls:>5}   {stats.time_used:7.3f}s     {stats.max_vram:4.2f}G")
                total_time += stats.time_used

            logger.info(f"TOTAL GRAPH EXECUTION TIME:  {total_time:7.3f}s")
            if torch.cuda.is_available():
                logger.info("Current VRAM utilization " + "%4.2fG" % (torch.cuda.memory_allocated() / 1e9))

            completed.add(graph_id)

        for graph_id in completed:
            del self._stats[graph_id]
