# Copyright 2023 Lincoln D. Stein <lincoln.stein@gmail.com>
"""Utility to collect execution time and GPU usage stats on invocations in flight

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

import psutil
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
from .model_manager_service import ModelManagerService
from invokeai.backend.model_management.model_cache import CacheStats

# size of GIG in bytes
GIG = 1073741824


@dataclass
class NodeStats:
    """Class for tracking execution stats of an invocation node"""

    calls: int = 0
    time_used: float = 0.0  # seconds
    max_vram: float = 0.0  # GB
    cache_hits: int = 0
    cache_misses: int = 0
    cache_high_watermark: int = 0


@dataclass
class NodeLog:
    """Class for tracking node usage"""

    # {node_type => NodeStats}
    nodes: Dict[str, NodeStats] = field(default_factory=dict)


class InvocationStatsServiceBase(ABC):
    "Abstract base class for recording node memory/time performance statistics"

    graph_execution_manager: ItemStorageABC["GraphExecutionState"]
    # {graph_id => NodeLog}
    _stats: Dict[str, NodeLog]
    _cache_stats: Dict[str, CacheStats]
    ram_used: float
    ram_changed: float

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

    @abstractmethod
    def update_mem_stats(
        self,
        ram_used: float,
        ram_changed: float,
    ):
        """
        Update the collector with RAM memory usage info.

        :param ram_used: How much RAM is currently in use.
        :param ram_changed: How much RAM changed since last generation.
        """
        pass


class InvocationStatsService(InvocationStatsServiceBase):
    """Accumulate performance information about a running graph. Collects time spent in each node,
    as well as the maximum and current VRAM utilisation for CUDA systems"""

    def __init__(self, graph_execution_manager: ItemStorageABC["GraphExecutionState"]):
        self.graph_execution_manager = graph_execution_manager
        # {graph_id => NodeLog}
        self._stats: Dict[str, NodeLog] = {}
        self._cache_stats: Dict[str, CacheStats] = {}
        self.ram_used: float = 0.0
        self.ram_changed: float = 0.0

    class StatsContext:
        """Context manager for collecting statistics."""

        invocation: BaseInvocation
        collector: "InvocationStatsServiceBase"
        graph_id: str
        start_time: float
        ram_used: int
        model_manager: ModelManagerService

        def __init__(
            self,
            invocation: BaseInvocation,
            graph_id: str,
            model_manager: ModelManagerService,
            collector: "InvocationStatsServiceBase",
        ):
            """Initialize statistics for this run."""
            self.invocation = invocation
            self.collector = collector
            self.graph_id = graph_id
            self.start_time = 0.0
            self.ram_used = 0
            self.model_manager = model_manager

        def __enter__(self):
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.ram_used = psutil.Process().memory_info().rss
            if self.model_manager:
                self.model_manager.collect_cache_stats(self.collector._cache_stats[self.graph_id])

        def __exit__(self, *args):
            """Called on exit from the context."""
            ram_used = psutil.Process().memory_info().rss
            self.collector.update_mem_stats(
                ram_used=ram_used / GIG,
                ram_changed=(ram_used - self.ram_used) / GIG,
            )
            self.collector.update_invocation_stats(
                graph_id=self.graph_id,
                invocation_type=self.invocation.type,  # type: ignore - `type` is not on the `BaseInvocation` model, but *is* on all invocations
                time_used=time.time() - self.start_time,
                vram_used=torch.cuda.max_memory_allocated() / GIG if torch.cuda.is_available() else 0.0,
            )

    def collect_stats(
        self,
        invocation: BaseInvocation,
        graph_execution_state_id: str,
        model_manager: ModelManagerService,
    ) -> StatsContext:
        if not self._stats.get(graph_execution_state_id):  # first time we're seeing this
            self._stats[graph_execution_state_id] = NodeLog()
            self._cache_stats[graph_execution_state_id] = CacheStats()
        return self.StatsContext(invocation, graph_execution_state_id, model_manager, self)

    def reset_all_stats(self):
        """Zero all statistics"""
        self._stats = {}

    def reset_stats(self, graph_execution_id: str):
        try:
            self._stats.pop(graph_execution_id)
        except KeyError:
            logger.warning(f"Attempted to clear statistics for unknown graph {graph_execution_id}")

    def update_mem_stats(
        self,
        ram_used: float,
        ram_changed: float,
    ):
        self.ram_used = ram_used
        self.ram_changed = ram_changed

    def update_invocation_stats(
        self,
        graph_id: str,
        invocation_type: str,
        time_used: float,
        vram_used: float,
    ):
        if not self._stats[graph_id].nodes.get(invocation_type):
            self._stats[graph_id].nodes[invocation_type] = NodeStats()
        stats = self._stats[graph_id].nodes[invocation_type]
        stats.calls += 1
        stats.time_used += time_used
        stats.max_vram = max(stats.max_vram, vram_used)

    def log_stats(self):
        completed = set()
        errored = set()
        for graph_id, node_log in self._stats.items():
            try:
                current_graph_state = self.graph_execution_manager.get(graph_id)
            except Exception:
                errored.add(graph_id)
                continue

            if not current_graph_state.is_complete():
                continue

            total_time = 0
            logger.info(f"Graph stats: {graph_id}")
            logger.info(f"{'Node':>30} {'Calls':>7}{'Seconds':>9} {'VRAM Used':>10}")
            for node_type, stats in self._stats[graph_id].nodes.items():
                logger.info(f"{node_type:>30}  {stats.calls:>4}   {stats.time_used:7.3f}s     {stats.max_vram:4.3f}G")
                total_time += stats.time_used

            cache_stats = self._cache_stats[graph_id]
            hwm = cache_stats.high_watermark / GIG
            tot = cache_stats.cache_size / GIG
            loaded = sum([v for v in cache_stats.loaded_model_sizes.values()]) / GIG

            logger.info(f"TOTAL GRAPH EXECUTION TIME:  {total_time:7.3f}s")
            logger.info("RAM used by InvokeAI process: " + "%4.2fG" % self.ram_used + f" ({self.ram_changed:+5.3f}G)")
            logger.info(f"RAM used to load models: {loaded:4.2f}G")
            if torch.cuda.is_available():
                logger.info("VRAM in use: " + "%4.3fG" % (torch.cuda.memory_allocated() / GIG))
            logger.info("RAM cache statistics:")
            logger.info(f"   Model cache hits: {cache_stats.hits}")
            logger.info(f"   Model cache misses: {cache_stats.misses}")
            logger.info(f"   Models cached: {cache_stats.in_cache}")
            logger.info(f"   Models cleared from cache: {cache_stats.cleared}")
            logger.info(f"   Cache high water mark: {hwm:4.2f}/{tot:4.2f}G")

            completed.add(graph_id)

        for graph_id in completed:
            del self._stats[graph_id]
            del self._cache_stats[graph_id]

        for graph_id in errored:
            del self._stats[graph_id]
            del self._cache_stats[graph_id]
