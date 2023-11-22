import time
from typing import Dict

import psutil
import torch

import invokeai.backend.util.logging as logger
from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
from invokeai.backend.model_management.model_cache import CacheStats

from .invocation_stats_base import InvocationStatsServiceBase
from .invocation_stats_common import GIG, NodeLog, NodeStats


class InvocationStatsService(InvocationStatsServiceBase):
    """Accumulate performance information about a running graph. Collects time spent in each node,
    as well as the maximum and current VRAM utilisation for CUDA systems"""

    _invoker: Invoker

    def __init__(self):
        # {graph_id => NodeLog}
        self._stats: Dict[str, NodeLog] = {}
        self._cache_stats: Dict[str, CacheStats] = {}
        self.ram_used: float = 0.0
        self.ram_changed: float = 0.0

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    class StatsContext:
        """Context manager for collecting statistics."""

        invocation: BaseInvocation
        collector: "InvocationStatsServiceBase"
        graph_id: str
        start_time: float
        ram_used: int
        model_manager: ModelManagerServiceBase

        def __init__(
            self,
            invocation: BaseInvocation,
            graph_id: str,
            model_manager: ModelManagerServiceBase,
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
                invocation_type=self.invocation.type,  # type: ignore # `type` is not on the `BaseInvocation` model, but *is* on all invocations
                time_used=time.time() - self.start_time,
                vram_used=torch.cuda.max_memory_allocated() / GIG if torch.cuda.is_available() else 0.0,
            )

    def collect_stats(
        self,
        invocation: BaseInvocation,
        graph_execution_state_id: str,
    ) -> StatsContext:
        if not self._stats.get(graph_execution_state_id):  # first time we're seeing this
            self._stats[graph_execution_state_id] = NodeLog()
            self._cache_stats[graph_execution_state_id] = CacheStats()
        return self.StatsContext(invocation, graph_execution_state_id, self._invoker.services.model_manager, self)

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
        for graph_id, _node_log in self._stats.items():
            try:
                current_graph_state = self._invoker.services.graph_execution_manager.get(graph_id)
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
            loaded = sum(list(cache_stats.loaded_model_sizes.values())) / GIG

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
