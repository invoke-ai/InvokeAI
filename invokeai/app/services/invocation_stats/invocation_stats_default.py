import time
from contextlib import contextmanager

import psutil
import torch

import invokeai.backend.util.logging as logger
from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_management.model_cache import CacheStats

from .invocation_stats_base import InvocationStatsServiceBase
from .invocation_stats_common import GraphExecutionStats, NodeExecutionStats

# Size of 1GB in bytes.
GB = 2**30


class InvocationStatsService(InvocationStatsServiceBase):
    """Accumulate performance information about a running graph. Collects time spent in each node,
    as well as the maximum and current VRAM utilisation for CUDA systems"""

    def __init__(self):
        # Maps graph_execution_state_id to GraphExecutionStats.
        self._stats: dict[str, GraphExecutionStats] = {}
        # Maps graph_execution_state_id to model manager CacheStats.
        self._cache_stats: dict[str, CacheStats] = {}

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    @contextmanager
    def collect_stats(self, invocation: BaseInvocation, graph_execution_state_id: str):
        if not self._stats.get(graph_execution_state_id):
            # First time we're seeing this graph_execution_state_id.
            self._stats[graph_execution_state_id] = GraphExecutionStats()
            self._cache_stats[graph_execution_state_id] = CacheStats()

            # Prune stale stats. There should be none since we're starting a new graph, but just in case.
            self._prune_stale_stats()

        # Record state before the invocation.
        start_time = time.time()
        start_ram = psutil.Process().memory_info().rss
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self._invoker.services.model_manager:
            self._invoker.services.model_manager.collect_cache_stats(self._cache_stats[graph_execution_state_id])

        try:
            # Let the invocation run.
            yield None
        finally:
            # Record state after the invocation.
            node_stats = NodeExecutionStats(
                invocation_type=invocation.type,
                start_time=start_time,
                end_time=time.time(),
                start_ram_gb=start_ram / GB,
                end_ram_gb=psutil.Process().memory_info().rss / GB,
                peak_vram_gb=torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else 0.0,
            )
            self._stats[graph_execution_state_id].add_node_execution_stats(node_stats)

    def _prune_stale_stats(self):
        """Check all graphs being tracked and prune any that have completed/errored.

        This shouldn't be necessary, but we don't have totally robust upstream handling of graph completions/errors, so
        for now we call this function periodically to prevent them from accumulating.
        """
        to_prune = []
        for graph_execution_state_id in self._stats:
            try:
                graph_execution_state = self._invoker.services.graph_execution_manager.get(graph_execution_state_id)
            except Exception:
                # TODO(ryand): What would cause this? Should this exception just be allowed to propagate?
                logger.warning(f"Failed to get graph state for {graph_execution_state_id}.")
                continue

            if not graph_execution_state.is_complete():
                # The graph is still running, don't prune it.
                continue

            to_prune.append(graph_execution_state_id)

        for graph_execution_state_id in to_prune:
            del self._stats[graph_execution_state_id]
            del self._cache_stats[graph_execution_state_id]

        if len(to_prune) > 0:
            logger.info(f"Pruned stale graph stats for {to_prune}.")

    def reset_stats(self, graph_execution_state_id: str):
        try:
            del self._stats[graph_execution_state_id]
            del self._cache_stats[graph_execution_state_id]
        except KeyError as e:
            logger.warning(f"Attempted to clear statistics for unknown graph {graph_execution_state_id}: {e}.")

    def log_stats(self, graph_execution_state_id: str):
        graph_stats = self._stats[graph_execution_state_id]
        cache_stats = self._cache_stats[graph_execution_state_id]

        log = graph_stats.get_pretty_log(graph_execution_state_id)

        hwm = cache_stats.high_watermark / GB
        tot = cache_stats.cache_size / GB
        loaded = sum(list(cache_stats.loaded_model_sizes.values())) / GB
        log += f"RAM used to load models: {loaded:4.2f}G\n"
        if torch.cuda.is_available():
            log += f"VRAM in use: {(torch.cuda.memory_allocated() / GB):4.3f}G\n"
        log += "RAM cache statistics:\n"
        log += f"   Model cache hits: {cache_stats.hits}\n"
        log += f"   Model cache misses: {cache_stats.misses}\n"
        log += f"   Models cached: {cache_stats.in_cache}\n"
        log += f"   Models cleared from cache: {cache_stats.cleared}\n"
        log += f"   Cache high water mark: {hwm:4.2f}/{tot:4.2f}G\n"
        logger.info(log)

        del self._stats[graph_execution_state_id]
        del self._cache_stats[graph_execution_state_id]
