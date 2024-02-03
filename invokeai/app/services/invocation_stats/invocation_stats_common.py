from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional


class GESStatsNotFoundError(Exception):
    """Raised when execution stats are not found for a given Graph Execution State."""


@dataclass
class NodeExecutionStatsSummary:
    """The stats for a specific type of node."""

    node_type: str
    num_calls: int
    time_used_seconds: float
    peak_vram_gb: float


@dataclass
class ModelCacheStatsSummary:
    """The stats for the model cache."""

    high_water_mark_gb: float
    cache_size_gb: float
    total_usage_gb: float
    cache_hits: int
    cache_misses: int
    models_cached: int
    models_cleared: int


@dataclass
class GraphExecutionStatsSummary:
    """The stats for the graph execution state."""

    graph_execution_state_id: str
    execution_time_seconds: float
    # `wall_time_seconds`, `ram_usage_gb` and `ram_change_gb` are derived from the node execution stats.
    # In some situations, there are no node stats, so these values are optional.
    wall_time_seconds: Optional[float]
    ram_usage_gb: Optional[float]
    ram_change_gb: Optional[float]


@dataclass
class InvocationStatsSummary:
    """
    The accumulated stats for a graph execution.
    Its `__str__` method returns a human-readable stats summary.
    """

    vram_usage_gb: Optional[float]
    graph_stats: GraphExecutionStatsSummary
    model_cache_stats: ModelCacheStatsSummary
    node_stats: list[NodeExecutionStatsSummary]

    def __str__(self) -> str:
        _str = ""
        _str = f"Graph stats: {self.graph_stats.graph_execution_state_id}\n"
        _str += f"{'Node':>30} {'Calls':>7} {'Seconds':>9} {'VRAM Used':>10}\n"

        for summary in self.node_stats:
            _str += f"{summary.node_type:>30} {summary.num_calls:>7} {summary.time_used_seconds:>8.3f}s {summary.peak_vram_gb:>9.3f}G\n"

        _str += f"TOTAL GRAPH EXECUTION TIME: {self.graph_stats.execution_time_seconds:7.3f}s\n"

        if self.graph_stats.wall_time_seconds is not None:
            _str += f"TOTAL GRAPH WALL TIME: {self.graph_stats.wall_time_seconds:7.3f}s\n"

        if self.graph_stats.ram_usage_gb is not None and self.graph_stats.ram_change_gb is not None:
            _str += f"RAM used by InvokeAI process: {self.graph_stats.ram_usage_gb:4.2f}G ({self.graph_stats.ram_change_gb:+5.3f}G)\n"

        _str += f"RAM used to load models: {self.model_cache_stats.total_usage_gb:4.2f}G\n"
        if self.vram_usage_gb:
            _str += f"VRAM in use: {self.vram_usage_gb:4.3f}G\n"
        _str += "RAM cache statistics:\n"
        _str += f"   Model cache hits: {self.model_cache_stats.cache_hits}\n"
        _str += f"   Model cache misses: {self.model_cache_stats.cache_misses}\n"
        _str += f"   Models cached: {self.model_cache_stats.models_cached}\n"
        _str += f"   Models cleared from cache: {self.model_cache_stats.models_cleared}\n"
        _str += f"   Cache high water mark: {self.model_cache_stats.high_water_mark_gb:4.2f}/{self.model_cache_stats.cache_size_gb:4.2f}G\n"

        return _str

    def as_dict(self) -> dict[str, Any]:
        """Returns the stats as a dictionary."""
        return asdict(self)


@dataclass
class NodeExecutionStats:
    """Class for tracking execution stats of an invocation node."""

    invocation_type: str

    start_time: float  # Seconds since the epoch.
    end_time: float  # Seconds since the epoch.

    start_ram_gb: float  # GB
    end_ram_gb: float  # GB

    peak_vram_gb: float  # GB

    def total_time(self) -> float:
        return self.end_time - self.start_time


class GraphExecutionStats:
    """Class for tracking execution stats of a graph."""

    def __init__(self):
        self._node_stats_list: list[NodeExecutionStats] = []

    def add_node_execution_stats(self, node_stats: NodeExecutionStats):
        self._node_stats_list.append(node_stats)

    def get_total_run_time(self) -> float:
        """Get the total time spent executing nodes in the graph."""
        total = 0.0
        for node_stats in self._node_stats_list:
            total += node_stats.total_time()
        return total

    def get_first_node_stats(self) -> NodeExecutionStats | None:
        """Get the stats of the first node in the graph (by start_time)."""
        first_node = None
        for node_stats in self._node_stats_list:
            if first_node is None or node_stats.start_time < first_node.start_time:
                first_node = node_stats

        assert first_node is not None
        return first_node

    def get_last_node_stats(self) -> NodeExecutionStats | None:
        """Get the stats of the last node in the graph (by end_time)."""
        last_node = None
        for node_stats in self._node_stats_list:
            if last_node is None or node_stats.end_time > last_node.end_time:
                last_node = node_stats

        return last_node

    def get_graph_stats_summary(self, graph_execution_state_id: str) -> GraphExecutionStatsSummary:
        """Get a summary of the graph stats."""
        first_node = self.get_first_node_stats()
        last_node = self.get_last_node_stats()

        wall_time_seconds: Optional[float] = None
        ram_usage_gb: Optional[float] = None
        ram_change_gb: Optional[float] = None

        if last_node and first_node:
            wall_time_seconds = last_node.end_time - first_node.start_time
            ram_usage_gb = last_node.end_ram_gb
            ram_change_gb = last_node.end_ram_gb - first_node.start_ram_gb

        return GraphExecutionStatsSummary(
            graph_execution_state_id=graph_execution_state_id,
            execution_time_seconds=self.get_total_run_time(),
            wall_time_seconds=wall_time_seconds,
            ram_usage_gb=ram_usage_gb,
            ram_change_gb=ram_change_gb,
        )

    def get_node_stats_summaries(self) -> list[NodeExecutionStatsSummary]:
        """Get a summary of the node stats."""
        summaries: list[NodeExecutionStatsSummary] = []
        node_stats_by_type: dict[str, list[NodeExecutionStats]] = defaultdict(list)

        for node_stats in self._node_stats_list:
            node_stats_by_type[node_stats.invocation_type].append(node_stats)

        for node_type, node_type_stats_list in node_stats_by_type.items():
            num_calls = len(node_type_stats_list)
            time_used = sum([n.total_time() for n in node_type_stats_list])
            peak_vram = max([n.peak_vram_gb for n in node_type_stats_list])
            summary = NodeExecutionStatsSummary(
                node_type=node_type, num_calls=num_calls, time_used_seconds=time_used, peak_vram_gb=peak_vram
            )
            summaries.append(summary)

        return summaries
