from collections import defaultdict
from dataclasses import dataclass

# size of GIG in bytes
GIG = 1073741824


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

    def get_pretty_log(self, graph_execution_state_id: str) -> str:
        log = f"Graph stats: {graph_execution_state_id}\n"
        log += f"{'Node':>30} {'Calls':>7}{'Seconds':>9} {'VRAM Used':>10}\n"

        # Log stats aggregated by node type.
        node_stats_by_type: dict[str, list[NodeExecutionStats]] = defaultdict(list)
        for node_stats in self._node_stats_list:
            node_stats_by_type[node_stats.invocation_type].append(node_stats)

        for node_type, node_type_stats_list in node_stats_by_type.items():
            num_calls = len(node_type_stats_list)
            time_used = sum([n.total_time() for n in node_type_stats_list])
            peak_vram = max([n.peak_vram_gb for n in node_type_stats_list])
            log += f"{node_type:>30}  {num_calls:>4}   {time_used:7.3f}s     {peak_vram:4.3f}G\n"

        # Log stats for the entire graph.
        log += f"TOTAL GRAPH EXECUTION TIME: {self.get_total_run_time():7.3f}s\n"

        first_node = self.get_first_node_stats()
        last_node = self.get_last_node_stats()
        if first_node is not None and last_node is not None:
            total_wall_time = last_node.end_time - first_node.start_time
            ram_change = last_node.end_ram_gb - first_node.start_ram_gb
            log += f"TOTAL GRAPH WALL TIME: {total_wall_time:7.3f}s\n"
            log += f"RAM used by InvokeAI process: {last_node.end_ram_gb:4.2f}G ({ram_change:+5.3f}G)\n"

        return log
