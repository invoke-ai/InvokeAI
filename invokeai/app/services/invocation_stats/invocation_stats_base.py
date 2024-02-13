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

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.invocation_stats.invocation_stats_common import InvocationStatsSummary


class InvocationStatsServiceBase(ABC):
    "Abstract base class for recording node memory/time performance statistics"

    @abstractmethod
    def __init__(self):
        """
        Initialize the InvocationStatsService and reset counters to zero
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
        :param graph_execution_state_id: The id of the current session.
        """
        pass

    @abstractmethod
    def reset_stats(self, graph_execution_state_id: str):
        """
        Reset all statistics for the indicated graph.
        :param graph_execution_state_id: The id of the session whose stats to reset.
        :raises GESStatsNotFoundError: if the graph isn't tracked in the stats.
        """
        pass

    @abstractmethod
    def log_stats(self, graph_execution_state_id: str):
        """
        Write out the accumulated statistics to the log or somewhere else.
        :param graph_execution_state_id: The id of the session whose stats to log.
        :raises GESStatsNotFoundError: if the graph isn't tracked in the stats.
        """
        pass

    @abstractmethod
    def get_stats(self, graph_execution_state_id: str) -> InvocationStatsSummary:
        """
        Gets the accumulated statistics for the indicated graph.
        :param graph_execution_state_id: The id of the session whose stats to get.
        :raises GESStatsNotFoundError: if the graph isn't tracked in the stats.
        """
        pass

    @abstractmethod
    def dump_stats(self, graph_execution_state_id: str, output_path: Path) -> None:
        """
        Write out the accumulated statistics to the indicated path as JSON.
        :param graph_execution_state_id: The id of the session whose stats to dump.
        :param output_path: The file to write the stats to.
        :raises GESStatsNotFoundError: if the graph isn't tracked in the stats.
        """
        pass
