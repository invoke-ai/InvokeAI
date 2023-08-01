# Copyright 2023 Lincoln D. Stein <lincoln.stein@gmail.com>
"""Utility to collect execution time and GPU usage stats on invocations in flight"""

"""
Usage:
statistics = InvocationStats()  # keep track of performance metrics
...
with statistics.collect_stats(invocation, graph_execution_state):
    outputs = invocation.invoke(
                                InvocationContext(
                                services=self.__invoker.services,
                                graph_execution_state_id=graph_execution_state.id,
                            )
                        )
...
statistics.log_stats()

Typical output:
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> Node                 Calls   Seconds
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> main_model_loader        1   0.006s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> clip_skip                1   0.005s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> compel                   2   0.351s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> rand_int                 1   0.001s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> range_of_size            1   0.001s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> iterate                  1   0.001s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> metadata_accumulator     1   0.002s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> noise                    1   0.002s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> t2l                      1   3.117s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> l2i                      1   0.377s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> TOTAL: 3.865s
[2023-08-01 17:34:44,585]::[InvokeAI]::INFO --> Max VRAM used for execution: 3.12G.
[2023-08-01 17:34:44,586]::[InvokeAI]::INFO --> Current VRAM utilization 2.31G.
"""

import time
from typing import Dict, List

import torch

from .graph import GraphExecutionState
from .invocation_queue import InvocationQueueItem
from ..invocations.baseinvocation import BaseInvocation

import invokeai.backend.util.logging as logger

class InvocationStats():
    """Accumulate performance information about a running graph. Collects time spent in each node,
    as well as the maximum and current VRAM utilisation for CUDA systems"""

    def __init__(self):
        self._stats: Dict[str, int] = {}
        
    class StatsContext():
        def __init__(self, invocation: BaseInvocation, collector):
            self.invocation = invocation
            self.collector = collector
            self.start_time = 0

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, *args):
            self.collector.log_time(self.invocation.type, time.time() - self.start_time)
    
    def collect_stats(self,
                      invocation: BaseInvocation,
                      graph_execution_state: GraphExecutionState,
                      ) -> StatsContext:
        """
        Return a context object that will capture the statistics.
        :param invocation: BaseInvocation object from the current graph.
        :param graph_execution_state: GraphExecutionState object from the current session.
        """
        if len(graph_execution_state.executed)==0:  # new graph is starting
            self.reset_stats()
        self._current_graph_state = graph_execution_state
        sc = self.StatsContext(invocation, self)
        return self.StatsContext(invocation, self)

    def reset_stats(self):
        """Zero the statistics. Ordinarily called internally."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._stats: Dict[str, List[int, float]] = {}


    def log_time(self, invocation_type: str, time_used: float):
        """
        Add timing information on execution of a node. Usually
        used internally.
        :param invocation_type: String literal type of the node
        :param time_used: Floating point seconds used by node's exection
        """
        if not self._stats.get(invocation_type):
            self._stats[invocation_type] = [0, 0.0]
        self._stats[invocation_type][0] += 1
        self._stats[invocation_type][1] += time_used
    
    def log_stats(self):
        """
        Send the statistics to the system logger at the info level.
        Stats will only be printed if when the execution of the graph
        is complete.
        """
        if self._current_graph_state.is_complete():
            logger.info('Node                 Calls   Seconds')
            for node_type, (calls, time_used) in self._stats.items():
                logger.info(f'{node_type:<20} {calls:>5}   {time_used:4.3f}s')
                
            total_time = sum([ticks for _,ticks in self._stats.values()])
            logger.info(f'TOTAL: {total_time:4.3f}s')
            if torch.cuda.is_available():
                logger.info('Max VRAM used for execution: '+'%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9))
                logger.info('Current VRAM utilization '+'%4.2fG' % (torch.cuda.memory_allocated() / 1e9))
                
