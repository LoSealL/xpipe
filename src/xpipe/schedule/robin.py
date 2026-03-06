"""
Copyright (C) 2026 The XPIPE Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import networkx as nx
from loguru import logger

from ..ir.graph import BaseOp, OpGraph
from ..system import Pipeline, System
from . import Scheduler


class RoundRobinScheduler(Scheduler):
    """A simple Round Robin scheduler."""

    @staticmethod
    def _dep_ready_time(op: BaseOp, pipe: Pipeline, graph: OpGraph) -> int:
        """Compute when all dependencies of ``op`` are ready on ``pipe``."""

        dep_ready = 0
        for dep in op.deps:
            comm = graph.get_edge_data(dep, op, {}).get("comm", 0)
            ready = dep.end_cycle + (comm if dep not in pipe else 0)
            dep_ready = max(dep_ready, ready)
        return dep_ready

    @staticmethod
    def _next_compatible_pipe(
        pipelines: list[Pipeline], op: BaseOp, start_index: int
    ) -> tuple[Pipeline, int]:
        """Pick the next compatible pipeline in round-robin order."""

        pipe_count = len(pipelines)
        for step in range(pipe_count):
            idx = (start_index + step) % pipe_count
            pipe = pipelines[idx]
            if pipe.is_compatible(op):
                return pipe, idx
        raise ValueError(f"No compatible pipeline found for operator {op.name}.")

    def schedule(self, system: System, graph: OpGraph):
        """Schedule the operators in the graph onto the pipelines in the system.

        Args:
            system (System): The system containing pipelines and memory slices.
            graph (OpGraph): The operator graph to be scheduled.

        Returns:
            dict: A mapping from operator to assigned pipeline.
        """

        pipelines = list(system.pipelines)
        if not pipelines:
            raise ValueError("System has no pipelines to schedule operators.")

        end_times = {pipe: 0 for pipe in pipelines}
        next_pipe_index = 0
        for op in nx.topological_sort(graph):
            assert isinstance(op, BaseOp)
            pipe, used_index = self._next_compatible_pipe(
                pipelines, op, next_pipe_index
            )
            dep_ready = self._dep_ready_time(op, pipe, graph)
            pipe_ready = end_times[pipe]
            start = max(dep_ready, pipe_ready)
            end = start + int(self.get_cost(op, pipe))
            op.start_cycle = start
            op.end_cycle = end
            logger.debug(f"schedule {op} on {pipe}")
            pipe.push(op)
            end_times[pipe] = end
            next_pipe_index = (used_index + 1) % len(pipelines)
