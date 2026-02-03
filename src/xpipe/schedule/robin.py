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

    def schedule(self, system: System, graph: OpGraph):
        """Schedule the operators in the graph onto the pipelines in the system.

        Args:
            system (System): The system containing pipelines and memory slices.
            graph (OpGraph): The operator graph to be scheduled.

        Returns:
            dict: A mapping from operator to assigned pipeline.
        """

        end_times = {pipe: 0.0 for pipe in system.pipelines}
        for op in nx.topological_sort(graph):
            assert isinstance(op, BaseOp)
            best_pipe: Pipeline | None = None
            for pipe in system.pipelines:
                if not pipe.is_compatible(op):
                    continue
                if best_pipe is None or len(pipe) < len(best_pipe):
                    best_pipe = pipe
            assert best_pipe is not None
            op.start_time = end_times[best_pipe]
            op.end_time = op.start_time + self.get_cost(op, best_pipe)
            logger.debug(f"schedule {op} on {best_pipe}")
            best_pipe.push(op)
            end_times[best_pipe] = op.end_time
