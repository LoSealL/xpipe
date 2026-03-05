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
import numpy as np
from loguru import logger

from ..ir.graph import OpGraph
from ..op import BaseOp
from ..system import Pipeline, System
from . import Scheduler


class HEFTScheduler(Scheduler):
    """Heterogeneous Earliest Finish Time (HEFT) scheduler.

    Reference::

    H. Topcuoglu, S. Hariri, and M.Y. Wu,
    Performance-effective and low-complexity task scheduling for heterogeneous computing
    IEEE Transactions on Parallel and Distributed Systems,
    vol. 13, no. 3, pp. 260–274, Mar. 2002, doi: 10.1109/71.993206.
    """

    def schedule(self, system: System, graph: OpGraph):
        self.ranku(system, graph)
        ops: list[BaseOp] = sorted(graph.nodes, key=lambda op: op["rank"], reverse=True)
        for op in ops:
            best_pipe: Pipeline | None = None
            best_eft = float("inf")
            for pipe in system.pipelines:
                if not pipe.is_compatible(op):
                    continue
                est = self.est(op, pipe, graph)
                eft = est + self.get_cost(op, pipe)
                if eft < best_eft:
                    best_eft = eft
                    best_pipe = pipe
            assert best_pipe is not None
            op.start_cycle = int(best_eft - self.get_cost(op, best_pipe))
            op.end_cycle = int(best_eft)
            logger.debug(f"schedule {op} on {best_pipe}")
            best_pipe.push(op)

    def ranku(self, system: System, graph: OpGraph):
        """Compute the upward rank for each operator in the graph.

        Args:
            graph (OpGraph): The operator graph.

        Changes:
            each node in graph will be assigned a 'rank' attribute.
        """

        rg = graph.reverse(copy=False)
        for op in nx.topological_sort(rg):
            ranku = np.mean(
                [
                    self.get_cost(op, pipe)
                    for pipe in system.pipelines
                    if pipe.is_compatible(op)
                ]
            ).item()
            if rg.pred[op]:
                ranku += max(
                    p["rank"] + graph.get_edge_data(op, p, {}).get("comm", 0)
                    for p in rg.predecessors(op)
                )
            op["rank"] = ranku

    def est(self, op: BaseOp, pipe: Pipeline, graph: OpGraph) -> int:
        """Compute the Earliest Start Time (EST) of an operator on a given pipeline.

        Args:
            op (BaseOp): The operator to be scheduled.
            pipe (Pipeline): The pipeline to schedule the operator on.
            graph (OpGraph): The operator graph.

        Returns:
            int: The earliest start time of the operator on the pipeline.
        """

        # earliest start time from all predecessors
        ests: list[int] = [0]
        for dep in op.deps:
            comm = graph.get_edge_data(dep, op, {}).get("comm", 0)
            ests.append(dep.end_cycle + (comm if dep not in pipe else 0))
        est = max(ests)
        free_times: list[tuple[float, float]] = []
        if len(pipe) == 0:
            free_times.append((0, float("inf")))
        else:
            # find all free time slots in the pipeline
            prev_end = 0
            for scheduled_op in pipe:
                if scheduled_op.start_cycle > prev_end:
                    free_times.append((prev_end, scheduled_op.start_cycle))
                prev_end = scheduled_op.end_cycle
            free_times.append((prev_end, float("inf")))
        # find the earliest free time slot that can fit the op
        for beg, end in free_times:
            if end - max(beg, est) >= self.get_cost(op, pipe):
                return int(max(beg, est))
        return est
