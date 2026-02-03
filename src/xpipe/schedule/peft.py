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

from collections import defaultdict

import networkx as nx
import numpy as np
from loguru import logger

from ..ir.graph import OpGraph
from ..op import BaseOp
from ..system import Pipeline, System
from .heft import HEFTScheduler


class PEFTScheduler(HEFTScheduler):
    """Predictive Earliest Finish Time (PEFT) scheduler.

    Reference::

    H. Arabnejad and J. G. Barbosa
    List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table
    IEEE Transactions on Parallel and Distributed Systems,
    vol. 25, no. 3, pp. 682-694, Mar. 2014, doi: 10.1109/TPDS.2013.57.
    """

    def schedule(self, system: System, graph: OpGraph):
        octable = self._gen_oct(system, graph)
        self.rank_oct(octable, graph)
        ops: list[BaseOp] = sorted(graph.nodes, key=lambda op: op["rank"], reverse=True)
        for op in ops:
            best_pipe: Pipeline | None = None
            best_eft = float("inf")
            min_oeft = float("inf")
            for pipe in system.pipelines:
                if not pipe.is_compatible(op):
                    continue
                est = self.est(op, pipe, graph)
                eft = est + self.get_cost(op, pipe)
                eft_optimistic = eft + octable[op][pipe]
                if eft_optimistic < min_oeft:
                    min_oeft = eft_optimistic
                    best_pipe = pipe
                    best_eft = eft
            assert best_pipe is not None
            op.start_time = best_eft - self.get_cost(op, best_pipe)
            op.end_time = best_eft
            logger.debug(f"schedule {op} on {best_pipe}")
            best_pipe.push(op)

    def _gen_oct(
        self, system: System, graph: OpGraph
    ) -> dict[BaseOp, dict[Pipeline, float]]:
        """Generate the Optimistic Cost Table (OCT).

        Args:
            system (System): The target system.
            graph (OpGraph): The operator graph.

        Returns:
            dict[BaseOp, dict[Pipeline, float]]: The OCT mapping each operator to a
                dict of optimistic costs on each pipeline.
        """

        oct_table: dict[BaseOp, dict[Pipeline, float]] = defaultdict(dict)
        for op in nx.topological_sort(graph.reverse(copy=False)):
            for pipe in system.pipelines:
                if not pipe.is_compatible(op):
                    continue
                max_oct = 0
                for tj in graph.successors(op):
                    comm = graph.get_edge_data(op, tj, {}).get("comm", 0)
                    min_costs = []
                    for pw, oc in oct_table[tj].items():
                        w_tj = self.get_cost(tj, pw)
                        min_costs.append(oc + w_tj + (comm if pw != pipe else 0))
                    max_oct = max(max_oct, float(np.min(min_costs)))
                oct_table[op][pipe] = max_oct
        return oct_table

    def rank_oct(self, octable, graph: OpGraph):
        """Compute the rank based on optimistic cost table."""

        for op in nx.topological_sort(graph.reverse(copy=False)):
            rank_oct = np.mean(list(octable[op].values())).item()
            op["rank"] = rank_oct
