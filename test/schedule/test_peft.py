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

from xpipe import BasicPipeline, MemorySlice, OpGraph, PEFTScheduler, System
from xpipe.cost import CostModel
from xpipe.op import BaseOp
from xpipe.system import Pipeline


def _build_graph():
    mem = MemorySlice("ddr")
    p1 = BasicPipeline("P1")
    p2 = BasicPipeline("P2")
    p3 = BasicPipeline("P3")
    graph = OpGraph()
    graph.add_nodes_from(
        [
            BaseOp("T1", [mem.placeholder(0, 0, 1)], [mem.placeholder(1, 1, 1)]),
            BaseOp("T2", [mem.placeholder(1, 1, 1)], [mem.placeholder(2, 2, 1)]),
            BaseOp("T3", [mem.placeholder(1, 1, 1)], [mem.placeholder(3, 3, 1)]),
            BaseOp("T4", [mem.placeholder(1, 1, 1)], [mem.placeholder(4, 4, 1)]),
            BaseOp("T5", [mem.placeholder(1, 1, 1)], [mem.placeholder(5, 5, 1)]),
            BaseOp("T6", [mem.placeholder(1, 1, 1)], [mem.placeholder(6, 6, 1)]),
            BaseOp("T7", [mem.placeholder(3, 3, 1)], [mem.placeholder(7, 7, 1)]),
            BaseOp(
                "T8",
                [
                    mem.placeholder(2, 2, 1),
                    mem.placeholder(4, 4, 1),
                    mem.placeholder(6, 6, 1),
                ],
                [mem.placeholder(8, 8, 1)],
            ),
            BaseOp(
                "T9",
                [
                    mem.placeholder(2, 2, 1),
                    mem.placeholder(4, 4, 1),
                    mem.placeholder(5, 5, 1),
                ],
                [mem.placeholder(9, 9, 1)],
            ),
            BaseOp(
                "T10",
                [
                    mem.placeholder(7, 7, 1),
                    mem.placeholder(8, 8, 1),
                    mem.placeholder(9, 9, 1),
                ],
                [mem.placeholder(10, 10, 1)],
            ),
        ]
    )
    graph.add_edge(graph["T1"], graph["T2"], comm=17)
    graph.add_edge(graph["T1"], graph["T3"], comm=31)
    graph.add_edge(graph["T1"], graph["T4"], comm=29)
    graph.add_edge(graph["T1"], graph["T5"], comm=13)
    graph.add_edge(graph["T1"], graph["T6"], comm=7)
    graph.add_edge(graph["T2"], graph["T8"], comm=3)
    graph.add_edge(graph["T2"], graph["T9"], comm=30)
    graph.add_edge(graph["T3"], graph["T7"], comm=16)
    graph.add_edge(graph["T4"], graph["T8"], comm=11)
    graph.add_edge(graph["T4"], graph["T9"], comm=7)
    graph.add_edge(graph["T5"], graph["T9"], comm=57)
    graph.add_edge(graph["T6"], graph["T8"], comm=5)
    graph.add_edge(graph["T7"], graph["T10"], comm=9)
    graph.add_edge(graph["T8"], graph["T10"], comm=42)
    graph.add_edge(graph["T9"], graph["T10"], comm=7)

    class _FixedCostModel(CostModel):
        _lut = {
            "T1": [22, 21, 36],
            "T2": [22, 18, 18],
            "T3": [32, 27, 43],
            "T4": [7, 10, 4],
            "T5": [29, 27, 35],
            "T6": [26, 17, 24],
            "T7": [14, 25, 30],
            "T8": [29, 23, 36],
            "T9": [15, 21, 8],
            "T10": [13, 16, 33],
        }

        def cost(self, op: BaseOp, pipe: Pipeline | None = None) -> float:
            assert pipe is not None
            pipe_id = int(pipe.name[1:]) - 1
            return self._lut[op.name][pipe_id]

    system = System([p1, p2, p3], [mem])
    return graph, system, _FixedCostModel()


def test_peft_scheduler():
    graph, system, cost_model = _build_graph()
    scheduler = PEFTScheduler(cost_model)
    scheduler.schedule(system, graph)

    assert round(graph["T1"]["rank"]) == 73
    assert round(graph["T2"]["rank"]) == 41
    assert round(graph["T3"]["rank"]) == 37
    assert round(graph["T4"]["rank"]) == 44
    assert round(graph["T5"]["rank"]) == 31
    assert round(graph["T6"]["rank"]) == 42
    assert round(graph["T7"]["rank"]) == 17
    assert round(graph["T8"]["rank"]) == 21
    assert round(graph["T9"]["rank"]) == 16
    assert round(graph["T10"]["rank"]) == 0
    assert len(system["P1"]) == 5
    assert len(system["P2"]) == 3
    assert len(system["P3"]) == 2
    assert graph["T10"].end_time == 122
