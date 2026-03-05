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

from xpipe import MemorySlice, OpGraph, RoundRobinScheduler, System
from xpipe.cost import CostModel
from xpipe.op import BaseOp
from xpipe.system import Pipeline


class CpuOp(BaseOp):
    pass


class GpuOp(BaseOp):
    pass


class CpuPipeline(Pipeline[CpuOp]):
    pass


class GpuPipeline(Pipeline[GpuOp]):
    pass


class _FixedCostModel(CostModel):
    def cost(self, op: BaseOp, pipe: Pipeline | None = None) -> int:
        return 10


def test_robin_scheduler_topological_compatible_timing():
    mem = MemorySlice("ddr")
    graph = OpGraph()

    op_a = CpuOp("A", [mem.placeholder(0, 0, 1)], [mem.placeholder(1, 1, 1)])
    op_b = GpuOp("B", [mem.placeholder(1, 1, 1)], [mem.placeholder(2, 2, 1)])
    op_c = CpuOp("C", [mem.placeholder(1, 1, 1)], [mem.placeholder(3, 3, 1)])
    op_d = GpuOp(
        "D",
        [mem.placeholder(2, 2, 1), mem.placeholder(3, 3, 1)],
        [mem.placeholder(4, 4, 1)],
    )

    graph.add_nodes_from([op_a, op_b, op_c, op_d])
    graph.add_edge(op_a, op_b, comm=2)
    graph.add_edge(op_a, op_c, comm=3)
    graph.add_edge(op_b, op_d, comm=4)
    graph.add_edge(op_c, op_d, comm=1)

    cpu0 = CpuPipeline("cpu0")
    cpu1 = CpuPipeline("cpu1")
    gpu0 = GpuPipeline("gpu0")
    system = System([cpu0, cpu1, gpu0], [mem])

    scheduler = RoundRobinScheduler(_FixedCostModel())
    scheduler.schedule(system, graph)

    # Round-robin pointer advances globally and skips incompatible pipelines.
    assert list(cpu0) == [op_a, op_c]
    assert list(cpu1) == []
    assert list(gpu0) == [op_b, op_d]

    # Start/end cycles honor both dependency readiness (with communication)
    # and selected pipeline availability.
    assert op_a.start_cycle == 0
    assert op_a.end_cycle == 10
    assert op_b.start_cycle == 12
    assert op_b.end_cycle == 22
    assert op_c.start_cycle == 10
    assert op_c.end_cycle == 20
    assert op_d.start_cycle == 22
    assert op_d.end_cycle == 32
