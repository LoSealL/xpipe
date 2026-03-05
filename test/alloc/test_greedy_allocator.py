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

from pathlib import Path

from pytest import fixture

from xpipe import CostModelExecutor, MLIRModel, PEFTScheduler, System
from xpipe.alloc import GreedyAllocator
from xpipe.ir.npu_mlir import DmaPipeline, DpuPipeline, DspPipeline, from_mlir


@fixture(params=["srcnn_xpipe.json", "yolov10_sample.json"])
def mlir_graph_file(request):
    return Path(__file__).parent.parent / f"ir/{request.param}"


def test_allocate_greedy(mlir_graph_file):
    # pylint: disable=protected-access
    g, mem = from_mlir(mlir_graph_file, remove_const_dma=False)
    sys = System(
        [
            DmaPipeline("dma", 64, 1000),
            DpuPipeline("dpu"),
            DspPipeline("dsp"),
        ],
        mem,
    )
    sched = PEFTScheduler(MLIRModel())
    sched.schedule(sys, g)
    print(f"Before allocation: DDR={mem['DDR'].capacity} CMX={mem['CMX'].capacity}")
    for _, b in mem["DDR"]._alloc_map.items():
        b.addr = -1
    for _, b in mem["CMX"]._alloc_map.items():
        b.addr = -1
    GreedyAllocator().alloc(g)
    unallocated = []
    max_ddr, max_cmx = 0, 0
    used_ddr, used_cmx = 0, 0
    for _, b in mem["DDR"]._alloc_map.items():
        if b.addr == -1:
            unallocated.append(b)
        else:
            max_ddr = max(max_ddr, b.addr + b.size)
            used_ddr += b.size
    for _, b in mem["CMX"]._alloc_map.items():
        if b.addr == -1:
            unallocated.append(b)
        else:
            max_cmx = max(max_cmx, b.addr + b.size)
            used_cmx += b.size
    assert len(unallocated) == 0, f"Unallocated buffers: {unallocated}"
    print(f"After allocation: DDR={max_ddr}, CMX={max_cmx}")
    print(f"DDR utilization: {used_ddr}, CMX utilization: {used_cmx}")
    overlaps = []
    for op, end_time in sys.step(CostModelExecutor(g)):
        if mem["CMX"].peak < mem["CMX"].size:
            print(f"{op} @ {end_time}: bad alloc")
            ref = [mem["CMX"]._alloc_map[r] for r in mem["CMX"]._ref]
            # check overlap

            for i, b1 in enumerate(ref):
                for j, b2 in enumerate(ref[i + 1 :], start=i + 1):
                    if b1.addr < b2.addr + b2.size and b2.addr < b1.addr + b1.size:
                        overlaps.append((b1, b2))
    print(overlaps)
    assert len(overlaps) == 0, f"Overlapping buffers: {overlaps}"
