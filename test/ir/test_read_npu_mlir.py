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

from xpipe import CostModelExecutor, MLIRModel, PEFTScheduler, System
from xpipe.ir.npu_mlir import DmaPipeline, DpuPipeline, DspPipeline, from_mlir


def test_from_mlir():
    mlir_path = Path(__file__).parent / "yolov10_sample.json"
    graph, memories = from_mlir(mlir_path, remove_const_dma=False)

    num_inputs = len([m for m in graph if graph.in_degree(m) == 0])
    num_outputs = len([m for m in graph if graph.out_degree(m) == 0])
    assert num_inputs == 198
    assert num_outputs == 14
    assert len(memories) == 2  # CMX + DDR


def test_schedule_mlir():
    g, mem = from_mlir(Path(__file__).parent / "srcnn_xpipe.json")
    sys = System(
        [
            DmaPipeline("dma0", 1e8 * 128),  # 128 GB/s
            DpuPipeline("dpu0", 2e8 * 4096),  # 4096 MACs @2GHz
            DspPipeline("dsp0"),
        ],
        mem,
    )
    scheduler = PEFTScheduler(MLIRModel())
    scheduler.schedule(sys, g)
    assert sys.run(CostModelExecutor(g)) > 0
