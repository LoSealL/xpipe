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

import argparse
from pathlib import Path

import onnx

from xpipe import (
    CostModelExecutor,
    HEFTScheduler,
    PEFTScheduler,
    RoundRobinScheduler,
    System,
)
from xpipe.alloc import GreedyAllocator
from xpipe.cost import DummyCostModel, MLIRModel, NBPerfMathModel, XeModel
from xpipe.ir import nbperf, npu_mlir, xe


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a scheduling test with configurable inputs.",
    )
    ir = parser.add_mutually_exclusive_group(required=True)
    ir.add_argument("--xpipe", "-x", help="Path to MLIR XPIPE .json file")
    ir.add_argument("--nbperf", "-nb", help="Path to NBPerf .json file")
    ir.add_argument("--xe", "-xe", help="Path to Xe dumped graph .json file")
    parser.add_argument(
        "--allocator", "-a", choices=["greedy", "vpurt"], default="greedy"
    )
    parser.add_argument(
        "--scheduler", "-s", choices=["heft", "peft", "robin"], default="peft"
    )
    parser.add_argument(
        "--dma-bandwidth",
        "-bw",
        type=int,
        default=64,
        help="DMA bandwidth in B/cycle",
    )
    parser.add_argument(
        "--dma-zero-overhead",
        type=int,
        default=1000,
        help="DMA overhead in cycles when transferring zero bytes",
    )
    parser.add_argument(
        "--dummy-cost-model", action="store_true", help="Use a dummy cost model"
    )
    parser.add_argument(
        "--output-trace",
        "-o",
        required=True,
        type=Path,
        help="Path to output execution trace .json file",
    )
    parser.add_argument(
        "--save-onnx", action="store_true", help="Save the scheduled graph as ONNX"
    )
    parser.add_argument("-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dma_bw = args.dma_bandwidth  # B/cycle
    dma_overhead = args.dma_zero_overhead  # cycles
    if args.xpipe:
        graph, mem = npu_mlir.from_mlir(args.xpipe)
        dma_pipeline = npu_mlir.DmaPipeline(
            "dma", bandwidth=dma_bw, zero_overhead=dma_overhead
        )
        dpu_pipeline = npu_mlir.DpuPipeline("dpu")
        dsp_pipeline = npu_mlir.DspPipeline("dsp")
    elif args.nbperf:
        graph, mem = nbperf.from_nbperf(args.nbperf)
        dma_pipeline = nbperf.DmaPipeline(
            "dma", bandwidth=dma_bw, zero_overhead=dma_overhead
        )
        dpu_pipeline = nbperf.DpuPipeline("dpu", mac=4096)
        dsp_pipeline = nbperf.DspPipeline("dsp")
    elif args.xe:
        graph, mem = xe.from_xe_graph(args.xe)
        dma_pipeline = None
        dpu_pipeline = xe.GpuPipeline("xe")
        dsp_pipeline = xe.CpuPipeline("cpu")
    else:
        raise RuntimeError

    pipelines = []
    if dma_pipeline is not None:
        pipelines.append(dma_pipeline)
    if dpu_pipeline is not None:
        pipelines.append(dpu_pipeline)
    if dsp_pipeline is not None:
        pipelines.append(dsp_pipeline)
    system = System(pipelines, mem)
    if args.dummy_cost_model:
        cost_model = DummyCostModel(1)
    elif args.xpipe:
        cost_model = MLIRModel()
    elif args.nbperf:
        cost_model = NBPerfMathModel()
    elif args.xe:
        cost_model = XeModel()
    else:
        raise RuntimeError

    if args.scheduler == "heft":
        scheduler = HEFTScheduler(cost_model)
    elif args.scheduler == "peft":
        scheduler = PEFTScheduler(cost_model)
    else:
        scheduler = RoundRobinScheduler(cost_model)
    scheduler.schedule(system, graph)

    if args.allocator == "greedy":
        allocator = GreedyAllocator()
        allocator.alloc(graph)
    endtime = system.run(CostModelExecutor(graph))
    print(f"Total execution time: {endtime:.6f} us")
    # system.replay()
    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    system.dump(args.output_trace.with_suffix(".json"))

    if args.save_onnx:
        onnx_graph = graph.to_onnx()
        onnx.save_model(onnx_graph, args.output_trace.with_suffix(".onnx"))


if __name__ == "__main__":
    main()
