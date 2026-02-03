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
from xpipe.cost import DummyCostModel, MLIRModel, NBPerfMathModel
from xpipe.ir import nbperf, npu_mlir


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a scheduling test with configurable inputs.",
    )
    ir = parser.add_mutually_exclusive_group(required=True)
    ir.add_argument("--xpipe", "-x", help="Path to MLIR XPIPE .json file")
    ir.add_argument("--nbperf", "-nb", help="Path to NBPerf .json file")
    parser.add_argument(
        "--allocator", "-a", choices=["greedy", "vpurt"], default="greedy"
    )
    parser.add_argument(
        "--scheduler", "-s", choices=["heft", "peft", "robin"], default="peft"
    )
    parser.add_argument(
        "--dma-bandwidth",
        "-bw",
        type=float,
        default=64.0,
        help="DMA bandwidth in GB/s",
    )
    parser.add_argument(
        "--dma-zero-overhead",
        type=float,
        default=1.0,
        help="DMA overhead in useconds when transferring zero bytes",
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
    if args.xpipe:
        graph, mem = npu_mlir.from_mlir(args.xpipe)
        dma_pipeline = npu_mlir.DmaPipeline
        dpu_pipeline = npu_mlir.DpuPipeline
        dsp_pipeline = npu_mlir.DspPipeline
    elif args.nbperf:
        graph, mem = nbperf.from_nbperf(args.nbperf)
        dma_pipeline = nbperf.DmaPipeline
        dpu_pipeline = nbperf.DpuPipeline
        dsp_pipeline = nbperf.DspPipeline
    else:
        raise RuntimeError

    dma_bw = args.dma_bandwidth * 1e9  # GB/s to B/s
    dma_overhead = args.dma_zero_overhead * 1e-6  # us to s
    system = System(
        [
            dma_pipeline("dma", bandwidth=dma_bw, zero_overhead=dma_overhead),
            dpu_pipeline("dpu"),
            dsp_pipeline("dsp"),
        ],
        mem,
    )
    if args.dummy_cost_model:
        cost_model = DummyCostModel(1)
    elif args.xpipe:
        cost_model = MLIRModel()
    elif args.nbperf:
        cost_model = NBPerfMathModel()
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
    else:
        raise NotImplementedError(f"Allocator {args.allocator} not implemented")
    endtime = system.run(CostModelExecutor(graph))
    print(f"Total execution time: {endtime:.6f} us")
    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    system.dump(args.output_trace.with_suffix(".json"))

    if args.save_onnx:
        onnx_graph = graph.to_onnx()
        onnx.save_model(onnx_graph, args.output_trace.with_suffix(".onnx"))


if __name__ == "__main__":
    main()
