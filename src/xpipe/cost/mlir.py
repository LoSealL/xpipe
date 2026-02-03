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

from typing import Optional

from ..ir.npu_mlir import DMAOp, DmaPipeline, DPUOp, DpuPipeline, SHAVEOp
from ..op import BaseOp
from ..system import Pipeline
from .cost_model import CostModel


class MLIRModel(CostModel):
    """Mathematical cost model based on static analysis of MLIR.

    DMA cost is modeled as:

    ..math::

        cost = scale * size / bandwidth

    DPU cost is modeled as:

    ..math::

        cost = scale * op["cost"]  # cost from VPUCostModel

    DSP cost is modeled as a constant.

    Args:
        dma_scaling (float): scaling factor to DMA cost.
        dpu_scaling (float): scaling factor to DPU cost.
    """

    def __init__(self, dma_scaling: float = 10, dpu_scaling: float = 10) -> None:
        self.dma_scaling = dma_scaling
        self.dpu_scaling = dpu_scaling

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> float:
        if isinstance(op, DMAOp):
            assert isinstance(pipe, DmaPipeline)
            size = op.inputs[0].size
            scale = self.dma_scaling
            return (size / pipe.bandwidth + pipe.overhead) * scale * 1e6
        elif isinstance(op, DPUOp):
            assert isinstance(pipe, DpuPipeline)
            # DPU cost model: cost = flops / throughput
            scale = self.dpu_scaling
            return op["cost"] * scale * 1e-3  # us
        elif isinstance(op, SHAVEOp):
            # SHAVE cost model: cost = cycles / frequency
            return 10
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")
