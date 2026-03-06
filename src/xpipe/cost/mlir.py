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
    """

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> int:
        if isinstance(op, DMAOp):
            assert isinstance(pipe, DmaPipeline)
            size = op.inputs[0].size
            return (size + pipe.bandwidth - 1) // pipe.bandwidth + pipe.overhead
        elif isinstance(op, DPUOp):
            assert isinstance(pipe, DpuPipeline)
            return int(op["cost"])
        elif isinstance(op, SHAVEOp):
            return 10
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")
