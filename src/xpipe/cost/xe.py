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

from ..ir.xe import XeOp, CpuOp, GpuPipeline, CpuPipeline
from ..op import BaseOp
from ..system import Pipeline
from .cost_model import CostModel


class XeModel(CostModel):
    """Mathematical cost model based on static analysis of MLIR.

    DMA cost is modeled as:

    ..math::

        cost = scale * size / bandwidth

    DPU cost is modeled as:

    ..math::

        cost = scale * op["cost"]  # cost from VPUCostModel

    DSP cost is modeled as a constant.

    Args:
        gpu_scaling (float): scaling factor to GPU cost.
        cpu_scaling (float): scaling factor to CPU cost.
    """

    def __init__(self, gpu_scaling: float = 10, cpu_scaling: float = 10) -> None:
        self.gpu_scaling = gpu_scaling
        self.cpu_scaling = cpu_scaling

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> float:
        if isinstance(op, XeOp):
            assert isinstance(pipe, GpuPipeline)
            scale = self.gpu_scaling
            return op["cost"] * scale * 1e-3  # us
        elif isinstance(op, CpuOp):
            assert isinstance(pipe, CpuPipeline)
            scale = self.cpu_scaling
            return op["cost"] * scale * 1e-3  # us
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")
