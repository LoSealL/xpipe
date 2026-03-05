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

from ..ir.xe import CpuOp, CpuPipeline, GpuPipeline, XeOp
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

    def __init__(self, gpu_freq: float = 2e9, cpu_freq: float = 2e9) -> None:
        self.gpu_freq = gpu_freq / 1e6  # MHz
        self.cpu_freq = cpu_freq / 1e6

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> int:
        if isinstance(op, XeOp):
            assert isinstance(pipe, GpuPipeline)
            return int(op["cost"] * self.gpu_freq)
        elif isinstance(op, CpuOp):
            assert isinstance(pipe, CpuPipeline)
            return int(op["cost"] * self.cpu_freq)
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")
