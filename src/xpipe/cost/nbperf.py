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

from ..ir.nbperf import DMAOp, DPUOp, DmaPipeline, DpuPipeline, SHAVEOp
from ..op import BaseOp
from ..system import Pipeline
from .cost_model import CostModel


class NBPerfMathModel(CostModel):
    """Mathematical cost model based on static analysis of nbperf IR.

    Note:

        # modeling function y=k*x+b for DMA copy compute cycle
        # y-> compute cycle
        # x-> data size in bytes
        modeling_function = {
            "vpu5": (1.861569e-2, 169.610585),
            "vpu6": (2.110191e-2, 162.010217),
            "vpu7": (2.020481e-2, 153.551327),
        }
    """

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> float:
        if isinstance(op, DMAOp):
            assert isinstance(pipe, DmaPipeline)
            # DMA cost model: cost = data_size / bandwidth + offset
            offset = 0
            if op.direction in ("DDR2CMX",):
                offset += 0  # simulate DDR read latency
            return (op.size / pipe.bandwidth + pipe.overhead) * 1e6 + offset  # unit: us
        elif isinstance(op, DPUOp):
            assert isinstance(pipe, DpuPipeline)
            # DPU cost model: cost = flops / throughput
            return (op.mac() / pipe.mac + op.read_latency()) * 1e6  # unit: us
        elif isinstance(op, SHAVEOp):
            # SHAVE cost model: cost = cycles / frequency
            return 1
        else:
            raise ValueError(f"Unsupported operation type: {type(op)}")
