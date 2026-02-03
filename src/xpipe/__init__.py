r"""XPIPE: A pipeline simulation and visualization system

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

try:
    import os

    os.environ.setdefault("LOGURU_LEVEL", "INFO")
except (ImportError, KeyError):
    pass

from .cost import DummyCostModel, MLIRModel, NBPerfMathModel, RandomCostModel
from .executor import CostModelExecutor, EasyExecutor
from .ir.graph import OpGraph
from .memory import MemorySlice
from .schedule.heft import HEFTScheduler
from .schedule.peft import PEFTScheduler
from .schedule.robin import RoundRobinScheduler
from .system import BasicPipeline, Pipeline, System
from .version import version

__version__ = version

__all__ = [
    "DummyCostModel",
    "MLIRModel",
    "NBPerfMathModel",
    "RandomCostModel",
    "CostModelExecutor",
    "EasyExecutor",
    "OpGraph",
    "MemorySlice",
    "HEFTScheduler",
    "PEFTScheduler",
    "RoundRobinScheduler",
    "BasicPipeline",
    "Pipeline",
    "System",
]
