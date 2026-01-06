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

import random
from typing import Optional

from ..op import BaseOp
from ..system import Pipeline
from .cost_model import CostModel


class DummyCostModel(CostModel):
    """A dummy cost model only for testing purposes.

    Each operation takes 1 unit time.
    """

    def __init__(self, time_unit: int = 1):
        self.unit = time_unit

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> int:
        """Get the cost of an operator.

        Args:
            op (BaseOp): The operator to get the cost for.

        Returns:
            Number: The cost of the operator.
        """
        return self.unit


class RandomCostModel(CostModel):
    """A cost model that randomly generates cost time."""

    def __init__(self, low: int = 1, high: int = 10):
        self.low = low
        self.high = high

    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> int:
        """Get the cost of an operator.

        Args:
            op (BaseOp): The operator to get the cost for.

        Returns:
            Number: The cost of the operator.
        """
        return random.randint(self.low, self.high)
