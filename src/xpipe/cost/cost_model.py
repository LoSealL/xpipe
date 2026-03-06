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

from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Literal, Optional

from ..op import BaseOp
from ..system import Pipeline


class CostModel(metaclass=ABCMeta):
    r"""A base cost model for estimating operation costs."""

    @abstractmethod
    def cost(self, op: BaseOp, pipe: Optional[Pipeline] = None) -> int:
        """Estimate the cost of an operator.

        Args:
            op (BaseOp): The operator to estimate cost for.
            pipe (Optional[Pipeline]): The pipeline on which the operator will run.

        Returns:
            int: The estimated cost of the operator.
        """

    @lru_cache(maxsize=1024, typed=True)
    def __call__(
        self,
        *ops: BaseOp,
        pipe: Optional[Pipeline] = None,
        reduce: Literal["sum", "mean"] = "sum",
    ) -> float:
        """Estimate the total cost of multiple operators.

        Args:
            *ops (BaseOp): The operators to estimate cost for.
            pipe (Optional[Pipeline]): The pipeline on which the operators will run.
            reduce (Literal["sum", "mean"]): The reduction method to combine costs.

        Returns:
            float: The total estimated cost of the operators.
        """

        total_cost = 0
        for op in ops:
            total_cost += self.cost(op, pipe)
        if reduce == "mean" and ops:
            return total_cost / len(ops)
        return total_cost
