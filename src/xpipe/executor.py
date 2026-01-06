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

from .ir.graph import OpGraph
from .op import BaseOp


class BaseExecutor(metaclass=ABCMeta):
    """A simulator for how operator should be executed.

    Users must implement `execute` method to assign start and end time to the
    op in executing.
    """

    def __init__(self, graph: OpGraph):
        self.graph = graph
        self._ts: float = -1

    @abstractmethod
    def execute(self, op: BaseOp) -> float:
        """Basicly the latency of the op.

        Returns:
            int: end time of the operator
        """
        return -1

    def is_ready(self, op: BaseOp) -> bool:
        """Check whether the op is ready for execution."""
        return all([p.end_time <= self._ts for p in self.graph.pred[op]])

    def step(self, timestamp: float):
        """Step into a new timestamp."""
        assert timestamp > self._ts
        self._ts = timestamp


class EasyExecutor(BaseExecutor):
    """A very easy executor that gives a fixed latency for all operators"""

    def __init__(self, graph: OpGraph, interval: float = 2):
        super().__init__(graph)
        self._int = interval

    def execute(self, op: BaseOp) -> float:
        op.start_time = self._ts
        op.end_time = self._ts + self._int
        print(f"{op.name}: {op.start_time} -> {op.end_time}")
        return op.end_time


class CostModelExecutor(BaseExecutor):
    """An executor based on cost model estimation."""

    def execute(self, op: BaseOp) -> float:
        return op.end_time
