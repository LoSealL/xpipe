"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Schedulers to push ordered operators into pipelines
"""

from abc import ABCMeta, abstractmethod

from ..cost import CostModel
from ..ir.graph import OpGraph
from ..op import BaseOp
from ..system import Pipeline, System


class Scheduler(metaclass=ABCMeta):
    r"""Base class for all schedulers.

    A scheduler is responsible for assigning operators to pipelines in a system
    based on certain criteria, such as minimizing execution time or balancing load.

    Args:
        cost_model (CostModel): A cost model to estimate the cost of operators.
    """

    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model
        self.op_costs: dict[tuple[BaseOp, Pipeline], float] = {}

    def get_cost(self, op: BaseOp, pipe: Pipeline) -> float:
        """Get and cache the cost of an operator."""

        if (op, pipe) in self.op_costs:
            return self.op_costs[(op, pipe)]
        cost = self.cost_model(op, pipe=pipe)
        self.op_costs[(op, pipe)] = cost
        return cost

    @abstractmethod
    def schedule(self, system: System, graph: OpGraph):
        """Schedule the operators in the graph onto the pipelines in the system.

        Args:
            system (System): The system containing pipelines and memory slices.
            graph (OpGraph): The operator graph to be scheduled.
        """
        raise NotImplementedError


__all__ = ["Scheduler"]
