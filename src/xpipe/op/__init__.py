"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Abstract Operators executed in pipelines
"""

from abc import ABCMeta
from collections.abc import Sequence
from typing import Any

import onnx

from ..memory import Buffer


class BaseOp(metaclass=ABCMeta):
    r"""Base class for all operators executed in pipelines."""

    def __init__(
        self,
        name: str,
        inputs: Sequence[Buffer],
        outputs: Sequence[Buffer],
        **attr: Any,
    ) -> None:
        """Initialize an operator node.

        Args:
            name: Unique operator name.
            inputs: Input buffers consumed by this operator.
            outputs: Output buffers produced by this operator.
            **attr: Additional operator attributes.
        """
        self.name = name
        self.attr = attr
        self._beg: int = 0
        self._end: float = float("inf")
        self._inputs: list[Buffer] = list(inputs)
        self._outputs: list[Buffer] = list(outputs)
        self._deps: set[BaseOp] = set()

    @property
    def start_cycle(self) -> int:
        """Start cycle of this operator in the schedule."""
        return self._beg

    @property
    def end_cycle(self) -> int:
        """End cycle of this operator in the schedule."""
        return int(self._end)

    @start_cycle.setter
    def start_cycle(self, ts: int) -> None:
        """Set the start cycle for this operator."""
        self._beg = ts

    @end_cycle.setter
    def end_cycle(self, ts: int) -> None:
        """Set the end cycle for this operator."""
        self._end = ts

    @property
    def type(self) -> str:
        """Type name of this operator class."""
        return type(self).__name__

    @property
    def inputs(self):
        """Input buffers consumed by this operator."""
        return self._inputs

    @property
    def outputs(self):
        """Output buffers produced by this operator."""
        return self._outputs

    @property
    def deps(self):
        """Dependency operators that must run before this operator."""
        return self._deps

    def __getitem__(self, key: Any) -> Any:
        """Return an attribute value by key."""
        return self.attr[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an attribute value by key."""
        self.attr[key] = value

    def __contains__(self, key: Any) -> bool:
        """Check whether an attribute key exists."""
        return key in self.attr

    def __hash__(self) -> int:
        """Hash by operator name for set and dict usage."""
        return hash(self.name)

    def __repr__(self) -> str:
        """Return a readable summary including schedule range and attributes."""
        attr_str = ", ".join(f"{k}={v}" for k, v in self.attr.items())
        if attr_str:
            attr_str = f"({attr_str})"
        return f"{self.name}[{self.start_cycle}->{self.end_cycle}]{attr_str}"

    def to_onnx(self) -> onnx.NodeProto:
        """Convert the operator to ONNX format.

        Returns:
            onnx.NodeProto: The ONNX representation of the operator.
        """

        def _port_name(buf: Buffer) -> str:
            return f"{buf.loc}{buf.tag}"

        return onnx.helper.make_node(
            op_type=self.__class__.__name__,
            inputs=[_port_name(buf) for buf in self.inputs],
            outputs=[_port_name(buf) for buf in self.outputs],
            name=self.name,
            **self.attr,
        )
