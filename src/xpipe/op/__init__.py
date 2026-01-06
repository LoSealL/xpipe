"""
Copyright Wenyi Tang 2026

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Abstract Operators executed in pipelines
"""

from abc import ABCMeta
from collections.abc import Sequence
from typing import Any

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
        self.name = name
        self.attr = attr
        self._beg: float = 0
        self._end: float = float("inf")
        self._inputs: list[Buffer] = list(inputs)
        self._outputs: list[Buffer] = list(outputs)
        self._deps: set[BaseOp] = set()

    @property
    def start_time(self) -> float:
        return self._beg

    @property
    def end_time(self) -> float:
        return self._end

    @start_time.setter
    def start_time(self, ts: float) -> None:
        self._beg = ts

    @end_time.setter
    def end_time(self, ts: float) -> None:
        self._end = ts

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def deps(self):
        return self._deps

    def __getitem__(self, key: Any) -> Any:
        return self.attr[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.attr[key] = value

    def __contains__(self, key: Any) -> bool:
        return key in self.attr

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        attr_str = ", ".join(f"{k}={v}" for k, v in self.attr.items())
        if attr_str:
            attr_str = f"({attr_str})"
        return f"{self.name}[{self.start_time}->{self.end_time}]{attr_str}"
