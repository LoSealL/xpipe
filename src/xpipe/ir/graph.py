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

from collections.abc import Iterator
from typing import Any, Generic, Iterable, TypeVar

import networkx as nx
import onnx

from ..op import BaseOp
from ..version import version

T = TypeVar("T", bound=BaseOp)


class OpGraph(nx.DiGraph, Generic[T]):
    """A DAG for ready-to-execute computing graph."""

    def __init__(self):
        super().__init__()

    def add_edge(self, u_of_edge: T, v_of_edge: T, **attr: Any) -> None:
        super().add_edge(u_of_edge, v_of_edge, **attr)
        v_of_edge.deps.add(u_of_edge)

    def add_edges_from(
        self,
        ebunch_to_add: Iterable[tuple[T, T] | tuple[T, T, dict[str, Any]]],
        **attr: Any,
    ) -> None:
        super().add_edges_from(ebunch_to_add, **attr)
        for u, v, *_ in ebunch_to_add:
            v.deps.add(u)

    def remove_edges_from(
        self, ebunch: Iterable[tuple | tuple[Any, Any, dict[str, Any]]]
    ) -> None:
        super().remove_edges_from(ebunch)
        for u, v, *_ in ebunch:
            v.deps.discard(u)

    def remove_edge(self, u: Any, v: Any) -> None:
        super().remove_edge(u, v)
        v.deps.discard(u)

    def remove_node(self, n: Any) -> None:
        for downstream in self.successors(n):
            downstream.deps.discard(n)
        super().remove_node(n)

    def remove_nodes_from(self, nodes: Iterable) -> None:
        for n in nodes:
            self.remove_node(n)

    def successors(self, n: Any) -> Iterator[T]:
        yield from super().successors(n)

    def predecessors(self, n: Any) -> Iterator[T]:
        yield from super().predecessors(n)

    def __getitem__(  # type: ignore
        self,
        n: str,
    ) -> T:
        for node in self.nodes:
            if node.name == n:
                return node
        raise KeyError(f"No node named {n} in graph.")

    def __iter__(self) -> Iterator[T]:
        for node in self.nodes:
            yield node

    def __contains__(self, n: object) -> bool:
        if isinstance(n, str):
            try:
                _ = self[n]
                return True
            except KeyError:
                return False
        else:
            return super().__contains__(n)

    def to_onnx(self) -> onnx.ModelProto:
        """Convert the graph to ONNX format.

        Returns:
            onnx.ModelProto: The ONNX representation of the graph.
        """
        onnx_nodes = []
        for node in nx.topological_sort(self):
            assert isinstance(node, BaseOp)
            onnx_nodes.append(node.to_onnx())
        graph = onnx.helper.make_graph(onnx_nodes, self.name, [], [], value_info=[])
        return onnx.helper.make_model(
            graph=graph,
            producer_name="xpipe",
            producer_version=version,
            ir_version=onnx.IR_VERSION,
            opset_imports=[onnx.helper.make_operatorsetid("", 21)],
        )
