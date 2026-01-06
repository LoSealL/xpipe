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

from typing import Any, Generic, Iterable, TypeVar

import networkx as nx

from ..op import BaseOp

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

    def __getitem__(  # pyright: ignore
        self,
        n: str,
    ) -> T:
        for node in self.nodes:
            if node.name == n:
                return node
        raise KeyError(f"No node named {n} in graph.")
