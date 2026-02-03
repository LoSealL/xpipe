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

from typing import Generic, Optional, TypeVar

from ..op import BaseOp
from .graph import OpGraph

T = TypeVar("T", bound=dict)


class TrieNode(Generic[T]):
    """Trie node class"""

    def __init__(self):
        """Initialize node"""
        self.children: dict[str, TrieNode[T]] = {}  # Children nodes dictionary
        self.items: list[tuple[T, int]] = []  # Store items passing through this node


class PrefixTree(Generic[T]):
    """Prefix tree class for efficient storage and querying of path-like names"""

    def __init__(self):
        """Initialize prefix tree"""
        self.root: TrieNode[T] = TrieNode[T]()
        self._ord: int = 0  # Order index for items insertion

    def insert(self, name: str, item: T, *, order: Optional[int] = None) -> None:
        """
        Insert item into prefix tree

        Args:
            name (str): Item name in path format
            item (T): Item dictionary
            order (int, optional): Original order index of the item
        """
        node = self.root
        parts = [
            part for part in name.split("/") if part
        ]  # Split path and filter empty strings

        for part in parts:
            if part not in node.children:
                node.children[part] = TrieNode[T]()
            node = node.children[part]

        if order is None:
            order = self._ord
            self._ord += 1
        node.items.append((item, order))

    def search(self, prefix: str) -> list[T]:
        """
        Search for matching items in prefix tree

        Args:
            prefix (str): Search prefix

        Returns:
            list[T]: List of matching items
        """
        node = self.root
        parts = [
            part for part in prefix.split("/") if part
        ]  # Split path and filter empty strings

        for part in parts:
            if part not in node.children:
                return []  # No matching items
            node = node.children[part]

        items = self._collect_items(node)
        return [items[i] for i in sorted(items)]

    def _collect_items(self, node: TrieNode[T]) -> dict[int, T]:
        """
        Recursively collect all items from node and its children

        Args:
            node (TrieNode[T]): Starting node

        Returns:
            dict[int, T]: Collected items dictionary
        """
        items: dict[int, T] = {}
        items.update({order: item for item, order in node.items})

        for child in node.children.values():
            for order, item in self._collect_items(child).items():
                items[order] = item
        return items

    def build(self, items: list[T], *, key: str = "name") -> "PrefixTree[T]":
        """
        Build prefix tree

        Args:
            items (list[T]): Items list
            key (str): Key name for item names

        Returns:
            PrefixTree[T]: Built prefix tree
        """
        for index, item in enumerate(items):
            self.insert(str(item[key]), item, order=index)

        return self


def to_dot(graph: OpGraph[BaseOp]) -> str:
    """Convert an OpGraph to DOT format for visualization.

    Args:
        graph (OpGraph): The OpGraph to convert.

    Returns:
        str: The DOT representation of the graph.
    """

    dotdoc = "digraph G {\n"
    for op in graph:
        dotdoc += f'  "{op.name}" [label="{op.name}\\n{type(op).__name__}"];\n'
        for succ in graph.successors(op):
            dotdoc += f'  "{op.name}" -> "{succ.name}";\n'
    dotdoc += "}\n"
    return dotdoc
