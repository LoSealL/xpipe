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

import json
import os
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import networkx as nx

from ..memory import MemorySlice, MemorySystem
from ..op import BaseOp
from ..system import Pipeline
from .graph import OpGraph


class XeOp(BaseOp):
    """A Xe kernel operation loaded from Xe dumped graph."""


class CpuOp(BaseOp):
    """A offloaded CPU operation loaded from Xe dumped graph."""


class GpuPipeline(Pipeline[XeOp]):
    """A pipeline consisting of XeOps loaded from Xe dumped graph.

    Args:
        name (str): The name of the pipeline.
    """

    def __init__(self, name: str = "xe"):
        super().__init__(name)


class CpuPipeline(Pipeline[CpuOp]):
    """A pipeline consisting of CpuOps loaded from Xe dumped graph.

    Args:
        name (str): The name of the pipeline.
    """

    def __init__(self, name: str = "cpu"):
        super().__init__(name)


def _collect_buffer_sizes(xe_graph: dict[str, Any]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for key in ("inputs", "outputs", "constants", "variants"):
        for item in xe_graph.get(key, []):
            mem_id = int(item["id"])
            size = int(item["size"])
            prev = sizes.get(mem_id)
            if prev is None:
                sizes[mem_id] = size
            else:
                sizes[mem_id] = max(prev, size)
    return sizes


def _regen_buffer_tags_by_dependency(graph: OpGraph[BaseOp]) -> MemorySlice:
    """Regenerate buffer tags by dependency-chain relation.

    This pass breaks the original memory-id reuse relation from Xe dump and keeps
    same tag only for buffers that are connected through producer->consumer
    dependencies.
    """

    in_occ: dict[int, list[tuple[BaseOp, int]]] = defaultdict(list)
    out_occ: dict[int, list[tuple[BaseOp, int]]] = defaultdict(list)
    all_occ: dict[int, list[tuple[str, BaseOp, int]]] = defaultdict(list)

    for op in graph:
        for idx, buf in enumerate(op.inputs):
            tag = int(buf.tag)
            in_occ[tag].append((op, idx))
            all_occ[tag].append(("in", op, idx))
        for idx, buf in enumerate(op.outputs):
            tag = int(buf.tag)
            out_occ[tag].append((op, idx))
            all_occ[tag].append(("out", op, idx))

    next_tag = 1000
    remap: dict[tuple[str, Hashable, int], int] = {}
    name_cache: dict[BaseOp, str] = {op: op.name for op in graph}

    for old_tag, occs in all_occ.items():
        adjacency: dict[tuple[str, Hashable, int], set[tuple[str, Hashable, int]]] = {}
        for kind, op, idx in occs:
            key = (kind, name_cache[op], idx)
            adjacency.setdefault(key, set())

        pred_out_idx: dict[BaseOp, list[int]] = defaultdict(list)
        succ_in_idx: dict[BaseOp, list[int]] = defaultdict(list)
        for op, idx in out_occ.get(old_tag, []):
            pred_out_idx[op].append(idx)
        for op, idx in in_occ.get(old_tag, []):
            succ_in_idx[op].append(idx)

        for pred, succ in graph.edges:
            pred_indices = pred_out_idx.get(pred, [])
            succ_indices = succ_in_idx.get(succ, [])
            if not pred_indices or not succ_indices:
                continue
            for p_idx in pred_indices:
                p_key = ("out", name_cache[pred], p_idx)
                for s_idx in succ_indices:
                    s_key = ("in", name_cache[succ], s_idx)
                    adjacency[p_key].add(s_key)
                    adjacency[s_key].add(p_key)

        visited: set[tuple[str, Hashable, int]] = set()
        for node in adjacency:
            if node in visited:
                continue
            stack = [node]
            comp = []
            visited.add(node)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adjacency[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

            for occ_key in comp:
                remap[occ_key] = next_tag
            next_tag += 1

    usm = MemorySlice("USM", strict=False)  # renew buffer
    for op in graph:
        for idx, old_buf in enumerate(op.inputs):
            key = ("in", op.name, idx)
            new_tag = remap[key]

            new_buf = usm.placeholder(
                new_tag,
                -1,
                old_buf.size,
            )
            op.inputs[idx] = new_buf
        for idx, old_buf in enumerate(op.outputs):
            key = ("out", op.name, idx)
            new_tag = remap[key]
            new_buf = usm.placeholder(
                new_tag,
                -1,
                old_buf.size,
            )
            op.outputs[idx] = new_buf
    return usm


def from_xe_graph(
    xe_graph_file: str | os.PathLike,
) -> tuple[OpGraph[BaseOp], MemorySystem]:
    r"""Load OpGraph from XPU Xe dumped graph (.json).

    Args:
        xe_graph_file (str | os.PathLike): The path to the XPU Xe dumped graph (.json).
    """
    with open(xe_graph_file, encoding="utf-8") as f:
        xe_graph = json.load(f)
        if "bin" in xe_graph:
            del xe_graph["bin"]  # remove the binary data to save memory

    graph: OpGraph[BaseOp] = OpGraph()
    buffer_sizes = _collect_buffer_sizes(xe_graph)
    usm = MemorySlice("USM", strict=False)

    ops_by_unique: dict[str, XeOp] = {}
    ops_by_id: dict[str, list[XeOp]] = defaultdict(list)

    for kernel in xe_graph["kernels"]:
        op_name = kernel["unique_id"]
        op_inputs = []
        op_outputs = []
        for arg in kernel.get("arguments", []):
            mem_id = int(arg["mem_id"])
            size = int(buffer_sizes.get(mem_id, 0))
            addr = mem_id + arg.get("alias_offset", 0)
            buf = usm.placeholder(mem_id, addr, size)

            arg_type = str(arg.get("type", "")).upper()
            if "OUTPUT" in arg_type:
                op_outputs.append(buf)
            else:
                op_inputs.append(buf)

        attrs: dict[str, Any] = {
            "entry": kernel.get("entry"),
            "domain": kernel.get("domain"),
            "group_size": kernel.get("group_size", []),
            "local_size": kernel.get("local_size", []),
            "input": bool(kernel.get("input", False)),
            "output": bool(kernel.get("output", False)),
            "input_name": kernel.get("input_name", ""),
            "cost": kernel.get("cost", 1000),
        }
        op = XeOp(op_name, op_inputs, op_outputs, **attrs)
        graph.add_node(op)
        ops_by_unique[op_name] = op
        ops_by_id[kernel["id"]].append(op)

    for kernel in xe_graph["kernels"]:
        curr_name = kernel["unique_id"]
        curr_op = ops_by_unique[curr_name]
        for dep in kernel.get("depends", []):
            for pred in ops_by_id.get(dep, []):
                graph.add_edge(pred, curr_op)

    if not nx.is_directed_acyclic_graph(graph):
        for cycle in nx.simple_cycles(graph):
            names = [node.name for node in cycle]
            print("Cycle detected:", " -> ".join(names))
        raise RuntimeError("The Xe graph contains cycles.")

    usm = _regen_buffer_tags_by_dependency(graph)
    return graph, MemorySystem({"USM": usm})
