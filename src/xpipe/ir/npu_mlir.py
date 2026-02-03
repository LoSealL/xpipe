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

import math
import os
from collections import defaultdict

import json5
import networkx as nx
from loguru import logger

from ..memory import MemorySlice, MemorySystem
from ..op import BaseOp, Buffer
from ..system import Pipeline
from .graph import OpGraph
from .utils import PrefixTree


class DPUOp(BaseOp):
    """A DPU operation in MLIR."""


class DMAOp(BaseOp):
    """A DMA operation in MLIR."""


class ConstDMAOp(DMAOp):
    """A special DMA operation which transfers const data from DDR to CMX."""


class SpillDMAOp(DMAOp):
    """A special DMA operation which spills data from CMX to DDR (or back)."""


class SHAVEOp(BaseOp):
    """A SW operation in MLIR."""


class DmaPipeline(Pipeline[DMAOp]):
    """A pipeline consisting of DMA operations.

    Args:
        name (str): The name of the pipeline.
        bandwidth (float): The bandwidth of the DMA pipeline in bytes/s.
        zero_overhead (float): The fixed overhead time in s for each DMA operation.
    """

    def __init__(
        self, name: str = "dma", bandwidth: float = 1, zero_overhead: float = 0
    ):
        super().__init__(name)
        self.bandwidth = bandwidth
        self.overhead = zero_overhead

    @property
    def is_dma(self) -> bool:
        return True


class DpuPipeline(Pipeline[DPUOp]):
    """A pipeline consisting of DPU operations.

    Args:
        name (str): The name of the pipeline.
        mac (float): The number of MAC operations per second.
    """

    def __init__(self, name: str = "dpu", mac: float = 1):
        super().__init__(name)
        self.mac = mac


class DspPipeline(Pipeline[SHAVEOp]):
    """A pipeline consisting of SHAVE operations.

    Args:
        name (str): The name of the pipeline.
    """

    def __init__(self, name: str = "dsp"):
        super().__init__(name)


def json_to_digraph(json_data: dict):
    """
    Convert JSON data to a networkx.DiGraph

    Args:
        json_data (dict): The JSON data representing the graph.

    Returns:
        networkx.DiGraph: A directed graph representation of the JSON data

    Raises:
        FileNotFoundError: If the provided file path does not exist
        json.JSONDecodeError: If the provided file or object is not valid JSON
        TypeError: If input_data is not a string or JSON object
        ValueError: If any node is missing required fields (inputs or outputs)
    """

    graph = nx.DiGraph()
    # output memory id to producing node ids (allow many nodes to a same memory)
    output_memory_map: dict[int, set[int]] = defaultdict(set)
    for item in json_data:
        node_id = item["id"]
        assert node_id not in graph
        # Add all attributes to the node
        graph.add_node(node_id, **item)

    # Create edges based on memory dependencies
    for item in json_data:  # json from mlir is SSA ordered
        node_id = item["id"]
        for input_mem in set(i["id"] for i in item["inputs"]):
            if input_mem == 0:
                input_mem = node_id  # assign DDR id (0) as the node id
            producing_nodes = output_memory_map.get(input_mem, set())
            for prod_node_id in producing_nodes:
                if prod_node_id == node_id:
                    # inplace op
                    continue
                # bind memory ID to edge data
                graph.add_edge(prod_node_id, node_id, mem_id=input_mem)
        for i in set(i["id"] for i in item["outputs"]):
            if i == 0:
                i = node_id
            output_memory_map[i].add(node_id)

    if not nx.is_directed_acyclic_graph(graph):
        for cycle in nx.simple_cycles(graph):
            print("Cycle detected:", " -> ".join(str(n) for n in cycle))
        raise RuntimeError("The graph contains cycles.")

    return graph


def query_num_clusters(mlir_dag: nx.DiGraph) -> int:
    """Query the number of clusters used in MLIR."""

    max_clusters = 1
    for i in mlir_dag:
        for o in mlir_dag.nodes[i]["outputs"]:
            cmx = len(o.get("shapes", []))
            max_clusters = max(max_clusters, cmx)
    return max_clusters


def assign_ir_to_graph(mlir_dag: nx.DiGraph, node_ir: list[dict]):
    """Mapping nodes in nodeGraph to nodes in nodeIR.

    Args:
        graph (nx.DiGraph): The MLIR graph.
        node_ir (list[dict]): The list of nodes in nodeIR.

    Returns:
        dict[int, list[dict]]: A mapping from nodeGraph ID to all related IR nodes.
    """

    trie: PrefixTree[dict] = PrefixTree().build(node_ir)
    ir_mapping: dict[int, int] = {}
    for i in mlir_dag:
        node = mlir_dag.nodes[i]
        for ir_node in trie.search(node["name"]):
            if ir_node["id"] not in ir_mapping:
                ir_mapping[ir_node["id"]] = i
                continue
            else:
                logger.error(f"Duplicate IR node name found: {ir_node['id']}")
                raise ValueError
    ir_nodes = {n["id"]: n for n in node_ir}
    node_mapping: dict[int, list[dict]] = defaultdict(list)
    for ir_id, graph_id in ir_mapping.items():
        node_mapping[graph_id].append(ir_nodes[ir_id])
    return node_mapping


def _append_io(
    args: list[dict],
    io_memo: list[Buffer],
    mem: MemorySystem,
):
    for _, op_io in enumerate(args):
        mem_id = op_io["id"]
        kind = op_io["loc"]
        size = sum(math.prod(shape) for shape in op_io["shapes"])
        if op_io["dtype"] in ("f16", "i16", "ui16", "si16"):
            size *= 2
        elif op_io["dtype"] in ("ui32", "si32", "i32"):
            size *= 4
        elif op_io["dtype"] in ("ui64", "si64", "i64"):
            size *= 8
        addr = -1
        if kind == "CMX":
            io_memo.append(mem["CMX"].placeholder(mem_id, addr, size))
        elif kind == "DDR":
            io_memo.append(mem["DDR"].placeholder(mem_id, addr, size))


def _is_spill(mlir_dag: nx.DiGraph, graph_node: int) -> bool:
    if not mlir_dag.pred[graph_node]:
        return False
    for pred in mlir_dag.predecessors(graph_node):
        if mlir_dag.nodes[pred]["type"] == "DMA":
            return False
    for succ in mlir_dag.successors(graph_node):
        # All successors are DMA from DDR to CMX
        if mlir_dag.nodes[succ]["type"] != "DMA":
            return False
        succ_in = mlir_dag.nodes[succ]["inputs"]
        succ_out = mlir_dag.nodes[succ]["outputs"]
        if not succ_in or succ_in[0]["loc"] != "DDR":
            return False
        if not succ_out or succ_out[0]["loc"] != "CMX":
            return False
    return True


def from_mlir(
    mlir_graph: str | os.PathLike,
    *,
    remove_const_dma: bool = True,
    from_vpurt_mlir: bool = False,
) -> tuple[OpGraph[BaseOp], MemorySystem]:
    r"""Load an MLIR graph from a file or string.

    Args:
        mlir_graph (str | os.PathLike): The path to the MLIR file or the MLIR string.

    Returns:
        OpGraph: The loaded MLIR graph.
        MemorySystem: The list of memory slices used in the graph.
    """
    with open(mlir_graph, encoding="utf-8") as f:
        mlir_db = json5.load(f, allow_duplicate_keys=False)
    mlir_dag = json_to_digraph(mlir_db["nodeGraph"])
    n_cmx = query_num_clusters(mlir_dag)
    logger.info(f"Parsed {len(mlir_dag)} nodes from MLIR, using {n_cmx} clusters.")

    mem = MemorySystem(
        dict(CMX=MemorySlice("CMX", strict=False), DDR=MemorySlice("DDR", strict=False))
    )
    graph: OpGraph[BaseOp] = OpGraph()

    for _, graph_node in enumerate(nx.topological_sort(mlir_dag)):
        # assemble ir nodes
        op_type = mlir_dag.nodes[graph_node]["type"]
        op_name = mlir_dag.nodes[graph_node]["name"]
        op_inputs: list[Buffer] = []
        op_outputs: list[Buffer] = []
        try:
            _append_io(mlir_dag.nodes[graph_node]["inputs"], op_inputs, mem)
            _append_io(mlir_dag.nodes[graph_node]["outputs"], op_outputs, mem)
        except IndexError:
            logger.error(f"Node {graph_node} encountered appending IO error")
            raise
        if op_type == "DMA":
            attrs: dict[str, str | int | float] = {}
            attrs["size"] = sum(b.size for b in op_inputs)
            attrs["dir"] = f"{op_inputs[0].loc}->{op_outputs[0].loc}"
            if _is_spill(mlir_dag, graph_node):
                op = SpillDMAOp(op_name, op_inputs, op_outputs, **attrs)
            elif mlir_dag.pred[graph_node]:
                op = DMAOp(op_name, op_inputs, op_outputs, **attrs)
            else:
                op = ConstDMAOp(op_name, op_inputs, op_outputs, **attrs)
        elif op_type == "DPU":
            cost = int(mlir_dag.nodes[graph_node]["attrs"]["cost"])
            sizes = [b.size for b in op_inputs + op_outputs]
            op = DPUOp(op_name, op_inputs, op_outputs, cost=cost, sizes=sizes)
        elif op_type == "SW":
            op = SHAVEOp(op_name, op_inputs, op_outputs)
        else:
            raise RuntimeError(f"Unsupported op type: {op_type}")
        assert op not in graph
        graph.add_node(op)
        for pred in mlir_dag.predecessors(graph_node):
            pred_name = mlir_dag.nodes[pred]["name"]
            graph.add_edge(graph[pred_name], op)

    graph = post_process(graph, remove_const_dma=remove_const_dma)
    return graph, mem


def post_process(graph: OpGraph, **kwargs) -> OpGraph:
    """Apply post processing on the graph."""

    h = graph.copy()

    # 1. remove spill DMA nodes
    def remove_spill_dma():
        for node in graph:
            if isinstance(node, SpillDMAOp):
                for u in graph.successors(node):
                    for uu in graph.successors(u):
                        h.add_edge(node, uu)
                    if u in h:
                        h.remove_node(u)
                preds = list(h.predecessors(node))
                for u in h.successors(node):
                    h.add_edges_from((pred, u) for pred in preds)
                h.remove_node(node)

    # 2. (opt) remove const DMA
    def remove_const_dma():
        for node in graph:
            if isinstance(node, ConstDMAOp):
                h.remove_node(node)

    if kwargs.get("remove_spill_dma"):
        remove_spill_dma()
    if kwargs.get("remove_const_dma"):
        remove_const_dma()
    return h
