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
from itertools import product
from math import prod
from types import SimpleNamespace
from typing import Literal

from ..memory import MemorySlice, MemorySystem
from ..op import BaseOp, Buffer
from ..system import Pipeline
from .graph import OpGraph


class DPUOp(BaseOp):
    """A DPU operation in nbperf IR."""

    def mac(self) -> int:
        """Get the total MACs of this DPU operation."""

        if self.op_type == "CONV":
            return self._conv_mac()
        if self.op_type in ("MAXPOOL", "AVGPOOL"):
            return self._pool_mac()
        if self.op_type == "ADD":
            return self._elewise_mac()
        return 0

    def read_latency(self) -> int:
        """Estimate the read latency of this DPU operation in cycles."""

        return 0

    def _conv_mac(self) -> int:
        """Calculate MACs for convolution operation."""

        data = self["params"]["inputs"][0]
        act = self["params"]["outputs"][0]
        weight = self["params"]["inputs"][1]
        data_shape = {o: data["pitch"][i] for i, o in enumerate(list(data["order"]))}
        act_shape = {o: act["pitch"][i] for i, o in enumerate(list(act["order"]))}
        dilations = self["params"].get("dilations", [1, 1])
        kernel_size = self["params"].get("kernel_shape", [1, 1])
        in_channels = data_shape["C"]
        out_channels = act_shape["C"]
        out_h, out_w = act_shape["H"], act_shape["W"]
        macs = (
            in_channels
            * out_channels
            * out_h
            * out_w
            * prod([(k - 1) * d + 1 for k, d in zip(kernel_size, dilations)])
        )
        batch = data_shape["N"]
        return macs * batch

    def _pool_mac(self) -> int:
        """Calculate MACs for pooling operation."""

        data = self["params"]["inputs"][0]
        act = self["params"]["outputs"][0]
        data_shape = {o: data["pitch"][i] for i, o in enumerate(list(data["order"]))}
        act_shape = {o: act["pitch"][i] for i, o in enumerate(list(act["order"]))}
        kernel_size = self["params"].get("kernel_size", [2, 2])

        batch = data_shape["N"]
        channels = data_shape["C"]
        out_h, out_w = act_shape["H"], act_shape["W"]
        return batch * channels * prod(kernel_size) * out_h * out_w

    def _elewise_mac(self) -> int:
        """Calculate MACs for element-wise operation."""

        data_shape: dict[str, int] = {}
        for data in self["params"]["inputs"]:
            curr_shape = {
                o: data["pitch"][i] for i, o in enumerate(list(data["order"]))
            }
            data_shape = {
                k: max(data_shape.get(k, 0), v) for k, v in curr_shape.items()
            }
        return prod(data_shape.values())

    @property
    def op_type(self) -> str:
        return self["params"]["layer_type"].upper()


class DMAOp(BaseOp):
    """A DMA operation in nbperf IR."""

    @property
    def params(self):
        params = self["params"]
        return SimpleNamespace(
            direction=params["direction"],
            size=params["size"],
            sparsity=params["sparsity"],
            type=params["type"],
        )

    @property
    def direction(self) -> str:
        return self.params.direction

    @property
    def size(self) -> int:
        return self.params.size

    @property
    def sparsity(self) -> float:
        return self.params.sparsity

    @property
    def type(self) -> str:
        return self.params.type


class SHAVEOp(BaseOp):
    """A SW operation in nbperf IR."""


class DmaPipeline(Pipeline[DMAOp]):
    """A pipeline consisting of DMA operations.

    Args:
        name (str): The name of the pipeline.
        bandwidth (int): The transaction byte per cycle of the DMA pipeline.
        zero_overhead (int): The fixed overhead time in cycles for each DMA operation.
    """

    def __init__(self, name: str = "dma", bandwidth: int = 1, zero_overhead: int = 0):
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
        mac (int): The number of MAC operations per cycle.
    """

    def __init__(self, name: str = "dpu", mac: int = 4096):
        super().__init__(name)
        self.mac = mac


class DspPipeline(Pipeline[SHAVEOp]):
    """A pipeline consisting of SHAVE operations.

    Args:
        name (str): The name of the pipeline.
    """

    def __init__(self, name: str = "dsp"):
        super().__init__(name)


def from_nbperf(nb_graph: str | os.PathLike) -> tuple[OpGraph, MemorySystem]:
    """Load an OpGraph from nbperf compiler output IR file."""

    with open(nb_graph, encoding="utf-8") as f:
        nbir: dict = json.load(f)
    tasks: list[BaseOp] = []
    slices: list[MemorySlice] = []
    cmx_clusters = nbir["graph"]["device_meta"]["cmx_clusters"]
    for i in range(cmx_clusters):
        slices.append(MemorySlice(name=f"cmx{i}"))
    mem_slices = MemorySystem(slices)
    for node in nbir.get("nodes", []):
        node_engine: Literal["DPU", "DMA", "DSP"] = node["engine"]
        node_id: int = node["id"]
        node_name: str = node["name"]
        try:
            for task in node.get("tasks", []):
                task_id: int = task["id"]
                inputs: list[Buffer] = []
                outputs: list[Buffer] = []
                params: dict = task.get("params", {})
                params["inputs"] = []
                params["outputs"] = []
                params["layer_type"] = node["layer_type"]
                if node_engine == "DPU":
                    for i in task["inputs"]:
                        cmx = i["locale_index"][0]
                        mem = mem_slices[cmx]
                        tag = int(f"{i['addr']}{cmx:02d}")
                        inputs.append(mem.placeholder(tag, i["addr"], i["pitch_size"]))
                        params["inputs"].append(i)
                    for i in task["outputs"]:
                        cmx = i["locale_index"][0]
                        mem = mem_slices[cmx]
                        tag = int(f"{i['addr']}{cmx:02d}")
                        outputs.append(mem.placeholder(tag, i["addr"], i["pitch_size"]))
                        params["outputs"].append(i)
                    for weight in task.get("weight", []):
                        cmx = weight["locale_index"][0]
                        mem = mem_slices[cmx]
                        tag = int(f"{weight['addr']}{cmx:02d}")
                        buf = mem.placeholder(tag, weight["addr"], weight["pitch_size"])
                        inputs.append(buf)
                        params["inputs"].append(weight)
                    name = f"{node_name}?id{node_id}:{task_id}"
                    op = DPUOp(
                        name=str(task_id),
                        inputs=inputs,
                        outputs=outputs,
                        friendly_name=name,
                        split_strategy=task.get("split_strategy"),
                        stencil=task.get("stencil"),
                        offset=task.get("offset", []),
                        params=params,
                    )
                    tasks.append(op)
                elif node_engine == "DMA":
                    name = f"{node_name}?id{node_id}:{task_id}"
                    params = task["params"]
                    desc = task["dma_descriptor"]
                    assert len(desc) == 1
                    desc = desc[0]
                    if params["direction"].upper() == "DDR2CMX":
                        for cmx in desc["dst_locale_index"]:
                            mem = mem_slices[cmx]
                            tag = int(f"{desc['DST_ADDR']}{cmx:02d}")
                            outputs.append(
                                mem.placeholder(tag, desc["DST_ADDR"], params["size"])
                            )
                    elif params["direction"].upper() == "CMX2DDR":
                        for cmx in desc["src_locale_index"]:
                            mem = mem_slices[cmx]
                            tag = int(f"{desc['SRC_ADDR']}{cmx:02d}")
                            inputs.append(
                                mem.placeholder(tag, desc["SRC_ADDR"], params["size"])
                            )
                    op = DMAOp(
                        name=str(task_id),
                        inputs=inputs,
                        outputs=outputs,
                        friendly_name=name,
                        params=params,
                        **desc,
                    )
                    tasks.append(op)
                elif node_engine == "DSP":
                    for i in task["inputs"]:
                        cmx = i["locale_index"][0]
                        mem = mem_slices[cmx]
                        tag = int(f"{i['addr']}{cmx:02d}")
                        inputs.append(mem.placeholder(tag, i["addr"], i["pitch_size"]))
                        params["inputs"].append(i)
                    for i in task["outputs"]:
                        cmx = i["locale_index"][0]
                        mem = mem_slices[cmx]
                        tag = int(f"{i['addr']}{cmx:02d}")
                        outputs.append(mem.placeholder(tag, i["addr"], i["pitch_size"]))
                        params["outputs"].append(i)
                    name = f"{node_name}?id{node_id}:{task_id}"
                    op = SHAVEOp(
                        name=str(task_id),
                        inputs=inputs,
                        outputs=outputs,
                        friendly_name=name,
                        nce_id=task.get("nce_id"),
                        params=params,
                    )
                    tasks.append(op)
        except Exception as ex:
            raise RuntimeError(
                f"Error when processing node {node_name} id {node_id}"
            ) from ex

    graph: OpGraph[BaseOp] = OpGraph()
    graph.add_nodes_from(tasks)
    assert len(graph) == len(tasks) == nbir["total_tasks"]

    for barrier in nbir["graph"]["barriers"]:
        bid: int = barrier["id"]
        consumer: list[int] = barrier["consumer"]
        producer: list[int] = barrier["producer"]
        for u, v in product(producer, consumer):
            graph.add_edge(graph[str(u)], graph[str(v)], barrier_id=bid)

    return graph, mem_slices
