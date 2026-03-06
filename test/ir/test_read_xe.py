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
from pathlib import Path

from xpipe.ir.xe import from_xe_graph


def test_from_xe_graph():
    xe_path = Path(__file__).parent / "srcnn_xe.json"
    graph, memories = from_xe_graph(xe_path)

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3

    num_inputs = len([node for node in graph if graph.in_degree(node) == 0])
    num_outputs = len([node for node in graph if graph.out_degree(node) == 0])
    assert num_inputs == 1
    assert num_outputs == 1

    assert len(memories) == 1

    conv0 = graph["convolution:Conv_0/WithoutBiases_0_split_0"]
    assert conv0["entry"] == "convolution_gpu_bfyx_os_iyx_osv32_5596476170970556257_0_0"
    assert len(conv0.inputs) == 3
    assert len(conv0.outputs) == 1


def test_regen_buffer_tags_by_dependency_chain(tmp_path: Path):
    xe_data = {
        "inputs": [],
        "outputs": [],
        "constants": [],
        "variants": [
            {"id": 1, "size": 16},
            {"id": 2, "size": 16},
            {"id": 3, "size": 16},
        ],
        "kernels": [
            {
                "input": False,
                "input_name": "",
                "output": False,
                "id": "a0",
                "unique_id": "a0_0",
                "entry": "e0",
                "domain": None,
                "group_size": [1, 1, 1],
                "local_size": [1, 1, 1],
                "arguments": [
                    {
                        "mem_id": 1,
                        "type": "OUTPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": False,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    }
                ],
                "depends": [],
                "barrier": False,
                "output_layouts": ["bfyx"],
            },
            {
                "input": False,
                "input_name": "",
                "output": False,
                "id": "a1",
                "unique_id": "a1_0",
                "entry": "e1",
                "domain": None,
                "group_size": [1, 1, 1],
                "local_size": [1, 1, 1],
                "arguments": [
                    {
                        "mem_id": 1,
                        "type": "INPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": True,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    },
                    {
                        "mem_id": 2,
                        "type": "OUTPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": False,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    },
                ],
                "depends": ["a0"],
                "barrier": False,
                "output_layouts": ["bfyx"],
            },
            {
                "input": False,
                "input_name": "",
                "output": False,
                "id": "b0",
                "unique_id": "b0_0",
                "entry": "e2",
                "domain": None,
                "group_size": [1, 1, 1],
                "local_size": [1, 1, 1],
                "arguments": [
                    {
                        "mem_id": 1,
                        "type": "OUTPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": False,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    }
                ],
                "depends": [],
                "barrier": False,
                "output_layouts": ["bfyx"],
            },
            {
                "input": False,
                "input_name": "",
                "output": False,
                "id": "b1",
                "unique_id": "b1_0",
                "entry": "e3",
                "domain": None,
                "group_size": [1, 1, 1],
                "local_size": [1, 1, 1],
                "arguments": [
                    {
                        "mem_id": 1,
                        "type": "INPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": True,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    },
                    {
                        "mem_id": 3,
                        "type": "OUTPUT",
                        "dtype": "f16",
                        "shape": [1],
                        "is_input": False,
                        "layout": "bfyx",
                        "block": {"block_sizes": [], "block_idx": [], "strides": []},
                        "memory_type": "gpu_usm",
                        "allocation_type": "usm_device",
                        "alias_offset": 0,
                    },
                ],
                "depends": ["b0"],
                "barrier": False,
                "output_layouts": ["bfyx"],
            },
        ],
    }
    xe_path = tmp_path / "reuse_split.json"
    xe_path.write_text(json.dumps(xe_data), encoding="utf-8")

    graph, _ = from_xe_graph(xe_path)

    a0 = graph["a0_0"]
    a1 = graph["a1_0"]
    b0 = graph["b0_0"]
    b1 = graph["b1_0"]

    assert a0.outputs[0].tag == a1.inputs[0].tag
    assert b0.outputs[0].tag == b1.inputs[0].tag
    assert a1.inputs[0].tag != b1.inputs[0].tag
