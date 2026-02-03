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

from xpipe.ir.utils import PrefixTree


def test_trie_basic():
    items = [
        {"id": 1, "name": "bar_prog_dma"},
        {"id": 2, "name": "/model.10/attn/Softmax"},
        {"id": 3, "name": "/model.10/attn/Softmax?t_Softmax/tile_0"},
        {"id": 4, "name": "/model.0/conv/Conv"},
        {"id": 5, "name": "/model.0/conv/Conv/WithoutBiases"},
    ]
    trie: PrefixTree[dict] = PrefixTree().build(items)
    items = trie.search("/model.0/conv")
    assert len(items) == 2
    assert items[0]["id"] == 4
    assert items[1]["id"] == 5

    trie.insert(
        "/model.10/attn/Matmul", {"id": 6, "name": "/model.10/attn/Matmul"}, order=0
    )
    items = trie.search("/model.10")
    assert len(items) == 3
    assert items[0]["id"] == 6
    assert items[1]["id"] == 2
    assert items[2]["id"] == 3

    assert len(trie.search("/non_exist")) == 0
    assert len(trie.search("/model")) == 0
