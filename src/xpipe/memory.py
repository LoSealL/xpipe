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

import weakref


class Buffer:
    """A buffer represents a placeholder on a specific memory slice.

    Args:
        tag (int): unique id to identify the buffer (memref), same tag refers to a
            same buffer regardless of the python object id.
        addr (int): virtual address of the buffer that will be allocated on.
        size (int): size in bytes for the buffer.
        memory_slice (MemorySlice): the memory slice this buffer belongs to.
            See `:class:MemorySlice`.
    """

    def __init__(self, tag: int, addr: int, size: int, memory_slice: "MemorySlice"):
        self._tag = tag
        self._addr = addr
        self._size = size
        self._slice = weakref.proxy(memory_slice)

    @property
    def addr(self) -> int:
        return self._addr

    @property
    def size(self) -> int:
        return self._size

    def consume(self):
        """Mark that this buffer has been consumed.

        This method will decrease the reference on the slice.
        """
        self._slice.free(self._tag)

    def produce(self):
        """Mark that this buffer will be used.

        This method will increase the reference on the slice.
        """

        self._slice.malloc(self._tag)


class MemorySlice:
    r"""A virtual memory represents a memory slice on the system (SRAM or DRAM).

    Args:
        name (str): a unique name to identify the memory slice.
        size (int): maximum capacity of the slice in bytes.
    """

    def __init__(self, name: str, size: int, *, strict: bool = False) -> None:
        self._name = name
        self._size = size
        self._alloc_map: dict[int, Buffer] = {}
        self._ref: dict[int, int] = {}
        self._strict = strict

    def placeholder(self, mem_id: int, addr: int, size: int) -> Buffer:
        """Get or create a buffer that will occupy a space on this slice."""

        if mem_id not in self._alloc_map:
            self._alloc_map[mem_id] = Buffer(mem_id, addr, size, self)
        return self._alloc_map[mem_id]

    def malloc(self, mem_id: int):
        """Allocate spaces for a placehold buffer, and increase reference
        count for it."""

        if mem_id not in self._alloc_map:
            raise RuntimeError(f"Memory {mem_id} not allocated on slice {self.name}.")
        if mem_id not in self._ref:
            self._ref[mem_id] = 1
        else:
            self._ref[mem_id] += 1

    def free(self, mem_id: int):
        """Decrease reference count on the buffer."""

        if mem_id in self._ref and self._ref[mem_id] > 0:
            self._ref[mem_id] -= 1
        elif self._strict:
            raise RuntimeError(f"Freeing unallocated memory {mem_id}.")

    @property
    def name(self) -> str:
        """Return name of the memory slice."""

        return self._name

    @property
    def capacity(self) -> int:
        """Get size in bytes of all active memory buffers."""

        used = 0
        for mem_id, ref in self._ref.items():
            if ref > 0:
                used += self._alloc_map[mem_id].size
        return used

    def __repr__(self) -> str:
        return f"{self.name}: {self.capacity}/{self._size}"
