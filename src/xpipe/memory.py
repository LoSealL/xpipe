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
from collections.abc import Sequence
from typing import overload

from loguru import logger


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
        self._size = size
        self._slice = weakref.proxy(memory_slice)
        self.addr = addr

    @property
    def tag(self) -> int:
        return self._tag

    @property
    def size(self) -> int:
        return self._size

    @property
    def loc(self) -> str:
        return self._slice.name

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
        strict (bool): whether to enforce strict memory allocation rules.
    """

    def __init__(self, name: str, *, strict: bool = True) -> None:
        self._name = name
        self._alloc_map: dict[int, Buffer] = {}
        self._ref: dict[int, int] = {}
        self._strict = strict

    def clear(self):
        """Clear all buffers on this slice."""
        self._alloc_map.clear()
        self._ref.clear()

    def reset(self):
        """Reset the memory slice by clearing all allocated buffers."""
        self._ref.clear()

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
            if self._alloc_map[mem_id].addr < 0:
                # Linear scan for the first available gap
                req_size = self._alloc_map[mem_id].size
                intervals = []
                for mid, count in self._ref.items():
                    if count > 0:
                        b = self._alloc_map[mid]
                        intervals.append((b.addr, b.addr + b.size))
                intervals.sort()

                addr = 0
                for start, end in intervals:
                    if start - addr >= req_size:
                        break
                    addr = max(addr, end)

                self._alloc_map[mem_id].addr = addr
                logger.debug(f"[{self.name}] Alloc {req_size} of {mem_id} at {addr}")
            self._ref[mem_id] = 1
        else:
            self._ref[mem_id] += 1

    def free(self, mem_id: int):
        """Decrease reference count on the buffer."""

        if mem_id in self._ref and self._ref[mem_id] > 0:
            self._ref[mem_id] -= 1
            if self._ref[mem_id] == 0:
                logger.debug(f"[{self.name}] Free {mem_id}")
                del self._ref[mem_id]
        elif self._strict:
            raise RuntimeError(f"Freeing unallocated memory {mem_id}.")

    @property
    def name(self) -> str:
        """Return name of the memory slice."""

        return self._name

    @property
    def peak(self) -> int:
        """Get peak address of the allocated buffers on this slice."""

        max_addr = 0
        for mem_id, ref in self._ref.items():
            mem = self._alloc_map[mem_id]
            if ref > 0:
                max_addr = max(max_addr, mem.addr + mem.size)
        return max_addr

    @property
    def size(self) -> int:
        """Get size in bytes of all active memory buffers."""
        used = 0
        for mem_id, ref in self._ref.items():
            mem = self._alloc_map[mem_id]
            if ref > 0:
                used += mem.size
        return used

    @property
    def capacity(self) -> int:
        """Get total capacity of the memory slice."""

        return sum(mem.size for mem in self._alloc_map.values())

    def __repr__(self) -> str:
        return f"{self.name}: peak={self.peak} usage={self.size} cap={self.capacity}"


class MemorySystem(dict[str, MemorySlice]):
    """A collection of memory slices in the system.

    Args:
        slices (list[MemorySlice]): list of memory slices in the system.
    """

    @overload
    def __init__(self, slices: dict[str, MemorySlice]) -> None: ...

    @overload
    def __init__(self, slices: Sequence[MemorySlice]) -> None: ...

    def __init__(self, slices: dict | Sequence) -> None:
        super().__init__()
        if isinstance(slices, dict):
            for name, mem in slices.items():
                self[name] = mem
        elif isinstance(slices, Sequence):
            for i, s in enumerate(slices):
                self[str(i)] = s
        elif isinstance(slice, MemorySlice):
            self["0"] = slices
        else:
            raise TypeError("slices must be a dict or a list of MemorySlice.")

    def __getitem__(self, key: str | int) -> MemorySlice:
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("MemorySystem index out of range.")
            for i, v in enumerate(self.values()):
                if i == key:
                    return v
            assert False, "Unreachable"
        else:
            return super().__getitem__(key)

    def reset(self):
        """Reset the memory system by clearing all allocated buffers."""
        for mem in self.values():
            mem.reset()
