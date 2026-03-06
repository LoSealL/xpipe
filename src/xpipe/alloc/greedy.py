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

from bisect import insort

from ..ir.graph import OpGraph
from ..memory import Buffer
from ..op import BaseOp
from .intervals import collect_intervals, merge_intervals


class GreedyAllocator:
    """A greedy allocator that allocates buffers to memory slices in a first-fit
    manner.
    """

    @staticmethod
    def _greedy_alloc(intervals: list[tuple[float, float, Buffer, int]]) -> int:
        """Allocate one memory slice using first-fit and return the peak size."""
        intervals.sort(key=lambda item: (item[0], item[1], -item[3]))
        active: list[tuple[float, int, int]] = []
        free: list[tuple[int, int]] = []
        peak = 0

        for start, end, buf, size in intervals:
            while active and active[0][0] <= start:
                _, release_addr, release_size = active.pop(0)
                free.append((release_addr, release_addr + release_size))
            free = merge_intervals(free)

            addr = None
            for idx, (seg_start, seg_end) in enumerate(free):
                if seg_end - seg_start >= size:
                    addr = seg_start
                    new_start = seg_start + size
                    if new_start < seg_end:
                        free[idx] = (new_start, seg_end)
                    else:
                        free.pop(idx)
                    break

            if addr is None:
                addr = peak
                peak += size

            buf.addr = addr
            insort(active, (end, addr, size))

        return peak

    def _alloc_single_loc(
        self, intervals: list[tuple[float, float, Buffer, int]]
    ) -> int:
        return self._greedy_alloc(intervals)

    def alloc(self, graph: OpGraph[BaseOp]):
        """Allocate buffers for all memory slices and return per-loc peaks.

        Args:
            graph: Scheduled operator graph.

        Returns:
            Peak memory end address for each memory location.
        """
        intervals_by_loc = collect_intervals(graph)
        peaks = {}
        for loc, intervals in intervals_by_loc.items():
            peaks[loc] = self._alloc_single_loc(intervals)
        return peaks
