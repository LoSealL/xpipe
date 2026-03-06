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

from ..memory import Buffer
from .greedy import GreedyAllocator
from .intervals import merge_intervals


class BestFitAllocator(GreedyAllocator):
    r"""A linear-scan allocator with best-fit gap reuse.

    Compared with first-fit greedy allocation, this allocator tends to reduce
    fragmentation and peak address by selecting the smallest free segment that
    satisfies each allocation request.
    """

    @staticmethod
    def _alloc_best_fit(
        intervals: list[tuple[float, float, Buffer, int]],
    ) -> tuple[int, list[int]]:
        intervals.sort(key=lambda item: (item[0], item[1], -item[3]))
        active: list[tuple[float, int, int]] = []
        free: list[tuple[int, int]] = []
        peak = 0
        addrs: list[int] = []

        for start, end, _, size in intervals:
            while active and active[0][0] <= start:
                _, release_addr, release_size = active.pop(0)
                free.append((release_addr, release_addr + release_size))
            free = merge_intervals(free)

            best_idx = -1
            best_waste = None
            for idx, (seg_start, seg_end) in enumerate(free):
                seg_size = seg_end - seg_start
                if seg_size < size:
                    continue
                waste = seg_size - size
                if best_waste is None or waste < best_waste:
                    best_waste = waste
                    best_idx = idx
                    if waste == 0:
                        break

            if best_idx >= 0:
                seg_start, seg_end = free[best_idx]
                addr = seg_start
                new_start = seg_start + size
                if new_start < seg_end:
                    free[best_idx] = (new_start, seg_end)
                else:
                    free.pop(best_idx)
            else:
                addr = peak
                peak += size

            addrs.append(addr)
            insort(active, (end, addr, size))

        return peak, addrs

    def _alloc_single_loc(self, intervals: list[tuple[float, float, Buffer, int]]):
        greedy_peak = self._greedy_alloc(intervals)
        best_peak, best_addrs = self._alloc_best_fit(list(intervals))

        if best_peak < greedy_peak:
            for (_, _, buf, _), addr in zip(intervals, best_addrs):
                buf.addr = addr
            return best_peak
        return greedy_peak
