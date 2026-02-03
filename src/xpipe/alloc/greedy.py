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
from collections import defaultdict
from typing import NotRequired, TypedDict

from ..ir.graph import OpGraph
from ..memory import Buffer
from ..op import BaseOp


class Interval(TypedDict):
    buf: NotRequired[Buffer]
    loc: str
    size: int
    prod_ends: list[float]
    cons_starts: list[float]
    cons_ends: list[float]


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class GreedyAllocator:
    """A greedy allocator that allocates buffers to memory slices in a first-fit
    manner.
    """

    @staticmethod
    def _collect_intervals(graph: OpGraph[BaseOp]):
        by_buf: dict[tuple[str, int], Interval] = defaultdict(
            lambda: Interval(
                loc="",
                size=0,
                prod_ends=[],
                cons_starts=[],
                cons_ends=[],
            )
        )

        for op in graph:
            for buf in op.outputs:
                key = (buf.loc, buf.tag)
                info = by_buf[key]
                info["buf"] = buf
                info["loc"] = buf.loc
                info["size"] = max(info["size"], int(buf.size))
                info["prod_ends"].append(float(op.end_time))

            for buf in op.inputs:
                key = (buf.loc, buf.tag)
                info = by_buf[key]
                info["buf"] = buf
                info["loc"] = buf.loc
                info["size"] = max(info["size"], int(buf.size))
                info["cons_starts"].append(float(op.start_time))
                info["cons_ends"].append(float(op.end_time))

        intervals_by_loc: dict[str, list[tuple[float, float, Buffer, int]]] = (
            defaultdict(list)
        )
        for info in by_buf.values():
            prod_ends = info["prod_ends"]
            cons_starts = info["cons_starts"]
            cons_ends = info["cons_ends"]
            size = int(info["size"])
            assert "buf" in info
            buf = info["buf"]

            if prod_ends:
                start = min(prod_ends)
            elif cons_starts:
                start = min(cons_starts)
            else:
                start = 0.0

            end_candidates = []
            if cons_ends:
                end_candidates.append(max(cons_ends))
            if prod_ends:
                end_candidates.append(max(prod_ends))
            end = max(end_candidates) if end_candidates else start
            if end < start:
                end = start

            intervals_by_loc[info["loc"]].append((start, end, buf, size))

        return intervals_by_loc

    @staticmethod
    def _alloc_single_loc(intervals: list[tuple[float, float, Buffer, int]]):
        intervals.sort(key=lambda item: (item[0], item[1], -item[3]))
        active: list[tuple[float, int, int]] = []
        free: list[tuple[int, int]] = []
        peak = 0

        for start, end, buf, size in intervals:
            while active and active[0][0] <= start:
                release_end, release_addr, release_size = active.pop(0)
                _ = release_end
                free.append((release_addr, release_addr + release_size))
                free = _merge_intervals(free)

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

    def alloc(self, graph: OpGraph[BaseOp]):
        intervals_by_loc = self._collect_intervals(graph)
        peaks = {}
        for loc, intervals in intervals_by_loc.items():
            peaks[loc] = self._alloc_single_loc(intervals)
        return peaks
