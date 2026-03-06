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

from collections import defaultdict
from typing import NotRequired, TypedDict

from ..ir.graph import OpGraph
from ..memory import Buffer
from ..op import BaseOp


class Interval(TypedDict):
    """Buffer lifetime info used to build allocation intervals.

    Attributes:
        buf: Buffer object for this lifetime record.
        loc: Memory location name (for example, ``CMX`` or ``DDR``).
        size: Max observed size of the buffer in bytes.
        prod_starts: Start cycles of producer ops.
        prod_ends: End cycles of producer ops.
        cons_starts: Start cycles of consumer ops.
        cons_ends: End cycles of consumer ops.
    """

    buf: NotRequired[Buffer]
    loc: str
    size: int
    prod_starts: list[float]
    prod_ends: list[float]
    cons_starts: list[float]
    cons_ends: list[float]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent address ranges.

    Args:
        intervals: Address ranges in ``(start, end)`` form.

    Returns:
        A sorted, merged list of non-overlapping ranges.
    """
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


def collect_intervals(
    graph: OpGraph[BaseOp],
) -> dict[str, list[tuple[float, float, Buffer, int]]]:
    """Build per-location lifetime intervals from a scheduled op graph.

    Args:
        graph: Scheduled operator graph with valid ``start_cycle`` and
            ``end_cycle`` on each op.

    Returns:
        Mapping from memory location name to interval records in
        ``(start, end, buffer, size)`` form.
    """
    by_buf: dict[tuple[str, int], Interval] = defaultdict(
        lambda: Interval(
            loc="",
            size=0,
            prod_starts=[],
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
            info["prod_starts"].append(float(op.start_cycle))
            info["prod_ends"].append(float(op.end_cycle))

        for buf in op.inputs:
            key = (buf.loc, buf.tag)
            info = by_buf[key]
            info["buf"] = buf
            info["loc"] = buf.loc
            info["size"] = max(info["size"], int(buf.size))
            info["cons_starts"].append(float(op.start_cycle))
            info["cons_ends"].append(float(op.end_cycle))

    intervals_by_loc: dict[str, list[tuple[float, float, Buffer, int]]] = defaultdict(
        list
    )
    for info in by_buf.values():
        prod_starts = info["prod_starts"]
        prod_ends = info["prod_ends"]
        cons_starts = info["cons_starts"]
        cons_ends = info["cons_ends"]
        size = int(info["size"])
        assert "buf" in info
        buf = info["buf"]

        if prod_starts:
            start = min(prod_starts)
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
