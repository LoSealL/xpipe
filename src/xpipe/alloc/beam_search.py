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
from typing import TypedDict

from ..memory import Buffer
from .greedy import GreedyAllocator
from .intervals import merge_intervals


class State(TypedDict):
    """Beam searching state"""

    active: list[tuple[float, int, int]]  # (end, addr, size)
    free: list[tuple[int, int]]  # (start, end)
    peak: int
    addrs: list[int]


class BeamSearchAllocator(GreedyAllocator):
    r"""A slower, higher-quality allocator based on beam search.

    It explores multiple allocation choices at each buffer start event and keeps
    the top-k candidate states, which usually yields lower peak memory than
    single-path greedy heuristics.

    Args:
        beam_width (int): Number of candidate states kept after each step.
        candidate_limit (int): Max placement choices expanded per state/step.
    """

    def __init__(
        self,
        beam_width: int = 64,
        candidate_limit: int = 12,
    ):
        self._beam_width = max(1, int(beam_width))
        self._candidate_limit = max(1, int(candidate_limit))

    @staticmethod
    def _score_state(peak: int, free: list[tuple[int, int]]):
        total_free = sum(end - start for start, end in free)
        largest_hole = max((end - start for start, end in free), default=0)
        frag = total_free - largest_hole
        return (peak, frag, len(free))

    def _candidate_addrs(
        self,
        free: list[tuple[int, int]],
        peak: int,
        size: int,
    ) -> list[int]:
        choices: list[int] = []
        for seg_start, seg_end in free:
            seg_size = seg_end - seg_start
            if seg_size < size:
                continue
            head = seg_start
            tail = seg_end - size
            choices.append(head)
            if tail != head:
                choices.append(tail)
        choices.append(peak)
        dedup = []
        seen = set()
        for addr in choices:
            if addr in seen:
                continue
            seen.add(addr)
            dedup.append(addr)
            if len(dedup) >= self._candidate_limit:
                break
        return dedup

    @staticmethod
    def _place_into_free(
        free: list[tuple[int, int]],
        addr: int,
        size: int,
    ) -> list[tuple[int, int]]:
        end_addr = addr + size
        placed = False
        new_free: list[tuple[int, int]] = []
        for seg_start, seg_end in free:
            if addr < seg_start or end_addr > seg_end or placed:
                new_free.append((seg_start, seg_end))
                continue
            if seg_start < addr:
                new_free.append((seg_start, addr))
            if end_addr < seg_end:
                new_free.append((end_addr, seg_end))
            placed = True
        return merge_intervals(new_free) if placed else free

    def _alloc_single_loc(self, intervals: list[tuple[float, float, Buffer, int]]):
        greedy_peak = super()._greedy_alloc(intervals)
        n = len(intervals)
        init_addrs = [-1] * n
        states: list[State] = [
            {"active": [], "free": [], "peak": 0, "addrs": init_addrs}
        ]

        for idx, (start, end, _, size) in enumerate(intervals):
            expanded = []
            for state in states:
                active = list(state["active"])
                free = list(state["free"])
                while active and active[0][0] <= start:
                    _, release_addr, release_size = active.pop(0)
                    free.append((release_addr, release_addr + release_size))
                free = merge_intervals(free)

                addrs = self._candidate_addrs(free, state["peak"], size)
                for addr in addrs:
                    new_active = list(active)
                    new_free = self._place_into_free(free, addr, size)
                    new_peak = max(state["peak"], addr + size)
                    insort(new_active, (end, addr, size))
                    new_addrs = list(state["addrs"])
                    new_addrs[idx] = addr
                    if new_peak > greedy_peak:
                        continue
                    expanded.append(
                        {
                            "active": new_active,
                            "free": new_free,
                            "peak": new_peak,
                            "addrs": new_addrs,
                        }
                    )

            expanded.sort(key=lambda s: self._score_state(s["peak"], s["free"]))
            next_states = []
            seen = set()
            for state in expanded:
                signature = (
                    state["peak"],
                    tuple(state["free"]),
                    tuple((e, a, sz) for e, a, sz in state["active"]),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                next_states.append(state)
                if len(next_states) >= self._beam_width:
                    break
            states = next_states
            if not states:
                break

        if not states:
            return greedy_peak

        best = min(states, key=lambda s: self._score_state(s["peak"], s["free"]))
        if best["peak"] < greedy_peak:
            for (_, _, buf, _), addr in zip(intervals, best["addrs"]):
                buf.addr = addr
            return best["peak"]
        return greedy_peak
