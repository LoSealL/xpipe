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

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Optional

from .op import BaseOp


class Phase(Enum):
    r"""Phases of trace events."""

    BEGIN = "B"
    END = "E"
    COMPLETE = "X"
    INSTANT = "i"
    COUNTER = "C"
    ASYNC_BEGIN = "b"
    ASYNC_END = "e"
    ASYNC_INSTANT = "n"
    FLOW_BEGIN = "s"
    FLOW_STEP = "t"
    FLOW_END = "f"
    OBJ_CREATE = "N"
    OBJ_SNAPSHOT = "O"
    OBJ_DESTROY = "D"
    METADATA = "M"
    MARK = "R"


@dataclass
class MetadataEvent:
    r"""A metadata event for trace files."""

    name: Literal[
        "process_name",
        "process_label",
        "process_sort_index",
        "thread_name",
        "thread_sort_index",
    ]
    pid: int = 0
    tid: int = 0
    args: dict = field(default_factory=dict)
    ph: str = Phase.METADATA.value


@dataclass
class TraceEvent:
    r"""A trace event records the execution of an operator in a pipeline."""

    name: str
    cat: str
    ts: float
    pid: int
    tid: int
    ph: str
    args: dict = field(default_factory=dict)
    cname: Optional[
        Literal["good", "bad", "terrible", "black", "grey", "white", "yellow", "olive"]
    ] = None


@dataclass
class CompleteEvent(TraceEvent):
    ph: str = Phase.COMPLETE.value
    dur: float = field(default=1)


@dataclass
class CounterEvent(TraceEvent):
    ph: str = Phase.COUNTER.value
    args: dict = field(default_factory=dict)


@dataclass
class FlowStartEvent(TraceEvent):
    ph: str = Phase.FLOW_BEGIN.value
    id: int = field(default=0)


@dataclass
class FlowFinishEvent(TraceEvent):
    ph: str = Phase.FLOW_END.value
    id: int = field(default=0)
    bp: str = field(default="e")  # set bind point as enclosing


class CatapultRecorder:
    r"""Record events in catapult trace format."""

    def __init__(self) -> None:
        self._events: dict[str, TraceEvent] = {}
        self._proc: dict[str, MetadataEvent] = {}
        self._thread: dict[str, MetadataEvent] = {}
        self._op2cat: dict[BaseOp, str] = {}

    def _may_create_cat(self, cat: str):
        if cat not in self._proc:
            self._proc[cat] = MetadataEvent(
                name="process_name",
                pid=len(self._proc),
                tid=len(self._thread),
                args={"name": cat},
            )
            self._thread[cat] = MetadataEvent(
                name="thread_name",
                pid=len(self._proc) - 1,
                tid=len(self._thread),
                args={"name": cat + "_thread"},
            )

    def record(self, cat: str, op: BaseOp):
        """Record the begin and end event (which combines a complete event)
        on the op."""

        self._may_create_cat(cat)
        if "friendly_name" in op:
            name = op["friendly_name"]
        else:
            name = op.name
        self._events[op.name] = CompleteEvent(
            name=name,
            cat=op.type,
            ts=op.start_time,
            dur=op.end_time - op.start_time,
            pid=self._proc[cat].pid,
            tid=self._thread[cat].tid,
            args={k: str(v) for k, v in op.attr.items()},
        )
        self._op2cat[op] = cat

    def record_memory_delta(self, cat: str, ts: float, size: int):
        """Record the snapshot of current memory size.

        Note:
            There is a visualizing bug in chrome://tracing that decline of size
            can not be shown.
        """

        if size == 0:
            return
        self._may_create_cat(cat)
        name = f"{cat}:{ts}"
        self._events[name] = CounterEvent(
            name=cat,
            cat=cat,
            ts=ts,
            pid=self._proc[cat].pid,
            tid=self._thread[cat].tid,
            args=dict(mem=size),
        )

    def record_dependency(self, src: BaseOp, dst: BaseOp):
        """Record a flow event from src to dst."""

        csrc = self._op2cat[src]
        cdst = self._op2cat[dst]
        if csrc not in self._proc or cdst not in self._proc:
            raise RuntimeError(f"{src} or {dst} has not been recorded yet")
        name = f"{src.name}->{dst.name}"
        hash_id = hash((src, dst))
        self._events[name + ":b"] = FlowStartEvent(
            name=name,
            cat="flow",
            ts=(src.start_time + src.end_time) // 2,
            pid=self._proc[csrc].pid,
            tid=self._thread[csrc].tid,
            id=hash_id,
        )
        self._events[name + ":e"] = FlowFinishEvent(
            name=name,
            cat="flow",
            ts=(dst.start_time + dst.end_time) // 2,
            pid=self._proc[cdst].pid,
            tid=self._thread[cdst].tid,
            id=hash_id,
        )

    def to_json(self) -> dict:
        """Serialize the records to json format."""

        events = [asdict(ev) for ev in self._events.values()]
        events.extend(asdict(ev) for ev in self._proc.values())
        events.extend(asdict(ev) for ev in self._thread.values())
        return {"traceEvents": events, "displayTimeUnit": "ns"}
