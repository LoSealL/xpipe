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
from collections import defaultdict
from collections.abc import Generator, Sequence
from heapq import heappop, heappush
from typing import Any, Generic, Optional, TypeVar, get_args, get_origin, overload

from loguru import logger

from .executor import BaseExecutor
from .memory import MemorySlice, MemorySystem
from .op import BaseOp
from .recorder import CatapultRecorder

T = TypeVar("T", bound=BaseOp)


class MetaPipe(type):
    r"""A metaclass for Pipeline to enforce type constraints."""

    def __new__(mcs, name, bases, namespace):
        namespace["__allow__"] = []
        cls = super().__new__(mcs, name, bases, namespace)
        if cls.__orig_bases__:  # type: ignore[attr-defined]
            base = cls.__orig_bases__[0]  # type: ignore[attr-defined]
            if get_origin(base) is not Generic:
                # Check the derived pipeline must assign a valid op constraint
                args = get_args(base)
                if not issubclass(args[0], BaseOp):
                    raise TypeError(
                        f"A pipeline class {name} declared with invalid op type: "
                        f"{name}[{args[0]}], expect a subclass of BaseOp."
                    )
                cls.__allow__.extend(args)  # type: ignore[attr-defined]
            elif name != "Pipeline":
                raise RuntimeError("Cannot instantiate generic Pipeline directly.")
        return cls


class Pipeline(Generic[T], metaclass=MetaPipe):
    r"""A pipeline is a FIFO queue for a single typed operators."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._buck: list[T] = []

    @property
    def allowed(self) -> list[type]:
        """Get the allowed operator types for this pipeline."""

        return getattr(self, "__allow__", [])

    @property
    def is_dma(self) -> bool:
        """Whether this pipeline is for DMA operations."""
        return False

    def push(self, item: T) -> None:
        """Add an operator to the end of the pipeline.

        Args:
            item (T): the operator to be added. The type T must be compatible
                with the pipeline's allowed types.
        """

        for t in self.allowed:
            if not isinstance(item, t):
                raise TypeError(
                    f"Pipeline {self.name} only allows to push operator of class or "
                    f"subclass of {t.__name__}, but got {type(item).__name__}."
                )
        if hasattr(item, "end_cycle") and item.end_cycle != float("inf"):
            assert hasattr(item, "start_cycle")
            # insert item to keep the pipeline ordered by start_cycle
            for i, op in enumerate(self._buck):
                if item.start_cycle < op.start_cycle:
                    self._buck.insert(i, item)
                    return
        self._buck.append(item)

    def is_compatible(self, item: Any) -> bool:
        """Check whether the operator is compatible with the pipeline."""

        for t in self.allowed:
            if isinstance(item, t):
                return True
        return False

    def pop(self) -> T:
        """Remove and return the operator at the front of the pipeline."""

        return self._buck.pop(0)

    def try_pop(self, executor: BaseExecutor) -> Optional[T]:
        """Check and pop the operator at the front of the pipeline if it's ready."""

        if len(self._buck) == 0:
            return None
        elif executor.is_ready(self._buck[0]):
            return self.pop()
        else:
            return None

    def clone(self) -> "Pipeline[T]":
        """Create a deep copy of the pipeline."""

        new_pipe = self.__class__(self.name)
        for i in self:
            new_pipe.push(i)
        return new_pipe

    def __getitem__(self, index: int) -> T:
        return self._buck[index]

    def __len__(self) -> int:
        return len(self._buck)

    def __contains__(self, item: T) -> bool:
        return item in self._buck

    def __iter__(self):
        for item in self._buck:
            yield item

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}({self.__class__.__name__}, {len(self)})"


class System:
    r"""A system is a collection of pipelines that compose the execution environment.

    Args:
        pipelines (Sequence[Pipeline]): all pipelines that presence in the system.
        mem_slices (MemorySystem): all memories that presence in the system.
    """

    @overload
    def __init__(
        self, pipelines: Sequence[Pipeline], mem_slices: MemorySystem
    ) -> None: ...

    @overload
    def __init__(
        self, pipelines: Sequence[Pipeline], mem_slices: Sequence[MemorySlice]
    ) -> None: ...

    def __init__(
        self,
        pipelines: Sequence[Pipeline],
        mem_slices: Sequence[MemorySlice] | MemorySystem,
    ):
        self._typed_pipes = defaultdict(list)
        for pipe in pipelines:
            self._typed_pipes[type(pipe)].append(pipe)
        if not isinstance(mem_slices, MemorySystem):
            mem_slices = MemorySystem(mem_slices)
        self._mem_slices = mem_slices
        self._rec = CatapultRecorder()

    @property
    def pipelines(self) -> Generator[Pipeline[BaseOp], None, None]:
        """Get all pipelines in the system."""
        for pipes in self._typed_pipes.values():
            for p in pipes:
                yield p

    def __getitem__(self, key: str) -> Pipeline:
        """Get a pipeline by its name.

        Example::

            system = System([...], [...])
            p1 = system["P1"]  # get the pipeline named "P1"
        """

        for pipe in self.pipelines:
            if pipe.name == key:
                return pipe
        raise KeyError(f"Pipeline named {key} not found in the system.")

    def run(self, executor: BaseExecutor) -> int:
        """Simulate all operations from pipelines by the given executor,
        and return the total execution time.

        Args:
            executor (BaseExecutor): An executor to simulate the execution.

        Returns:
            int: the total execution time (in microseconds).
        """
        last_end_time = -1
        for _, end_time in self.step(executor):
            last_end_time = max(last_end_time, end_time)
        return last_end_time

    def step(self, executor: BaseExecutor) -> Generator[tuple[BaseOp, int], None, None]:
        """Perform a single step of execution by the given executor, and yield the
        executed operator and its end time."""
        self._mem_slices.reset()
        self._rec.reset()
        pipelines = [p.clone() for p in self.pipelines]
        next_timestamp: list[int] = [0]
        end_times: dict[int, list[BaseOp]] = defaultdict(list)
        while len(next_timestamp) > 0:
            ts = heappop(next_timestamp)
            executor.step(ts)
            op_times: list[tuple[BaseOp, int, int]] = []
            if ts in end_times:
                for end_op in end_times.pop(ts):
                    for inp in end_op.inputs:
                        logger.debug(f"consume {inp.tag} by {end_op.name}")
                        inp.consume()
                    for s in self._mem_slices.values():
                        peak = s.peak
                        size = s.size
                        logger.debug(f"{s.name}: peak={peak} size={size} bytes")
                        self._rec.record_memory_delta(s.name + "_PEAK", ts, peak)
                        self._rec.record_memory_delta(s.name + "_USED", ts, size)
            for pipe in pipelines:
                if op := pipe.try_pop(executor):
                    for outp in op.outputs:
                        logger.debug(f"produce {outp.tag} by {op.name}")
                        outp.produce()
                    for s in self._mem_slices.values():
                        peak = s.peak
                        size = s.size
                        logger.debug(f"{s.name}: peak={peak} size={size} bytes")
                        self._rec.record_memory_delta(s.name + "_PEAK", ts, peak)
                        self._rec.record_memory_delta(s.name + "_USED", ts, size)
                    end_time = executor.execute(op)
                    assert op.start_cycle == ts
                    op_times.append((op, ts, end_time))
                    end_times[end_time].append(op)
                    logger.debug(f"exec {op} on {pipe} from {ts} to {end_time}")
                    self._rec.record(pipe.name, op)
                    for dep in op.deps:
                        self._rec.record_dependency(dep, op)
                    yield op, end_time
            if not op_times:
                # No op was executed at this timestamp. To avoid terminating
                # early when there is an idle gap before the next op starts,
                # advance time to the earliest pending start_cycle, if any.
                next_start: Optional[int] = None
                for pipe in pipelines:
                    # Try common attributes for the internal op queue without
                    # assuming a specific Pipeline API.
                    next_op = pipe[0] if len(pipe) > 0 else None
                    if next_op is None:
                        continue
                    start = next_op.start_cycle
                    if start > ts and (next_start is None or start < next_start):
                        next_start = start
                if next_start is not None:
                    heappush(next_timestamp, next_start)
                continue
            for end_time in set(i[-1] for i in op_times):
                heappush(next_timestamp, end_time)

    def dump(self, filepath: str = "trace.json") -> None:
        """Dump the execution trace recorded by the system to a JSON file."""

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._rec.to_json(), f, indent=4)


class BasicPipeline(Pipeline[BaseOp]):
    r"""A basic pipeline that can accept any operator."""
