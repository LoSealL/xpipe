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
from copy import deepcopy, copy
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
        if hasattr(item, "end_time") and item.end_time != float("inf"):
            assert hasattr(item, "start_time")
            # insert item to keep the pipeline ordered by start_time
            for i, op in enumerate(self._buck):
                if item.start_time < op.start_time:
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

    def run(self, executor: BaseExecutor) -> float:
        """Simulate all operations from pipelines by the given executor,
        and return the total execution time.

        Args:
            executor (BaseExecutor): An executor to simulate the execution.

        Returns:
            float: the total execution time (in microseconds).
        """
        last_end_time = -1
        for _, end_time in self.step(executor):
            last_end_time = max(last_end_time, end_time)
        return last_end_time

    def step(
        self, executor: BaseExecutor
    ) -> Generator[tuple[BaseOp, float], None, None]:
        """Perform a single step of execution by the given executor, and yield the
        executed operator and its end time."""
        self._mem_slices.reset()
        self._rec.reset()
        pipelines = [p.clone() for p in self.pipelines]
        next_timestamp: list[float] = [0]
        while len(next_timestamp) > 0:
            ts = heappop(next_timestamp)
            executor.step(ts)
            end_times: set[float] = set()
            for pipe in pipelines:
                if op := pipe.try_pop(executor):
                    for outp in op.outputs:
                        outp.produce()
                    for s in self._mem_slices.values():
                        peak = s.peak
                        size = s.size
                        logger.debug(f"{s.name}: peak={peak} size={size} bytes")
                        self._rec.record_memory_delta(s.name + "_PEAK", ts, peak)
                        self._rec.record_memory_delta(s.name + "_USED", ts, size)
                    end_time = executor.execute(op)
                    logger.debug(f"exec {op} on {pipe} from {ts:.2f} to {end_time:.2f}")
                    end_times.add(end_time)
                    for inp in op.inputs:
                        inp.consume()
                    for s in self._mem_slices.values():
                        peak = s.peak
                        size = s.size
                        logger.debug(f"{s.name}: peak={peak} size={size} bytes")
                        self._rec.record_memory_delta(s.name + "_PEAK", end_time, peak)
                        self._rec.record_memory_delta(s.name + "_USED", end_time, size)
                    self._rec.record(pipe.name, op)
                    for dep in op.deps:
                        self._rec.record_dependency(dep, op)
                    yield op, end_time
            if not end_times:
                continue
            for end_time in end_times:
                heappush(next_timestamp, end_time)

    def dump(self, filepath: str = "trace.json") -> None:
        """Dump the execution trace recorded by the system to a JSON file."""

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._rec.to_json(), f, indent=4)


class BasicPipeline(Pipeline[BaseOp]):
    r"""A basic pipeline that can accept any operator."""
