# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead, request-scoped CPU/CUDA profiling helpers.

CUDA regions record events asynchronously.  Synchronization happens once in
``finalize`` so instrumented model code does not serialize every operation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch


class ModuleProfiler:
    def __init__(self, *, enabled: bool, request_id: int, warmup: bool = False) -> None:
        self.enabled = enabled
        self.request_id = request_id
        self.warmup = warmup
        self._cpu_samples: dict[str, list[float]] = defaultdict(list)
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._finalized = False

    @contextmanager
    def cpu(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_samples[name].append((time.perf_counter() - start) * 1000.0)

    @contextmanager
    def cuda(self, name: str) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda_events[name].append((start, end))

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            self._counters[name] += value

    @staticmethod
    def _summary(samples: list[float]) -> dict[str, float | int]:
        return {
            "count": len(samples),
            "total_ms": sum(samples),
            "mean_ms": sum(samples) / len(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("ModuleProfiler.finalize() may only be called once")
        self._finalized = True
        if not self.enabled:
            return {"request_id": self.request_id, "enabled": False}

        if self._cuda_events:
            torch.cuda.synchronize()
        cuda_samples = {
            name: [start.elapsed_time(end) for start, end in events]
            for name, events in self._cuda_events.items()
        }
        memory: dict[str, float] = {}
        if torch.cuda.is_available():
            memory = {
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            }
        return {
            "request_id": self.request_id,
            "enabled": True,
            "warmup": self.warmup,
            "cpu": {name: self._summary(values) for name, values in self._cpu_samples.items()},
            "cuda": {name: self._summary(values) for name, values in cuda_samples.items()},
            "counters": dict(self._counters),
            "memory": memory,
        }


def profile_cpu(owner: Any, name: str):
    profiler = getattr(owner, "_module_profiler", None)
    return profiler.cpu(name) if profiler is not None else nullcontext()


def profile_cuda(owner: Any, name: str):
    profiler = getattr(owner, "_module_profiler", None)
    return profiler.cuda(name) if profiler is not None else nullcontext()


def profile_increment(owner: Any, name: str, value: int = 1) -> None:
    profiler = getattr(owner, "_module_profiler", None)
    if profiler is not None:
        profiler.increment(name, value)
