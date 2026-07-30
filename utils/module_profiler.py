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

import numpy as np
import torch


class ModuleProfiler:
    def __init__(self, *, enabled: bool, request_id: int, warmup: bool = False) -> None:
        self.enabled = enabled
        self.request_id = request_id
        self.warmup = warmup
        self._cpu_samples: dict[str, list[float]] = defaultdict(list)
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._flops: dict[str, int] = defaultdict(int)
        self._shapes: dict[str, dict[str, Any]] = {}
        self._active_regions: list[str] = []
        self._finalized = False

    @contextmanager
    def cpu(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        self._active_regions.append(name)
        try:
            yield
        finally:
            self._active_regions.pop()
            self._cpu_samples[name].append((time.perf_counter() - start) * 1000.0)

    @contextmanager
    def cuda(self, name: str) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._active_regions.append(name)
        try:
            yield
        finally:
            self._active_regions.pop()
            end.record()
            self._cuda_events[name].append((start, end))

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            self._counters[name] += value

    def add_flops(self, name: str, value: int) -> None:
        """Record theoretical FLOPs and roll them up through active regions."""
        if not self.enabled:
            return
        value = int(value)
        if value < 0:
            raise ValueError(f"FLOPs must be non-negative, got {value} for {name!r}")
        self._flops[name] += value
        for region in dict.fromkeys(self._active_regions):
            if region != name:
                self._flops[region] += value

    @staticmethod
    def _describe_shape(value: Any) -> Any:
        """Return a JSON-serializable shape/dtype description without reading tensor data."""
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "device": "cpu"}
        if isinstance(value, dict):
            return {str(key): ModuleProfiler._describe_shape(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ModuleProfiler._describe_shape(item) for item in value]
        if value is None:
            return None
        return {"type": type(value).__name__}

    def record_shape(self, name: str, value: Any) -> None:
        """Record one key tensor structure and count repeated observations of the same shape."""
        if not self.enabled:
            return
        description = self._describe_shape(value)
        existing = self._shapes.get(name)
        if existing is None:
            self._shapes[name] = {"count": 1, "value": description}
            return
        if existing["value"] == description:
            existing["count"] += 1
            return
        variants = existing.setdefault("variants", [])
        for variant in variants:
            if variant["value"] == description:
                variant["count"] += 1
                return
        variants.append({"count": 1, "value": description})

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
            "flops": {
                name: {"total": value, "tflops": value / 1e12}
                for name, value in self._flops.items()
            },
            "shapes": self._shapes,
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


def profile_flops(owner: Any, name: str, value: int) -> None:
    profiler = getattr(owner, "_module_profiler", None)
    if profiler is not None:
        profiler.add_flops(name, value)


def profile_shape(owner: Any, name: str, value: Any) -> None:
    profiler = getattr(owner, "_module_profiler", None)
    if profiler is not None:
        profiler.record_shape(name, value)
