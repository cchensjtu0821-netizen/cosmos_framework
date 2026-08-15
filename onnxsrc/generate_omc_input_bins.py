#!/usr/bin/env python3
"""Generate deterministic synthetic FP16 input binaries for Edge Policy OMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path

FP16_LE = struct.Struct("<e")
FP16_MAX = 65504.0
DEFAULT_INPUT_SHAPE = (
    "video_latent:48,9,33,40;action_latent:33,64;vision_timestep:2720;"
    "action_timestep:32;prompt_embeddings:108,2048"
)
DEFAULT_INPUT_TYPE = (
    "video_latent:FP16;action_latent:FP16;vision_timestep:FP16;"
    "action_timestep:FP16;prompt_embeddings:FP16"
)


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    scale: float | None = None


INPUT_SPECS = (
    InputSpec("video_latent", (48, 9, 33, 40), 1.0),
    InputSpec("action_latent", (33, 64), 0.25),
    InputSpec("vision_timestep", (2720,)),
    InputSpec("action_timestep", (32,)),
    InputSpec("prompt_embeddings", (108, 2048), 0.02),
)


def _synthetic_input_bytes(
    spec: InputSpec,
    *,
    rng: random.Random,
    timestep: float,
) -> bytes:
    element_count = math.prod(spec.shape)
    if spec.name in {"vision_timestep", "action_timestep"}:
        return FP16_LE.pack(timestep) * element_count

    assert spec.scale is not None
    values = bytearray(element_count * FP16_LE.size)
    for index in range(element_count):
        FP16_LE.pack_into(values, index * FP16_LE.size, rng.gauss(0.0, spec.scale))
    return bytes(values)


def _summary(values: bytes) -> tuple[float, float, float]:
    iterator = (value for (value,) in struct.iter_unpack("<e", values))
    first = next(iterator)
    minimum = maximum = total = first
    count = 1
    for value in iterator:
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        total += value
        count += 1
    return minimum, maximum, total / count


def _write_binary(path: Path, values: bytes, *, overwrite: bool) -> None:
    mode = "wb" if overwrite else "xb"
    with path.open(mode) as stream:
        stream.write(values)


def generate_input_bins(
    output_dir: Path,
    *,
    seed: int,
    timestep: float,
    overwrite: bool,
) -> Path:
    if not math.isfinite(timestep) or abs(timestep) > FP16_MAX:
        raise ValueError(f"--timestep must be finite and representable as FP16, got {timestep}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"{spec.name}.bin" for spec in INPUT_SPECS]
    manifest_path = output_dir / "manifest.json"
    if not overwrite:
        existing_paths = [path for path in (*output_paths, manifest_path) if path.exists()]
        if existing_paths:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing_paths[0]}")

    rng = random.Random(seed)
    inputs: list[dict[str, object]] = []
    for spec, path in zip(INPUT_SPECS, output_paths, strict=True):
        values = _synthetic_input_bytes(spec, rng=rng, timestep=timestep)
        _write_binary(path, values, overwrite=overwrite)
        minimum, maximum, mean = _summary(values)
        element_count = math.prod(spec.shape)
        inputs.append(
            {
                "name": spec.name,
                "file": path.name,
                "dtype": "FP16_LE",
                "shape": list(spec.shape),
                "elements": element_count,
                "bytes": len(values),
                "sha256": hashlib.sha256(values).hexdigest(),
                "min": minimum,
                "max": maximum,
                "mean": mean,
            }
        )
        print(f"{spec.name}: shape={spec.shape} dtype=FP16 bytes={len(values)} file={path}")

    manifest = {
        "purpose": "synthetic Edge Policy OMC smoke-test inputs; not real Policy preprocessing output",
        "seed": seed,
        "timestep": timestep,
        "layout": "C-order raw little-endian FP16 with no header",
        "input_shape": DEFAULT_INPUT_SHAPE,
        "input_type": DEFAULT_INPUT_TYPE,
        "input_order": [spec.name for spec in INPUT_SPECS],
        "inputs": inputs,
    }
    mode = "w" if overwrite else "x"
    with manifest_path.open(mode, encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Synthetic-data random seed (default: 0)")
    parser.add_argument(
        "--timestep",
        type=float,
        default=500.0,
        help="Value repeated across both timestep inputs (default: 500.0)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files")
    args = parser.parse_args()
    generate_input_bins(
        args.output_dir,
        seed=args.seed,
        timestep=args.timestep,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
