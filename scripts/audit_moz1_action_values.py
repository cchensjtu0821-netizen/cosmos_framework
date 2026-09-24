# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inspect stored MOZ1 action columns and the training recipe's second transform.

Reads numeric Parquet columns only; it does not decode video or alter the dataset.
Run from the repository root with ``python -m
cosmos_framework.scripts.audit_moz1_action_values --root "$MOZ1_ROOT"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# Same feature order and aliases as the MOZ1 training recipe/adapter.
ACTION_FIELDS = (
    (
        "leftarm_cmd_cart_pos",
        6,
        ("leftarm_action_cart_pos", "leftarm_command_cart_pos", "left_arm_action_cart_pos"),
    ),
    (
        "leftarm_gripper_cmd_pos",
        1,
        ("left_gripper_action", "left_gripper_cmd", "left_gripper_cmd_pos", "left_gripper_action_pos"),
    ),
    (
        "rightarm_cmd_cart_pos",
        6,
        ("rightarm_action_cart_pos", "rightarm_command_cart_pos", "right_arm_action_cart_pos"),
    ),
    (
        "rightarm_gripper_cmd_pos",
        1,
        ("right_gripper_action", "right_gripper_cmd", "right_gripper_cmd_pos", "right_gripper_action_pos"),
    ),
    ("torso_cmd_cart_pos", 6, ("torso_action_cart_pos", "torso_command_cart_pos")),
)


def _resolve_feature(features: dict, requested: str, aliases: tuple[str, ...]) -> str:
    def matches(name: str) -> list[str]:
        return [name] if name in features else [key for key in features if key.rsplit(".", 1)[-1] == name]

    found = matches(requested)
    if not found:
        found = sorted({key for alias in aliases for key in matches(alias)})
    if len(found) != 1:
        raise ValueError(f"Cannot resolve action feature {requested!r} uniquely: {found}")
    return found[0]


def _stats(root: Path, stats_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    path = Path(stats_path) if stats_path else Path("norm_stats.json")
    path = path if path.is_absolute() else root / path
    raw = json.loads(path.read_text())
    container = raw.get("norm_stats", raw)
    actions = container["actions"]
    q01 = np.asarray(actions["q01"], dtype=np.float64)
    q99 = np.asarray(actions["q99"], dtype=np.float64)
    if q01.shape != (20,) or q99.shape != (20,) or not (np.isfinite(q01).all() and np.isfinite(q99).all()):
        raise ValueError("norm_stats.actions.q01/q99 must contain 20 finite numbers each")
    return q01, q99


def _read_actions(
    root: Path, fields: tuple[tuple[str, int], ...], max_files: int, max_rows: int
) -> tuple[np.ndarray, int]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {root / 'data'}")
    selected = [files[i] for i in np.linspace(0, len(files) - 1, min(max_files, len(files)), dtype=int)]
    per_file = max(1, max_rows // len(selected))
    chunks: list[np.ndarray] = []
    used = 0
    total_rows = 0
    for path in selected:
        parquet = pq.ParquetFile(path)
        missing = [key for key, _ in fields if key not in parquet.schema_arrow.names]
        if missing:
            raise KeyError(f"Parquet columns missing in {path}: {missing}")
        remaining = min(per_file, max_rows - total_rows)
        for batch in parquet.iter_batches(batch_size=min(4096, max(remaining, 1)), columns=[key for key, _ in fields]):
            count = min(batch.num_rows, remaining)
            if count == 0:
                break
            parts = []
            for key, width in fields:
                column = batch.column(batch.schema.get_field_index(key)).slice(0, count)
                values = np.asarray(column.to_pylist(), dtype=np.float64)
                if values.size != count * width:
                    raise ValueError(f"Unexpected shape for {key} in {path}: {values.shape}")
                parts.append(values.reshape(count, width))
            chunks.append(np.concatenate(parts, axis=1))
            remaining -= count
            total_rows += count
            if remaining == 0:
                break
        used += 1
        if total_rows >= max_rows:
            break
    if not chunks:
        raise ValueError("Selected Parquet files contain no action rows")
    values = np.concatenate(chunks, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite action values found in sampled Parquet rows")
    return values, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="MOZ1 LeRobot-v3 dataset root")
    parser.add_argument("--stats-path", help="Defaults to <root>/norm_stats.json; relative paths resolve under --root")
    parser.add_argument("--max-files", type=int, default=24, help="Number of evenly spaced Parquet files to sample")
    parser.add_argument("--max-rows", type=int, default=50000, help="Total sampled rows")
    args = parser.parse_args()
    if args.max_files < 1 or args.max_rows < 1:
        parser.error("--max-files and --max-rows must be positive")

    info = json.loads((args.root / "meta" / "info.json").read_text())
    features = info["features"]
    fields = tuple((_resolve_feature(features, name, aliases), width) for name, width, aliases in ACTION_FIELDS)
    q01, q99 = _stats(args.root, args.stats_path)
    stored, file_count = _read_actions(args.root, fields, args.max_files, args.max_rows)

    # Mirrors resolve_action_normalization("quantile") and ActionAffineNormalization.
    offset = (q99 + q01) / 2.0
    scale = np.maximum(q99 - q01, 1e-8) / 2.0
    trained = (stored - offset) / scale  # No forward clipping in the current recipe.
    lo = -1.0 - 1e-6
    hi = 1.0 + 1e-6
    print(f"sampled_rows={len(stored)} sampled_files={file_count} (deterministic, not a full-dataset scan)")
    print("Training action columns, in model order:")
    for name, width in fields:
        print(f"  {name}: {width}")
    print("First stored 20-D action row:", np.array2string(stored[0], precision=5, separator=","))
    print("Same row after current training normalization:", np.array2string(trained[0], precision=5, separator=","))
    print(
        "dim  stored_min stored_p01 stored_p99 stored_max  at_-1% at_+1% out_% "
        "stats_q01 stats_q99  trained_min trained_max trained_out_%"
    )
    for dim in range(20):
        raw = stored[:, dim]
        out = trained[:, dim]
        p01, p99 = np.quantile(raw, [0.01, 0.99])
        at_neg = 100.0 * np.mean(np.isclose(raw, -1.0, rtol=0.0, atol=1e-6))
        at_pos = 100.0 * np.mean(np.isclose(raw, 1.0, rtol=0.0, atol=1e-6))
        raw_out = 100.0 * np.mean((raw < lo) | (raw > hi))
        trained_out = 100.0 * np.mean((out < lo) | (out > hi))
        print(
            f"{dim:>3} {raw.min():>11.5g} {p01:>10.5g} {p99:>10.5g} {raw.max():>10.5g} "
            f"{at_neg:>7.2f} {at_pos:>7.2f} {raw_out:>6.2f} "
            f"{q01[dim]:>10.5g} {q99[dim]:>10.5g} {out.min():>12.5g} {out.max():>11.5g} {trained_out:>13.2f}"
        )
    print(
        "All-channel summary: "
        f"stored_outside_[-1,1]={100 * np.mean((stored < lo) | (stored > hi)):.2f}% "
        f"trained_outside_[-1,1]={100 * np.mean((trained < lo) | (trained > hi)):.2f}% "
        f"changed_by_more_than_0.05={100 * np.mean(np.abs(trained - stored) > 0.05):.2f}%"
    )
    print(
        "Clipped storage usually has no values outside [-1,1] and a pileup at -1/+1; "
        "bounded raw commands can look similar."
    )


if __name__ == "__main__":
    main()
