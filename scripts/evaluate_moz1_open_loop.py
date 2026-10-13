# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare held-out MOZ1 policy samples and score their open-loop actions.

Run ``prepare``, ``validate``, and ``score`` through ``run_moz1_open_loop_eval.sh`` so inference
loads the checkpoint only once for all selected samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

_ACTION_SHAPE = (16, 20)
_GROUPS = {
    "left_position": range(0, 3),
    "left_rotation": range(3, 6),
    "left_gripper": range(6, 7),
    "right_position": range(7, 10),
    "right_rotation": range(10, 13),
    "right_gripper": range(13, 14),
    "torso_position": range(14, 17),
    "torso_rotation": range(17, 20),
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _action_matrix(value: Any, *, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != _ACTION_SHAPE[0]:
        raise ValueError(f"{name} must have shape {_ACTION_SHAPE}")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != _ACTION_SHAPE[1]:
            raise ValueError(f"{name} must have shape {_ACTION_SHAPE}")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise ValueError(f"{name} must contain only numeric values")
        numbers = [float(item) for item in row]
        if not all(math.isfinite(item) for item in numbers):
            raise ValueError(f"{name} has non-finite values")
        matrix.append(numbers)
    return matrix


def prepare(*, dataset_root: Path, checkpoint_path: Path, out_dir: Path, max_samples: int, seed: int) -> None:
    import torch
    from PIL import Image

    from cosmos_framework.data.generator.action.datasets.moz1_lerobot_dataset import MOZ1LeRobotDataset

    if max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if out_dir.exists():
        raise FileExistsError(f"Evaluation directory already exists: {out_dir}")
    stats_path = (dataset_root / "norm_stats.json").resolve(strict=True)
    dataset = MOZ1LeRobotDataset(
        root=str(dataset_root),
        fps=30.0,
        chunk_length=16,
        mode="policy",
        split="val",
        split_seed=42,
        split_val_ratio=0.01,
        action_normalization="quantile",
        sample_stride=1,
    )
    if len(dataset) == 0:
        raise ValueError("The training-matched 1% validation episode split has no valid 16-step windows")

    selected = sorted(random.Random(seed).sample(range(len(dataset)), min(max_samples, len(dataset))))
    out_dir.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for idx in selected:
        _, row_index, episode_id, frame_offset = dataset._resolve_index(idx)
        _, _, _, sample = dataset._fetch_sample(idx)
        name = f"moz1_val_{idx:08d}"
        case_dir = out_dir / "inputs" / name
        case_dir.mkdir(parents=True)

        # The adapter builds the exact three-view layout and action-field order
        # used for training. Its raw fetch retains targets in dataset-stored space.
        video = dataset._compose_video(sample)
        first = video[0]
        last = video[-1]
        for frame, filename in ((first, "conditioning.png"), (last, "ground_truth_last.png")):
            pixels = (frame * 255.0).clamp(0, 255).to(dtype=torch.uint8)
            Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy(), mode="RGB").save(case_dir / filename)

        state = dataset._build_initial_state(sample).cpu().tolist()
        target = dataset._build_action_chunk(sample).cpu().tolist()
        _action_matrix(target, name=f"target for {name}")
        _write_json(case_dir / "state.json", state)
        _write_json(case_dir / "target_action.json", target)

        task = str(sample["task"])
        prompt = next((part.strip() for part in task.split(" | ") if part.strip()), "")
        if not prompt:
            raise ValueError(f"Empty task prompt for validation index {idx}")
        _write_json(
            case_dir / "sample.json",
            {
                "name": name,
                "model_mode": "policy",
                "domain_name": "moz1",
                "vision_path": str((case_dir / "conditioning.png").resolve()),
                "state_path": str((case_dir / "state.json").resolve()),
                "stats_path": str(stats_path),
                "view_point": "concat_view",
                "prompt": prompt,
                "fps": 30,
                "action_chunk_size": 16,
                "image_size": 256,
                "num_steps": 30,
                "guidance": 1.0,
                "shift": 3.0,
                "seed": seed + idx,
            },
        )
        entries.append(
            {
                "name": name,
                "dataset_index": idx,
                "episode_id": episode_id,
                "row_index": row_index,
                "frame_offset": frame_offset,
                "target_action": str((case_dir / "target_action.json").resolve()),
            }
        )

    _write_json(
        out_dir / "manifest.json",
        {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "split": "val",
            "split_seed": 42,
            "split_val_ratio": 0.01,
            "sample_seed": seed,
            "available_windows": len(dataset),
            "selected_windows": len(entries),
            "action_space": "dataset_stored_20d",
            "samples": entries,
        },
    )
    print(f"Prepared {len(entries)} validation windows from {len(dataset)} candidates in {out_dir}")


def validate(*, out_dir: Path, checkpoint_path: Path) -> None:
    """Only reuse a complete prepared input set for the selected checkpoint."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if Path(manifest["checkpoint_path"]).resolve() != checkpoint_path.resolve():
        raise ValueError("Prepared samples belong to a different checkpoint")
    samples = manifest["samples"]
    if not samples or manifest["selected_windows"] != len(samples):
        raise ValueError("Prepared sample count does not match manifest")
    for entry in samples:
        name = entry["name"]
        case_dir = out_dir / "inputs" / name
        for path in (case_dir / "sample.json", case_dir / "conditioning.png", case_dir / "state.json",
                     Path(entry["target_action"])):
            if not path.is_file():
                raise FileNotFoundError(f"Prepared sample {name} is incomplete: {path}")
    print(f"Reusing {len(samples)} prepared validation windows in {out_dir}")


def _summary(predictions: list[list[list[float]]], targets: list[list[list[float]]]) -> dict[str, Any]:
    count = len(predictions)
    if count == 0 or count != len(targets):
        raise ValueError("Prediction and target counts must match and be nonzero")
    dim_abs = [0.0] * 20
    step_abs = [0.0] * 16
    total_abs = total_sq = zero_abs = zero_sq = 0.0
    group_sums = {name: [0.0, 0.0, 0.0] for name in _GROUPS}
    group_for_dim = {dim: name for name, dims in _GROUPS.items() for dim in dims}
    for prediction, target in zip(predictions, targets, strict=True):
        for step, (pred_row, target_row) in enumerate(zip(prediction, target, strict=True)):
            for dim, (pred, truth) in enumerate(zip(pred_row, target_row, strict=True)):
                difference = pred - truth
                absolute = abs(difference)
                squared = difference * difference
                target_abs = abs(truth)
                total_abs += absolute
                total_sq += squared
                zero_abs += target_abs
                zero_sq += truth * truth
                dim_abs[dim] += absolute
                step_abs[step] += absolute
                group = group_sums[group_for_dim[dim]]
                group[0] += absolute
                group[1] += squared
                group[2] += target_abs
    total_elements = count * 16 * 20
    return {
        "mae": total_abs / total_elements,
        "rmse": math.sqrt(total_sq / total_elements),
        "zero_action_mae": zero_abs / total_elements,
        "zero_action_rmse": math.sqrt(zero_sq / total_elements),
        "per_dimension_mae": [value / (count * 16) for value in dim_abs],
        "per_step_mae": [value / (count * 20) for value in step_abs],
        "groups": {
            name: {
                "mae": group_sums[name][0] / (count * 16 * len(indices)),
                "rmse": math.sqrt(group_sums[name][1] / (count * 16 * len(indices))),
                "zero_action_mae": group_sums[name][2] / (count * 16 * len(indices)),
            }
            for name, indices in _GROUPS.items()
        },
    }


def score(*, out_dir: Path) -> None:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    samples = manifest["samples"]
    if not samples:
        raise ValueError("No evaluation samples in manifest")

    predictions: list[list[list[float]]] = []
    targets: list[list[list[float]]] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in samples:
        name = entry["name"]
        prediction_path = out_dir / "predictions" / name / "action.json"
        status_path = prediction_path.parent / "sample_outputs.json"
        if not prediction_path.is_file() or not status_path.is_file():
            missing.append(name)
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "success":
            raise RuntimeError(f"Inference for {name} returned {status.get('status')}: {status.get('message')}")
        prediction = _action_matrix(json.loads(prediction_path.read_text()), name=f"prediction for {name}")
        target = _action_matrix(json.loads(Path(entry["target_action"]).read_text()), name=f"target for {name}")
        predictions.append(prediction)
        targets.append(target)
        errors = [
            pred - truth
            for pred_row, target_row in zip(prediction, target, strict=True)
            for pred, truth in zip(pred_row, target_row, strict=True)
        ]
        rows.append(
            {
                "name": name,
                "dataset_index": entry["dataset_index"],
                "episode_id": entry["episode_id"],
                "mae": sum(abs(error) for error in errors) / 320,
                "rmse": math.sqrt(sum(error * error for error in errors) / 320),
                "final_step_l2": math.sqrt(
                    sum((pred - truth) ** 2 for pred, truth in zip(prediction[-1], target[-1], strict=True))
                ),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing predictions for {len(missing)} of {len(samples)} samples: {missing}")

    metrics = {
        "checkpoint_path": manifest.get("checkpoint_path"),
        "split": manifest["split"],
        "split_seed": manifest["split_seed"],
        "sample_seed": manifest["sample_seed"],
        "num_samples": len(samples),
        "num_episodes": len({entry["episode_id"] for entry in samples}),
        "action_shape": list(_ACTION_SHAPE),
        "action_space": "dataset_stored_20d",
        "note": (
            "Open-loop action reconstruction only; these are not robot task-success metrics "
            "or physical-unit errors."
        ),
        **_summary(predictions, targets),
    }
    _write_json(out_dir / "metrics.json", metrics)
    with (out_dir / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(samples)} samples from {metrics['num_episodes']} validation episodes")
    print(f"Action MAE={metrics['mae']:.6g}, RMSE={metrics['rmse']:.6g}")
    print(f"Zero-action baseline MAE={metrics['zero_action_mae']:.6g}")
    print(f"Details: {out_dir / 'metrics.json'} and {out_dir / 'per_sample.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint-path", type=Path, required=True)
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.add_argument("--max-samples", type=int, default=32)
    prepare_parser.add_argument("--seed", type=int, default=42)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--out-dir", type=Path, required=True)
    validate_parser.add_argument("--checkpoint-path", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            dataset_root=args.dataset_root,
            checkpoint_path=args.checkpoint_path,
            out_dir=args.out_dir,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    elif args.command == "validate":
        validate(out_dir=args.out_dir, checkpoint_path=args.checkpoint_path)
    else:
        score(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
