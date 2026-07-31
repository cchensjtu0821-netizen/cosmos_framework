#!/usr/bin/env python3
"""Configure Cosmos3 Linear layers for signed INT8 weight + dyn_s8 input."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def _matches_any(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def update_config(
    config_path: Path,
    *,
    bit: int,
    exclude_regexes: list[str],
    report_path: Path,
) -> None:
    if bit != 8:
        raise ValueError("Cosmos3 dyn_s8 workflow currently supports only --bit 8")
    if not config_path.is_file():
        raise FileNotFoundError(f"DOPT config does not exist: {config_path}")

    backup_path = config_path.with_suffix(config_path.suffix + ".backup")
    shutil.copy2(config_path, backup_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    strategies = config.get("layer_strategy")
    if not isinstance(strategies, dict):
        raise ValueError("DOPT config has no object-valued 'layer_strategy'")

    exclusions = [re.compile(expression) for expression in exclude_regexes]
    selected: list[str] = []
    excluded: list[str] = []
    non_linear: list[str] = []
    quant_strategy: dict[str, Any] = {
        "weight": {
            "bit": 8,
            "per_channel": True,
            "weight_algo": "min_max",
            "unsigned_quant": False,
        },
        "input": {
            "bit": 8,
            "input_algo": "dyn_s8",
        },
    }

    for layer_name, layer_config in strategies.items():
        if not isinstance(layer_config, dict):
            continue
        layer_type = str(layer_config.get("type", ""))
        layer_config["quant_strategy"] = "float"
        for stale_key in ("weight", "input", "output"):
            layer_config.pop(stale_key, None)

        if "torch.nn.modules.linear" not in layer_type:
            non_linear.append(layer_name)
            continue
        if _matches_any(layer_name, exclusions):
            excluded.append(layer_name)
            continue

        layer_config["weight"] = dict(quant_strategy["weight"])
        layer_config["input"] = dict(quant_strategy["input"])
        selected.append(layer_name)

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "config_path": str(config_path.resolve()),
        "backup_path": str(backup_path.resolve()),
        "policy": "all torch.nn.Linear: signed INT8 per-channel weight + dyn_s8 input",
        "bit": bit,
        "exclude_regexes": exclude_regexes,
        "selected_linear_count": len(selected),
        "excluded_linear_count": len(excluded),
        "non_linear_count": len(non_linear),
        "selected_linear_layers": selected,
        "excluded_linear_layers": excluded,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selected Linear layers: {len(selected)}")
    print(f"Excluded Linear layers: {len(excluded)}")
    print(f"Updated config: {config_path}")
    print(f"Selection report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--exclude-regex", action="append", default=[])
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    update_config(
        args.config_path,
        bit=args.bit,
        exclude_regexes=args.exclude_regex,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
