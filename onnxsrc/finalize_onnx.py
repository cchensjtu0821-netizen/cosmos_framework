#!/usr/bin/env python3
"""Normalize ONNX node names and assign unique names to unnamed nodes."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _normalized(name: str) -> str:
    value = name.replace("/", ".")
    value = re.sub(r"\.+", ".", value).strip(".")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    if args.input.parent.resolve() != args.output.parent.resolve():
        raise ValueError("Input and output must share a directory so external-data references remain valid")

    import onnx

    model = onnx.load_model(str(args.input), load_external_data=False)
    used: set[str] = set()
    empty_before = 0
    renamed = 0
    for index, node in enumerate(model.graph.node):
        original = node.name
        base = _normalized(original) if original else f"{node.op_type}_{index}"
        if not original:
            empty_before += 1
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        if candidate != original:
            node.name = candidate
            renamed += 1
        used.add(candidate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    onnx.checker.check_model(str(args.output))
    counts = Counter(node.name for node in model.graph.node)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "node_count": len(model.graph.node),
        "empty_names_before": empty_before,
        "renamed_nodes": renamed,
        "empty_names_after": sum(not node.name for node in model.graph.node),
        "duplicate_names_after": duplicates,
    }
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
