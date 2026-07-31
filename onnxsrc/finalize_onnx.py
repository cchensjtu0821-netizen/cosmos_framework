#!/usr/bin/env python3
"""Normalize ONNX node names and align Gemm names with weight module paths."""

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


def _gemm_weight_name(node, initializer_names: set[str]) -> tuple[str | None, list[str]]:
    """Return the module path for a Gemm's unique direct weight initializer."""

    weight_inputs = [
        input_name
        for input_name in node.input
        if input_name in initializer_names and _normalized(input_name).endswith(".weight")
    ]
    if len(weight_inputs) != 1:
        return None, weight_inputs

    module_name = _normalized(weight_inputs[0])[: -len(".weight")]
    if module_name.startswith("net."):
        module_name = module_name[len("net.") :]
    return module_name or None, weight_inputs


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
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    used: set[str] = set()
    empty_before = 0
    renamed = 0
    gemm_nodes = 0
    gemm_weight_renamed = 0
    gemm_weight_mappings: list[dict[str, object]] = []
    unresolved_gemm_nodes: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        original = node.name
        base = _normalized(original) if original else f"{node.op_type}_{index}"
        weight_input = None
        if node.op_type == "Gemm":
            gemm_nodes += 1
            module_name, weight_inputs = _gemm_weight_name(node, initializer_names)
            if module_name is not None:
                base = module_name
                weight_input = weight_inputs[0]
            else:
                unresolved_gemm_nodes.append(
                    {
                        "index": index,
                        "original_name": original,
                        "weight_initializer_candidates": weight_inputs,
                        "inputs": list(node.input),
                    }
                )
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
        if weight_input is not None:
            gemm_weight_renamed += 1
            gemm_weight_mappings.append(
                {
                    "index": index,
                    "original_name": original,
                    "weight_initializer": weight_input,
                    "module_name": base,
                    "assigned_name": candidate,
                    "name_collision": candidate != base,
                }
            )
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
        "gemm_nodes": gemm_nodes,
        "gemm_weight_renamed": gemm_weight_renamed,
        "gemm_weight_mappings": gemm_weight_mappings,
        "unresolved_gemm_nodes": unresolved_gemm_nodes,
    }
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
