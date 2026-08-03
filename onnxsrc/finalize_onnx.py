#!/usr/bin/env python3
"""Normalize ONNX node names and reject tensors above the deployment rank limit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from cosmos_framework.scripts.inspect_action_policy_onnx import inspect_model


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


def _eliminate_graph_output_identities(graph) -> list[dict[str, object]]:
    """Bypass redundant Identity nodes that directly produce graph outputs."""

    graph_output_names = {value.name for value in graph.output}
    graph_input_names = {value.name for value in graph.input}
    initializer_names = {initializer.name for initializer in graph.initializer}
    consumers: dict[str, list[object]] = {}
    producers: dict[str, object] = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
        for output_name in node.output:
            if output_name:
                producers[output_name] = node

    eliminated: list[dict[str, object]] = []
    for node in list(graph.node):
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        source_name = node.input[0]
        output_name = node.output[0]
        source_producer = producers.get(source_name)
        source_consumers = consumers.get(source_name, [])
        if (
            output_name not in graph_output_names
            or source_producer is None
            or source_name in graph_output_names
            or source_name in graph_input_names
            or source_name in initializer_names
            or len(source_consumers) != 1
            or source_consumers[0] is not node
        ):
            continue
        producer_output_index = next(
            (index for index, name in enumerate(source_producer.output) if name == source_name),
            None,
        )
        if producer_output_index is None:
            continue
        source_producer.output[producer_output_index] = output_name
        graph.node.remove(node)
        for value_info in list(graph.value_info):
            if value_info.name == source_name:
                graph.value_info.remove(value_info)
        eliminated.append(
            {
                "identity_name": node.name,
                "source_tensor": source_name,
                "graph_output": output_name,
                "source_producer": source_producer.name,
            }
        )
    return eliminated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--max-rank",
        type=int,
        default=4,
        help="Fail when a node input/output or graph I/O tensor exceeds this rank.",
    )
    args = parser.parse_args()
    if args.max_rank < 0:
        parser.error("--max-rank must be non-negative")
    if args.input.parent.resolve() != args.output.parent.resolve():
        raise ValueError("Input and output must share a directory so external-data references remain valid")

    import onnx

    model = onnx.load_model(str(args.input), load_external_data=False)
    eliminated_graph_output_identities = _eliminate_graph_output_identities(model.graph)
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
    rank_audit = inspect_model(args.output, (), args.max_rank)
    rank_summary = rank_audit["summary"]
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "node_count": len(model.graph.node),
        "empty_names_before": empty_before,
        "renamed_nodes": renamed,
        "empty_names_after": sum(not node.name for node in model.graph.node),
        "duplicate_names_after": duplicates,
        "eliminated_graph_output_identity_count": len(eliminated_graph_output_identities),
        "eliminated_graph_output_identities": eliminated_graph_output_identities,
        "max_rank": args.max_rank,
        "high_rank_node_count": rank_summary["high_rank_node_count"],
        "high_rank_nodes": rank_audit["high_rank_nodes"],
        "high_rank_graph_io_count": rank_summary["high_rank_graph_io_count"],
        "high_rank_graph_io": rank_audit["high_rank_graph_io"],
        "gemm_nodes": gemm_nodes,
        "gemm_weight_renamed": gemm_weight_renamed,
        "gemm_weight_mappings": gemm_weight_mappings,
        "unresolved_gemm_nodes": unresolved_gemm_nodes,
    }
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if (
        report["empty_names_after"]
        or report["duplicate_names_after"]
        or report["high_rank_node_count"]
        or report["high_rank_graph_io_count"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
