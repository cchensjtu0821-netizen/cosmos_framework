# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Audit an ONNX model for unsupported operators and tensors above a rank limit.

The model is loaded with ``load_external_data=False`` so multi-GB external
weights are never read into memory. The report uses shape metadata already
present in graph inputs, outputs, value_info, and initializers.

Example:
  python -m cosmos_framework.scripts.inspect_action_policy_onnx \
    /path/to/edge_policy.onnx
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TARGET_OPS = (
    "ConstantOfShape",
    "Einsum",
    "Gather",
    "ScatterElements",
    "Where",
)

REWRITE_HINTS = {
    "ConstantOfShape": "Fold fixed shapes to an initializer, or use scalar + Expand.",
    "Einsum": "Lower the node's equation to Transpose/Reshape/MatMul or Mul/ReduceSum.",
    "Gather": "Fold constant indices, use Slice for contiguous indices, or move embedding lookup outside ONNX.",
    "ScatterElements": "Replace fixed-index writes with Slice/Concat or a constant mapping matrix.",
    "Where": "Fold a constant condition; arithmetic masking is safe only when NaN/Inf semantics are controlled.",
}


def _dim_value(dim: Any) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return None


def _value_info_shape(value_info: Any) -> list[int | str | None] | None:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return None
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return [_dim_value(dim) for dim in tensor_type.shape.dim]


def _constant_output_shapes(graph: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.HasField("t"):
                shapes[node.output[0]] = [int(dim) for dim in attribute.t.dims]
                break
    return shapes


def _collect_shapes(graph: Any) -> tuple[dict[str, list[int | str | None]], dict[str, str]]:
    shapes: dict[str, list[int | str | None]] = {}
    sources: dict[str, str] = {}
    for source, values in (
        ("graph_input", graph.input),
        ("graph_output", graph.output),
        ("value_info", graph.value_info),
    ):
        for value in values:
            shape = _value_info_shape(value)
            if shape is not None:
                shapes[value.name] = shape
                sources[value.name] = source
    for initializer in graph.initializer:
        shapes[initializer.name] = [int(dim) for dim in initializer.dims]
        sources[initializer.name] = "initializer"
    for name, shape in _constant_output_shapes(graph).items():
        shapes.setdefault(name, shape)
        sources.setdefault(name, "constant")
    return shapes, sources


def _attribute_value(attribute: Any, onnx: Any) -> Any:
    if attribute.type == onnx.AttributeProto.STRING:
        return attribute.s.decode("utf-8", errors="replace")
    if attribute.type == onnx.AttributeProto.INT:
        return int(attribute.i)
    if attribute.type == onnx.AttributeProto.FLOAT:
        return float(attribute.f)
    if attribute.type == onnx.AttributeProto.INTS:
        return [int(value) for value in attribute.ints]
    if attribute.type == onnx.AttributeProto.FLOATS:
        return [float(value) for value in attribute.floats]
    if attribute.type == onnx.AttributeProto.STRINGS:
        return [value.decode("utf-8", errors="replace") for value in attribute.strings]
    if attribute.type == onnx.AttributeProto.TENSOR:
        return {
            "tensor_name": attribute.t.name,
            "dims": [int(dim) for dim in attribute.t.dims],
            "data_type": int(attribute.t.data_type),
        }
    return f"<attribute_type_{int(attribute.type)}>"


def _tensor_record(
    name: str,
    shapes: dict[str, list[int | str | None]],
    sources: dict[str, str],
) -> dict[str, Any]:
    shape = shapes.get(name)
    return {
        "name": name,
        "rank": len(shape) if shape is not None else None,
        "shape": shape,
        "shape_source": sources.get(name),
    }


def inspect_model(model_path: Path, target_ops: Iterable[str], max_rank: int) -> dict[str, Any]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `onnx` to inspect an ONNX model") from exc

    model = onnx.load_model(str(model_path), load_external_data=False)
    graph = model.graph
    shapes, sources = _collect_shapes(graph)
    target_op_set = set(target_ops)
    op_counts = Counter(node.op_type for node in graph.node)
    target_nodes: list[dict[str, Any]] = []
    high_rank_nodes: list[dict[str, Any]] = []
    unknown_rank_tensors: set[str] = set()

    for index, node in enumerate(graph.node):
        inputs = [_tensor_record(name, shapes, sources) for name in node.input if name]
        outputs = [_tensor_record(name, shapes, sources) for name in node.output if name]
        tensors = inputs + outputs
        for tensor in tensors:
            if tensor["rank"] is None:
                unknown_rank_tensors.add(tensor["name"])

        record = {
            "index": index,
            "name": node.name or f"node_{index}",
            "op_type": node.op_type,
            "domain": node.domain,
            "inputs": inputs,
            "outputs": outputs,
            "attributes": {attribute.name: _attribute_value(attribute, onnx) for attribute in node.attribute},
        }
        if node.op_type in target_op_set:
            record["rewrite_hint"] = REWRITE_HINTS.get(node.op_type)
            target_nodes.append(record)

        offending_tensors = [tensor for tensor in tensors if tensor["rank"] is not None and tensor["rank"] > max_rank]
        if offending_tensors:
            high_rank_record = dict(record)
            high_rank_record["offending_tensors"] = offending_tensors
            high_rank_nodes.append(high_rank_record)

    graph_io = [
        _tensor_record(value.name, shapes, sources)
        for value in [*graph.input, *graph.output]
    ]
    high_rank_graph_io = [
        tensor for tensor in graph_io if tensor["rank"] is not None and tensor["rank"] > max_rank
    ]
    return {
        "model_path": str(model_path.resolve()),
        "ir_version": int(model.ir_version),
        "opset_imports": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "settings": {"target_ops": sorted(target_op_set), "max_rank": max_rank},
        "summary": {
            "node_count": len(graph.node),
            "op_counts": dict(sorted(op_counts.items())),
            "target_node_count": len(target_nodes),
            "high_rank_node_count": len(high_rank_nodes),
            "high_rank_graph_io_count": len(high_rank_graph_io),
            "unknown_rank_tensor_count": len(unknown_rank_tensors),
        },
        "graph_io": graph_io,
        "high_rank_graph_io": high_rank_graph_io,
        "target_nodes": target_nodes,
        "high_rank_nodes": high_rank_nodes,
        "unknown_rank_tensors": sorted(unknown_rank_tensors),
    }


def _print_summary(report: dict[str, Any], report_path: Path, print_limit: int) -> None:
    summary = report["summary"]
    print(f"Model: {report['model_path']}")
    print(f"Nodes: {summary['node_count']}")
    print(f"Target unsupported nodes: {summary['target_node_count']}")
    print(f"Nodes with tensor rank > {report['settings']['max_rank']}: {summary['high_rank_node_count']}")
    print(f"Graph I/O with tensor rank > {report['settings']['max_rank']}: {summary['high_rank_graph_io_count']}")
    print(f"Tensors with unknown rank: {summary['unknown_rank_tensor_count']}")

    for heading, records in (
        ("Target unsupported nodes", report["target_nodes"]),
        ("High-rank nodes", report["high_rank_nodes"]),
    ):
        print(f"\n{heading}:")
        if not records:
            print("  none")
            continue
        for record in records[:print_limit]:
            ranks = [
                f"{tensor['name']}:{tensor['rank']}"
                for tensor in [*record["inputs"], *record["outputs"]]
                if tensor["rank"] is not None
            ]
            print(f"  [{record['index']}] {record['name']} ({record['op_type']}) ranks={','.join(ranks)}")
        if len(records) > print_limit:
            print(f"  ... {len(records) - print_limit} more; see JSON report")
    print(f"\nJSON report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--target-op", action="append", dest="target_ops", default=None)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--print-limit", type=int, default=100)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    if args.max_rank < 0:
        parser.error("--max-rank must be non-negative")
    if not args.model_path.is_file():
        parser.error(f"model does not exist: {args.model_path}")

    target_ops = [*DEFAULT_TARGET_OPS, *(args.target_ops or [])]
    report_path = args.report_path or args.model_path.with_suffix(args.model_path.suffix + ".compatibility.json")
    report = inspect_model(args.model_path, target_ops, args.max_rank)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _print_summary(report, report_path, args.print_limit)

    if args.fail_on_findings and (
        report["summary"]["target_node_count"]
        or report["summary"]["high_rank_node_count"]
        or report["summary"]["high_rank_graph_io_count"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
