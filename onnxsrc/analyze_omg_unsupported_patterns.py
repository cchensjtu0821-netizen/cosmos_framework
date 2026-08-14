#!/usr/bin/env python3
"""Classify exact lowering candidates for selected OMG-unsupported ONNX operators.

This tool is intentionally read-only.  It does not load external tensor data and
does not rewrite the model.  Its report separates shape-preserving operations
from value-preserving transformations so unsupported operators are not removed
merely because their input and output ranks match.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Shape = tuple[int, ...]
BINARY_BROADCAST_OPS = frozenset({"Add", "Div", "Max", "Min", "Mul", "Pow", "Sub"})
PURE_CONSTANT_OPS = frozenset(
    {
        "Abs",
        "Add",
        "Cast",
        "Ceil",
        "Concat",
        "Constant",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Floor",
        "Gather",
        "Identity",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "Reshape",
        "Sin",
        "Slice",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)


def _broadcast_shape(left: Shape, right: Shape) -> Shape | None:
    """Return the exact static NumPy/ONNX broadcast result, or None."""
    result: list[int] = []
    for left_dim, right_dim in zip(reversed(left), reversed(right), strict=False):
        if left_dim == right_dim:
            result.append(left_dim)
        elif left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1:
            result.append(left_dim)
        else:
            return None
    longer = left if len(left) > len(right) else right
    result.extend(reversed(longer[: abs(len(left) - len(right))]))
    return tuple(reversed(result))


def _single_axis_concat_candidate(source: Shape, target: Shape) -> dict[str, int] | None:
    """Classify an equal-rank Expand that can be repeated by one Concat."""
    if len(source) != len(target) or not source or any(dim <= 0 for dim in (*source, *target)):
        return None
    changed = [axis for axis, dims in enumerate(zip(source, target)) if dims[0] != dims[1]]
    if len(changed) != 1:
        return None
    axis = changed[0]
    if source[axis] != 1 or target[axis] <= 1:
        return None
    if any(
        source_dim != target_dim
        for index, (source_dim, target_dim) in enumerate(zip(source, target))
        if index != axis
    ):
        return None
    return {"axis": axis, "copies": target[axis]}


def _binary_bypass_candidate(
    source: Shape,
    expanded: Shape,
    other: Shape,
    consumer_output: Shape | None,
) -> dict[str, list[int]] | None:
    """Prove that a binary consumer can broadcast the pre-Expand input itself."""
    current_result = _broadcast_shape(expanded, other)
    bypass_result = _broadcast_shape(source, other)
    if current_result is None or bypass_result != current_result:
        return None
    if consumer_output is not None and consumer_output != current_result:
        return None
    return {
        "source_shape": list(source),
        "expanded_shape": list(expanded),
        "other_shape": list(other),
        "result_shape": list(current_result),
    }


def _space_to_depth_conv_candidate(
    input_shape: Shape | None,
    output_shape: Shape | None,
    block_size: int,
) -> dict[str, Any]:
    """Check exact SpaceToDepth lowering to a fixed one-hot Conv."""
    result: dict[str, Any] = {
        "eligible": False,
        "replacement": "Conv",
        "block_size": block_size,
    }
    if input_shape is None or output_shape is None:
        result["reason"] = "missing static input or output shape"
        return result
    if len(input_shape) != 4 or len(output_shape) != 4:
        result["reason"] = "Conv lowering requires rank-4 NCHW input and output"
        return result
    if block_size <= 0:
        result["reason"] = "blocksize must be positive"
        return result
    batch, channels, height, width = input_shape
    if min(input_shape) <= 0 or height % block_size or width % block_size:
        result["reason"] = "input dimensions must be positive and H/W divisible by blocksize"
        return result
    expected = (
        batch,
        channels * block_size * block_size,
        height // block_size,
        width // block_size,
    )
    if output_shape != expected:
        result["reason"] = f"output shape does not match {expected}"
        return result
    output_channels = expected[1]
    result.update(
        {
            "eligible": True,
            "reason": "exact with fixed one-hot Conv, kernel=stride=blocksize and no padding",
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "conv_group": 1,
            "conv_kernel_shape": [block_size, block_size],
            "conv_strides": [block_size, block_size],
            "weight_shape": [output_channels, channels, block_size, block_size],
            "weight_element_count": output_channels * channels * block_size * block_size,
            "weight_nonzero_count": output_channels,
        }
    )
    return result


def _depth_to_space_conv_transpose_candidate(
    input_shape: Shape | None,
    output_shape: Shape | None,
    block_size: int,
    mode: str,
) -> dict[str, Any]:
    """Check exact DepthToSpace lowering to a fixed one-hot ConvTranspose."""
    result: dict[str, Any] = {
        "eligible": False,
        "replacement": "ConvTranspose",
        "block_size": block_size,
        "mode": mode,
    }
    if input_shape is None or output_shape is None:
        result["reason"] = "missing static input or output shape"
        return result
    if len(input_shape) != 4 or len(output_shape) != 4:
        result["reason"] = "ConvTranspose lowering requires rank-4 NCHW input and output"
        return result
    if block_size <= 0 or mode not in {"DCR", "CRD"}:
        result["reason"] = "unsupported blocksize or DepthToSpace mode"
        return result
    batch, input_channels, height, width = input_shape
    divisor = block_size * block_size
    if min(input_shape) <= 0 or input_channels % divisor:
        result["reason"] = "input channels must be positive and divisible by blocksize squared"
        return result
    output_channels = input_channels // divisor
    expected = (batch, output_channels, height * block_size, width * block_size)
    if output_shape != expected:
        result["reason"] = f"output shape does not match {expected}"
        return result
    result.update(
        {
            "eligible": True,
            "reason": (
                "exact with mode-aware fixed one-hot ConvTranspose, "
                "kernel=stride=blocksize and no overlap"
            ),
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "conv_transpose_group": 1,
            "conv_transpose_kernel_shape": [block_size, block_size],
            "conv_transpose_strides": [block_size, block_size],
            "weight_shape": [input_channels, output_channels, block_size, block_size],
            "weight_element_count": input_channels
            * output_channels
            * block_size
            * block_size,
            "weight_nonzero_count": input_channels,
        }
    )
    return result


def _shape_map(graph: Any) -> dict[str, Shape]:
    shapes: dict[str, Shape] = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions = tensor_type.shape.dim
        if all(dimension.HasField("dim_value") for dimension in dimensions):
            shapes[value.name] = tuple(int(dimension.dim_value) for dimension in dimensions)
    for tensor in graph.initializer:
        shapes[tensor.name] = tuple(int(dimension) for dimension in tensor.dims)
    return shapes


def _attribute(node: Any, name: str, onnx: Any, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _constant_tensor_values(graph: Any, onnx: Any) -> dict[str, list[Any]]:
    """Load only inline constants; external model weights remain untouched."""
    values: dict[str, list[Any]] = {}
    for tensor in graph.initializer:
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
            continue
        try:
            values[tensor.name] = onnx.numpy_helper.to_array(tensor).reshape(-1).tolist()
        except (TypeError, ValueError):
            continue
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        tensor = _attribute(node, "value", onnx)
        if tensor is not None:
            try:
                values[node.output[0]] = onnx.numpy_helper.to_array(tensor).reshape(-1).tolist()
            except (TypeError, ValueError):
                pass
            continue
        ints = _attribute(node, "value_ints", onnx)
        integer = _attribute(node, "value_int", onnx)
        if ints is not None:
            values[node.output[0]] = [int(item) for item in ints]
        elif integer is not None:
            values[node.output[0]] = [int(integer)]
    return values


def _dependency_roots(
    tensor_name: str,
    *,
    graph_inputs: set[str],
    initializers: set[str],
    producers: dict[str, Any],
    shapes: dict[str, Shape],
    seen: set[str] | None = None,
) -> set[str]:
    """Describe whether a tensor ultimately depends on runtime graph inputs."""
    seen = set() if seen is None else seen
    if tensor_name in seen:
        return {f"cycle:{tensor_name}"}
    seen.add(tensor_name)
    if tensor_name in initializers:
        return {f"initializer:{tensor_name}"}
    producer = producers.get(tensor_name)
    if producer is None:
        prefix = "graph_input" if tensor_name in graph_inputs else "unknown"
        return {f"{prefix}:{tensor_name}"}
    if producer.op_type == "Constant":
        return {f"constant:{producer.name or tensor_name}"}
    if producer.op_type == "Shape" and producer.input and producer.input[0] in shapes:
        return {f"static_shape:{producer.input[0]}"}
    if producer.op_type not in PURE_CONSTANT_OPS:
        return {f"dynamic_op:{producer.name or producer.op_type}"}
    roots: set[str] = set()
    for input_name in producer.input:
        if input_name:
            roots.update(
                _dependency_roots(
                    input_name,
                    graph_inputs=graph_inputs,
                    initializers=initializers,
                    producers=producers,
                    shapes=shapes,
                    seen=seen.copy(),
                )
            )
    return roots or {f"unknown:{producer.name or producer.op_type}"}


def _node_label(node: Any, index_by_id: dict[int, int]) -> str:
    return node.name or f"{node.op_type}_{index_by_id[id(node)]}"


def analyze(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load_model(str(model_path), load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    index_by_id = {id(node): index for index, node in enumerate(nodes)}
    shapes = _shape_map(graph)
    constants = _constant_tensor_values(graph, onnx)
    initializers = {tensor.name for tensor in graph.initializer}
    graph_inputs = {value.name for value in graph.input}
    producers = {
        output: node for node in nodes for output in node.output if output
    }
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    trigonometric: list[dict[str, Any]] = []
    expands: list[dict[str, Any]] = []
    space_to_depth: list[dict[str, Any]] = []
    depth_to_space: list[dict[str, Any]] = []

    for node in nodes:
        label = _node_label(node, index_by_id)
        if node.op_type in {"Sin", "Cos"}:
            input_name = node.input[0] if node.input else ""
            output_name = node.output[0] if node.output else ""
            roots = _dependency_roots(
                input_name,
                graph_inputs=graph_inputs,
                initializers=initializers,
                producers=producers,
                shapes=shapes,
            )
            runtime_roots = sorted(
                root
                for root in roots
                if root.startswith(("graph_input:", "unknown:", "dynamic_op:", "cycle:"))
            )
            trigonometric.append(
                {
                    "index": index_by_id[id(node)],
                    "name": label,
                    "op_type": node.op_type,
                    "input": input_name,
                    "input_shape": list(shapes[input_name]) if input_name in shapes else None,
                    "input_producer": (
                        _node_label(producers[input_name], index_by_id)
                        if input_name in producers
                        else None
                    ),
                    "output": output_name,
                    "output_shape": list(shapes[output_name]) if output_name in shapes else None,
                    "consumers": [
                        {
                            "name": _node_label(consumer, index_by_id),
                            "op_type": consumer.op_type,
                        }
                        for consumer in consumers.get(output_name, [])
                    ],
                    "dependency_roots": sorted(roots),
                    "constant_fold_candidate": not runtime_roots,
                    "safe_to_delete": False,
                    "decision": (
                        "fold to an initializer after numerical evaluation"
                        if not runtime_roots
                        else "retain or use a numerically equivalent backend-supported implementation"
                    ),
                    "warning": "Shape preservation does not make Sin/Cos value-preserving to delete.",
                }
            )
            continue

        if node.op_type == "Expand":
            source_name = node.input[0] if node.input else ""
            shape_name = node.input[1] if len(node.input) > 1 else ""
            output_name = node.output[0] if node.output else ""
            source_shape = shapes.get(source_name)
            target_values = constants.get(shape_name)
            target_shape = (
                tuple(int(value) for value in target_values)
                if target_values is not None and all(int(value) > 0 for value in target_values)
                else shapes.get(output_name)
            )
            expanded_shape = shapes.get(output_name) or target_shape
            concat_candidate = (
                _single_axis_concat_candidate(source_shape, expanded_shape)
                if source_shape is not None and expanded_shape is not None
                else None
            )
            consumer_reports: list[dict[str, Any]] = []
            for consumer in consumers.get(output_name, []):
                report: dict[str, Any] = {
                    "name": _node_label(consumer, index_by_id),
                    "op_type": consumer.op_type,
                    "bypass_expand": False,
                }
                if consumer.op_type in BINARY_BROADCAST_OPS and len(consumer.input) == 2:
                    positions = [
                        index
                        for index, input_name in enumerate(consumer.input)
                        if input_name == output_name
                    ]
                    if len(positions) == 1:
                        position = positions[0]
                        other_name = consumer.input[1 - position]
                        consumer_output = consumer.output[0] if consumer.output else ""
                        bypass = (
                            _binary_bypass_candidate(
                                source_shape,
                                expanded_shape,
                                shapes[other_name],
                                shapes.get(consumer_output),
                            )
                            if source_shape is not None
                            and expanded_shape is not None
                            and other_name in shapes
                            else None
                        )
                        if bypass is not None:
                            report.update(
                                {
                                    "bypass_expand": True,
                                    "expanded_input_index": position,
                                    "other_input": other_name,
                                    "proof": bypass,
                                }
                            )
                consumer_reports.append(report)
            all_consumers_bypass = bool(consumer_reports) and all(
                report["bypass_expand"] for report in consumer_reports
            )
            expands.append(
                {
                    "index": index_by_id[id(node)],
                    "name": label,
                    "source": source_name,
                    "source_shape": list(source_shape) if source_shape is not None else None,
                    "shape_input": shape_name,
                    "target_shape": list(target_shape) if target_shape is not None else None,
                    "output": output_name,
                    "output_shape": (
                        list(expanded_shape) if expanded_shape is not None else None
                    ),
                    "identity_candidate": source_shape is not None and source_shape == expanded_shape,
                    "concat_candidate": concat_candidate,
                    "consumer_count": len(consumer_reports),
                    "all_consumers_can_bypass": all_consumers_bypass,
                    "consumers": consumer_reports,
                }
            )
            continue

        if node.op_type == "SpaceToDepth":
            input_name = node.input[0] if node.input else ""
            output_name = node.output[0] if node.output else ""
            block_size = int(_attribute(node, "blocksize", onnx, 0))
            candidate = _space_to_depth_conv_candidate(
                shapes.get(input_name), shapes.get(output_name), block_size
            )
            candidate.update(
                {
                    "index": index_by_id[id(node)],
                    "name": label,
                    "input": input_name,
                    "output": output_name,
                    "channel_order_note": (
                        "one-hot Conv weights must implement ONNX SpaceToDepth's exact "
                        "block/channel order"
                    ),
                }
            )
            space_to_depth.append(candidate)
            continue

        if node.op_type == "DepthToSpace":
            input_name = node.input[0] if node.input else ""
            output_name = node.output[0] if node.output else ""
            block_size = int(_attribute(node, "blocksize", onnx, 0))
            raw_mode = _attribute(node, "mode", onnx, b"DCR")
            mode = raw_mode.decode() if isinstance(raw_mode, bytes) else str(raw_mode)
            candidate = _depth_to_space_conv_transpose_candidate(
                shapes.get(input_name), shapes.get(output_name), block_size, mode
            )
            candidate.update(
                {
                    "index": index_by_id[id(node)],
                    "name": label,
                    "input": input_name,
                    "output": output_name,
                    "channel_order_note": (
                        "one-hot ConvTranspose weights must implement the declared "
                        f"DepthToSpace mode {mode} exactly"
                    ),
                }
            )
            depth_to_space.append(candidate)

    bypass_edges = sum(
        int(consumer["bypass_expand"])
        for node_report in expands
        for consumer in node_report["consumers"]
    )
    summary = {
        "node_count": len(nodes),
        "sin_cos_count": len(trigonometric),
        "sin_cos_constant_fold_candidate_count": sum(
            int(report["constant_fold_candidate"]) for report in trigonometric
        ),
        "sin_cos_safe_delete_count": 0,
        "expand_count": len(expands),
        "expand_identity_candidate_count": sum(
            int(report["identity_candidate"]) for report in expands
        ),
        "expand_single_axis_concat_candidate_count": sum(
            int(report["concat_candidate"] is not None) for report in expands
        ),
        "expand_all_consumers_bypass_candidate_count": sum(
            int(report["all_consumers_can_bypass"]) for report in expands
        ),
        "expand_bypass_consumer_edge_count": bypass_edges,
        "space_to_depth_count": len(space_to_depth),
        "space_to_depth_conv_candidate_count": sum(
            int(report["eligible"]) for report in space_to_depth
        ),
        "depth_to_space_count": len(depth_to_space),
        "depth_to_space_conv_transpose_candidate_count": sum(
            int(report["eligible"]) for report in depth_to_space
        ),
    }
    return {
        "model_path": str(model_path.resolve()),
        "external_data_loaded": False,
        "operator_counts": dict(sorted(Counter(node.op_type for node in nodes).items())),
        "summary": summary,
        "sin_cos": trigonometric,
        "expand": expands,
        "space_to_depth": space_to_depth,
        "depth_to_space": depth_to_space,
        "safety_notes": [
            "Sin/Cos deletion is never classified as safe because equal shapes do not imply equal values.",
            "Expand bypass candidates prove ONNX broadcast equivalence only; backend support for the "
            "resulting implicit broadcast still requires OMG/OMC validation.",
            "Conv/ConvTranspose candidates require generated one-hot weights with exact ONNX channel order.",
            "Every implemented rewrite still requires checker and original-versus-rewritten output comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    report_path = args.report or args.onnx.with_suffix(args.onnx.suffix + ".omg_patterns.json")
    report = analyze(args.onnx)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    summary = report["summary"]
    print(f"ONNX: {args.onnx.resolve()}")
    print(
        "Sin/Cos: "
        f"total={summary['sin_cos_count']} "
        f"constant_fold={summary['sin_cos_constant_fold_candidate_count']} "
        "safe_delete=0"
    )
    print(
        "Expand: "
        f"total={summary['expand_count']} "
        f"identity={summary['expand_identity_candidate_count']} "
        f"concat={summary['expand_single_axis_concat_candidate_count']} "
        f"bypass_all_consumers={summary['expand_all_consumers_bypass_candidate_count']} "
        f"bypass_edges={summary['expand_bypass_consumer_edge_count']}"
    )
    print(
        "SpaceToDepth: "
        f"total={summary['space_to_depth_count']} "
        f"conv={summary['space_to_depth_conv_candidate_count']}"
    )
    print(
        "DepthToSpace: "
        f"total={summary['depth_to_space_count']} "
        f"conv_transpose={summary['depth_to_space_conv_transpose_candidate_count']}"
    )
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
