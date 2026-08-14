#!/usr/bin/env python3
"""Apply exact fixed-layout lowerings for selected OMG-unsupported ONNX ops.

The input and output models must share a directory so existing external-data
references remain valid.  Large external model weights are never loaded or
rewritten; newly materialized constants and one-hot convolution weights remain
inline in the output ONNX file. Runtime-dependent Sin/Cos nodes are retained by
default. An explicit profiling-only switch can bypass them when numerical
equivalence is not required.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .analyze_omg_unsupported_patterns import analyze
else:
    from analyze_omg_unsupported_patterns import analyze


Shape = tuple[int, ...]


def _space_to_depth_output_channel(
    input_channel: int,
    block_row: int,
    block_col: int,
    channels: int,
    block_size: int,
) -> int:
    """ONNX SpaceToDepth uses [block_row, block_col, channel] depth order."""
    return (block_row * block_size + block_col) * channels + input_channel


def _depth_to_space_target(
    input_channel: int,
    output_channels: int,
    block_size: int,
    mode: str,
) -> tuple[int, int, int]:
    """Return output channel and spatial offsets for one DepthToSpace channel."""
    block_area = block_size * block_size
    if mode == "DCR":
        block_offset, output_channel = divmod(input_channel, output_channels)
    elif mode == "CRD":
        output_channel, block_offset = divmod(input_channel, block_area)
    else:
        raise ValueError(f"Unsupported DepthToSpace mode: {mode}")
    block_row, block_col = divmod(block_offset, block_size)
    return output_channel, block_row, block_col


def _attribute(node: Any, name: str, onnx: Any, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


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


def _element_type_map(graph: Any) -> dict[str, int]:
    element_types: dict[str, int] = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        element_types[value.name] = int(value.type.tensor_type.elem_type)
    for tensor in graph.initializer:
        element_types[tensor.name] = int(tensor.data_type)
    return element_types


def _inline_constant(name: str, graph: Any, onnx: Any) -> Any | None:
    for tensor in graph.initializer:
        if tensor.name != name:
            continue
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
            return None
        return onnx.numpy_helper.to_array(tensor)
    for node in graph.node:
        if node.op_type != "Constant" or name not in node.output:
            continue
        tensor = _attribute(node, "value", onnx)
        if tensor is not None:
            return onnx.numpy_helper.to_array(tensor)
        values = _attribute(node, "value_ints", onnx)
        value = _attribute(node, "value_int", onnx)
        import numpy as np

        if values is not None:
            return np.asarray(values, dtype=np.int64)
        if value is not None:
            return np.asarray(value, dtype=np.int64)
    return None


def _replace_with_initializer(graph: Any, node: Any, value: Any, onnx: Any) -> None:
    if len(node.output) != 1:
        raise ValueError(f"Expected one output for {node.name!r}, got {list(node.output)}")
    output_name = node.output[0]
    if any(tensor.name == output_name for tensor in graph.initializer):
        raise ValueError(f"Initializer already produces {output_name!r}")
    graph.initializer.append(onnx.numpy_helper.from_array(value, name=output_name))
    graph.node.remove(node)


def _constant_ancestor_submodel(
    model: Any,
    output_names: list[str],
    onnx: Any,
) -> Any:
    """Extract a no-input standard-ONNX submodel for constant evaluation."""
    graph = model.graph
    nodes = list(graph.node)
    producers = {
        output_name: node_index
        for node_index, node in enumerate(nodes)
        for output_name in node.output
        if output_name
    }
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    graph_inputs = {value.name for value in graph.input}
    required_nodes: set[int] = set()
    required_initializers: set[str] = set()
    visiting: set[str] = set()

    def visit(tensor_name: str) -> None:
        if tensor_name in required_initializers:
            return
        tensor = initializers.get(tensor_name)
        if tensor is not None:
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError(f"Constant subgraph depends on external tensor {tensor_name!r}")
            required_initializers.add(tensor_name)
            return
        if tensor_name in graph_inputs:
            raise ValueError(f"Constant subgraph depends on graph input {tensor_name!r}")
        if tensor_name in visiting:
            raise ValueError(f"Cycle while resolving constant tensor {tensor_name!r}")
        producer_index = producers.get(tensor_name)
        if producer_index is None:
            raise ValueError(f"No producer for constant tensor {tensor_name!r}")
        producer = nodes[producer_index]
        visiting.add(tensor_name)
        required_nodes.add(producer_index)
        for input_name in producer.input:
            if input_name:
                visit(input_name)
        visiting.remove(tensor_name)

    for output_name in output_names:
        visit(output_name)

    value_infos = {
        value.name: value for value in [*graph.input, *graph.output, *graph.value_info]
    }
    outputs = []
    for output_name in output_names:
        value_info = value_infos.get(output_name)
        if value_info is None:
            raise ValueError(f"Missing type/shape metadata for constant output {output_name!r}")
        copied = type(value_info)()
        copied.CopyFrom(value_info)
        outputs.append(copied)

    selected_nodes = [node for index, node in enumerate(nodes) if index in required_nodes]
    selected_initializers = [
        tensor for tensor in graph.initializer if tensor.name in required_initializers
    ]
    constant_graph = onnx.helper.make_graph(
        selected_nodes,
        "constant_omg_rewrite_subgraph",
        [],
        outputs,
        initializer=selected_initializers,
    )
    submodel = onnx.helper.make_model(
        constant_graph,
        opset_imports=[opset for opset in model.opset_import],
    )
    submodel.ir_version = model.ir_version
    return submodel


def _evaluate_constant_outputs(
    model: Any,
    output_names: list[str],
    onnx: Any,
) -> dict[str, Any]:
    if not output_names:
        return {}
    submodel = _constant_ancestor_submodel(model, output_names, onnx)
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(submodel)
    values = evaluator.run(output_names, {})
    return dict(zip(output_names, values, strict=True))


def _make_space_to_depth_weight(
    input_shape: Shape,
    block_size: int,
    element_type: int,
    onnx: Any,
) -> Any:
    import numpy as np

    channels = input_shape[1]
    output_channels = channels * block_size * block_size
    dtype = onnx.helper.tensor_dtype_to_np_dtype(element_type)
    weight = np.zeros(
        (output_channels, channels, block_size, block_size),
        dtype=dtype,
    )
    for block_row in range(block_size):
        for block_col in range(block_size):
            for input_channel in range(channels):
                output_channel = _space_to_depth_output_channel(
                    input_channel,
                    block_row,
                    block_col,
                    channels,
                    block_size,
                )
                weight[output_channel, input_channel, block_row, block_col] = 1
    return weight


def _make_depth_to_space_weight(
    input_shape: Shape,
    block_size: int,
    mode: str,
    element_type: int,
    onnx: Any,
) -> Any:
    import numpy as np

    input_channels = input_shape[1]
    output_channels = input_channels // (block_size * block_size)
    dtype = onnx.helper.tensor_dtype_to_np_dtype(element_type)
    weight = np.zeros(
        (input_channels, output_channels, block_size, block_size),
        dtype=dtype,
    )
    for input_channel in range(input_channels):
        output_channel, block_row, block_col = _depth_to_space_target(
            input_channel,
            output_channels,
            block_size,
            mode,
        )
        weight[input_channel, output_channel, block_row, block_col] = 1
    return weight


def _unique_tensor_name(graph: Any, base: str) -> str:
    names = {
        name
        for node in graph.node
        for name in [*node.input, *node.output]
        if name
    }
    names.update(tensor.name for tensor in graph.initializer)
    candidate = base
    suffix = 1
    while candidate in names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _prune_unreachable(graph: Any) -> dict[str, Any]:
    if any(
        (hasattr(attribute, "HasField") and attribute.HasField("g"))
        or bool(getattr(attribute, "graphs", ()))
        for node in graph.node
        for attribute in node.attribute
    ):
        raise ValueError("Refusing to prune a graph containing control-flow subgraphs")
    producers = {
        output_name: index
        for index, node in enumerate(graph.node)
        for output_name in node.output
        if output_name
    }
    required_tensors = {output.name for output in graph.output}
    required_nodes: set[int] = set()
    stack = list(required_tensors)
    while stack:
        tensor_name = stack.pop()
        node_index = producers.get(tensor_name)
        if node_index is None or node_index in required_nodes:
            continue
        required_nodes.add(node_index)
        for input_name in graph.node[node_index].input:
            if input_name and input_name not in required_tensors:
                required_tensors.add(input_name)
                stack.append(input_name)

    nodes_before = list(graph.node)
    removed = [node for index, node in enumerate(nodes_before) if index not in required_nodes]
    kept = [node for index, node in enumerate(nodes_before) if index in required_nodes]
    del graph.node[:]
    graph.node.extend(kept)

    initializers_before = len(graph.initializer)
    kept_initializers = [
        tensor for tensor in graph.initializer if tensor.name in required_tensors
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    value_info_before = len(graph.value_info)
    kept_value_info = [value for value in graph.value_info if value.name in required_tensors]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
    return {
        "removed_node_count": len(removed),
        "removed_node_op_counts": dict(sorted(Counter(node.op_type for node in removed).items())),
        "removed_initializer_count": initializers_before - len(kept_initializers),
        "removed_value_info_count": value_info_before - len(kept_value_info),
    }


def rewrite(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    max_concat_copies: int,
    overwrite: bool,
    bypass_dynamic_trig_for_profiling: bool = False,
) -> dict[str, Any]:
    if input_path.parent.resolve() != output_path.parent.resolve():
        raise ValueError("Input and output must share a directory for external-data references")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output ONNX paths must be distinct")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")
    if max_concat_copies < 2:
        raise ValueError("max_concat_copies must be at least 2")

    import numpy as np
    import onnx

    analysis = analyze(input_path)
    model = onnx.load_model(str(input_path), load_external_data=False)
    graph = model.graph
    shapes = _shape_map(graph)
    element_types = _element_type_map(graph)
    nodes_by_name = {node.name: node for node in graph.node if node.name}
    if len(nodes_by_name) != len(graph.node):
        raise ValueError("Rewrite requires unique non-empty node names")

    constant_trig_reports = [
        item for item in analysis["sin_cos"] if item["constant_fold_candidate"]
    ]
    dynamic_trig_reports = [
        item for item in analysis["sin_cos"] if not item["constant_fold_candidate"]
    ]
    constant_trig_outputs = [item["output"] for item in constant_trig_reports]
    constant_trig_values = _evaluate_constant_outputs(model, constant_trig_outputs, onnx)

    expand_plans: list[dict[str, Any]] = []
    for item in analysis["expand"]:
        node = nodes_by_name[item["name"]]
        if item["all_consumers_can_bypass"]:
            expand_plans.append({"kind": "bypass", "item": item, "node": node})
            continue
        concat = item["concat_candidate"]
        if concat is not None and int(concat["copies"]) <= max_concat_copies:
            expand_plans.append({"kind": "concat", "item": item, "node": node})
            continue
        source = _inline_constant(node.input[0], graph, onnx)
        output_shape = tuple(int(dimension) for dimension in item["output_shape"] or ())
        if source is None or not output_shape:
            raise ValueError(
                f"Expand {node.name!r} has no exact enabled lowering: "
                f"source_shape={item['source_shape']} output_shape={item['output_shape']}"
            )
        value = np.broadcast_to(source, output_shape).copy()
        expand_plans.append(
            {"kind": "constant", "item": item, "node": node, "value": value}
        )

    if any(not item["eligible"] for item in analysis["space_to_depth"]):
        raise ValueError("At least one SpaceToDepth node is not eligible for exact Conv lowering")
    if any(not item["eligible"] for item in analysis["depth_to_space"]):
        raise ValueError(
            "At least one DepthToSpace node is not eligible for exact ConvTranspose lowering"
        )

    changes: list[dict[str, Any]] = []
    for plan in expand_plans:
        node = plan["node"]
        item = plan["item"]
        if plan["kind"] == "bypass":
            node_name = node.name
            source_name = node.input[0]
            output_name = node.output[0]
            consumer_names = []
            for consumer in graph.node:
                replaced = False
                for index, input_name in enumerate(consumer.input):
                    if input_name == output_name:
                        consumer.input[index] = source_name
                        replaced = True
                if replaced:
                    consumer_names.append(consumer.name)
            graph.node.remove(node)
            changes.append(
                {
                    "node": node_name,
                    "rewrite": "Expand->implicit binary broadcast",
                    "source_shape": item["source_shape"],
                    "output_shape": item["output_shape"],
                    "consumers": consumer_names,
                }
            )
        elif plan["kind"] == "concat":
            concat = item["concat_candidate"]
            source_name = node.input[0]
            copies = int(concat["copies"])
            del node.input[:]
            node.input.extend([source_name] * copies)
            del node.attribute[:]
            node.attribute.append(onnx.helper.make_attribute("axis", int(concat["axis"])))
            node.op_type = "Concat"
            changes.append(
                {
                    "node": node.name,
                    "rewrite": "Expand->Concat(repeated input)",
                    "axis": int(concat["axis"]),
                    "copies": copies,
                    "source_shape": item["source_shape"],
                    "output_shape": item["output_shape"],
                }
            )
        else:
            node_name = node.name
            output_name = node.output[0]
            _replace_with_initializer(graph, node, plan["value"], onnx)
            changes.append(
                {
                    "node": node_name,
                    "rewrite": "constant Expand->initializer",
                    "output": output_name,
                    "output_shape": item["output_shape"],
                    "element_count": int(plan["value"].size),
                }
            )

    for item in constant_trig_reports:
        node = nodes_by_name[item["name"]]
        value = np.asarray(constant_trig_values[item["output"]])
        node_name = node.name
        op_type = node.op_type
        output_name = node.output[0]
        _replace_with_initializer(graph, node, value, onnx)
        changes.append(
            {
                "node": node_name,
                "rewrite": f"constant {op_type}->initializer",
                "output": output_name,
                "output_shape": list(value.shape),
                "element_count": int(value.size),
            }
        )

    if bypass_dynamic_trig_for_profiling:
        graph_outputs = {output.name for output in graph.output}
        for item in dynamic_trig_reports:
            node = nodes_by_name[item["name"]]
            if item["input_shape"] != item["output_shape"]:
                raise ValueError(
                    f"Profiling-only bypass requires equal input/output shapes: {item}"
                )
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError(
                    f"Profiling-only bypass requires one input and output: {node.name!r}"
                )
            node_name = node.name
            op_type = node.op_type
            input_name = node.input[0]
            output_name = node.output[0]
            if output_name in graph_outputs:
                raise ValueError(
                    f"Refusing to bypass graph-output trigonometric node {node_name!r}"
                )
            consumer_names: list[str] = []
            for consumer in graph.node:
                replaced = False
                for input_index, consumer_input in enumerate(consumer.input):
                    if consumer_input == output_name:
                        consumer.input[input_index] = input_name
                        replaced = True
                if replaced:
                    consumer_names.append(consumer.name)
            graph.node.remove(node)
            changes.append(
                {
                    "node": node_name,
                    "rewrite": f"profiling-only {op_type}->input bypass",
                    "input": input_name,
                    "output": output_name,
                    "shape": item["output_shape"],
                    "consumers": consumer_names,
                    "numerically_equivalent": False,
                }
            )

    for item in analysis["space_to_depth"]:
        node = nodes_by_name[item["name"]]
        input_name = node.input[0]
        input_shape = shapes[input_name]
        element_type = element_types[input_name]
        weight_name = _unique_tensor_name(graph, f"{node.name}_one_hot_weight")
        weight = _make_space_to_depth_weight(
            input_shape,
            int(item["block_size"]),
            element_type,
            onnx,
        )
        graph.initializer.append(onnx.numpy_helper.from_array(weight, name=weight_name))
        del node.input[:]
        node.input.extend([input_name, weight_name])
        del node.attribute[:]
        node.attribute.extend(
            [
                onnx.helper.make_attribute("kernel_shape", [item["block_size"]] * 2),
                onnx.helper.make_attribute("strides", [item["block_size"]] * 2),
                onnx.helper.make_attribute("pads", [0, 0, 0, 0]),
                onnx.helper.make_attribute("group", 1),
            ]
        )
        node.op_type = "Conv"
        changes.append(
            {
                "node": node.name,
                "rewrite": "SpaceToDepth->one-hot Conv",
                "weight": weight_name,
                "weight_shape": list(weight.shape),
                "weight_nonzero_count": int(np.count_nonzero(weight)),
            }
        )

    for item in analysis["depth_to_space"]:
        node = nodes_by_name[item["name"]]
        input_name = node.input[0]
        input_shape = shapes[input_name]
        element_type = element_types[input_name]
        weight_name = _unique_tensor_name(graph, f"{node.name}_one_hot_weight")
        weight = _make_depth_to_space_weight(
            input_shape,
            int(item["block_size"]),
            str(item["mode"]),
            element_type,
            onnx,
        )
        graph.initializer.append(onnx.numpy_helper.from_array(weight, name=weight_name))
        del node.input[:]
        node.input.extend([input_name, weight_name])
        del node.attribute[:]
        node.attribute.extend(
            [
                onnx.helper.make_attribute("kernel_shape", [item["block_size"]] * 2),
                onnx.helper.make_attribute("strides", [item["block_size"]] * 2),
                onnx.helper.make_attribute("pads", [0, 0, 0, 0]),
                onnx.helper.make_attribute("group", 1),
            ]
        )
        node.op_type = "ConvTranspose"
        changes.append(
            {
                "node": node.name,
                "rewrite": f"DepthToSpace({item['mode']})->one-hot ConvTranspose",
                "weight": weight_name,
                "weight_shape": list(weight.shape),
                "weight_nonzero_count": int(np.count_nonzero(weight)),
            }
        )

    prune_report = _prune_unreachable(graph)
    remaining = [
        {"name": node.name, "op_type": node.op_type}
        for node in graph.node
        if node.op_type in {"Expand", "Sin", "Cos", "SpaceToDepth", "DepthToSpace"}
    ]
    remaining_structural = [
        item
        for item in remaining
        if item["op_type"] in {"Expand", "SpaceToDepth", "DepthToSpace"}
    ]
    if remaining_structural:
        raise RuntimeError(f"Structural unsupported nodes remain: {remaining_structural}")

    onnx.save_model(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    report = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "external_data_loaded": False,
        "onnx_checker": "passed",
        "node_count_before": analysis["summary"]["node_count"],
        "node_count_after": len(graph.node),
        "op_counts_before": analysis["operator_counts"],
        "op_counts_after": dict(sorted(Counter(node.op_type for node in graph.node).items())),
        "change_count": len(changes),
        "changes": changes,
        "prune": prune_report,
        "remaining_target_nodes": remaining,
        "remaining_dynamic_trig_count": sum(
            int(item["op_type"] in {"Sin", "Cos"}) for item in remaining
        ),
        "profiling_only_dynamic_trig_bypass": bypass_dynamic_trig_for_profiling,
        "semantic_equivalence_expected": not bypass_dynamic_trig_for_profiling,
        "numerical_equivalence_required": not bypass_dynamic_trig_for_profiling,
        "profiling_only_warning": (
            "Dynamic timestep Sin/Cos outputs were replaced by their inputs. "
            "This model is only for structural performance/power evaluation."
            if bypass_dynamic_trig_for_profiling
            else None
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-concat-copies", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bypass-dynamic-trig-for-profiling",
        action="store_true",
        help=(
            "NON-EQUIVALENT: replace runtime-dependent Sin/Cos outputs with "
            "their inputs for performance/power evaluation only"
        ),
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input ONNX does not exist: {args.input}")
    result = rewrite(
        args.input,
        args.output,
        args.report,
        max_concat_copies=args.max_concat_copies,
        overwrite=args.overwrite,
        bypass_dynamic_trig_for_profiling=args.bypass_dynamic_trig_for_profiling,
    )
    print(
        f"nodes={result['node_count_before']}->{result['node_count_after']} "
        f"changes={result['change_count']} "
        f"remaining_dynamic_trig={result['remaining_dynamic_trig_count']} "
        f"profiling_only={str(result['profiling_only_dynamic_trig_bypass']).lower()}"
    )
    if result["profiling_only_dynamic_trig_bypass"]:
        print(f"WARNING: {result['profiling_only_warning']}")
    print(f"Output: {args.output.resolve()}")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
