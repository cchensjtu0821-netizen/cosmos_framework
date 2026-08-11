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


def _constant_resolver(graph, onnx):
    import numpy as np

    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {
        output: node for node in graph.node for output in node.output if output
    }

    def resolve(name: str, seen: set[str] | None = None):
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        tensor = initializers.get(name)
        if tensor is not None:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                return None
            return np.asarray(onnx.numpy_helper.to_array(tensor))
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type in {"Cast", "Identity"} and node.input:
            return resolve(node.input[0], seen)
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    return np.asarray(onnx.numpy_helper.to_array(attribute.t))
        return None

    return resolve


def _normalize_safe_reshape_allowzero(graph, resolve) -> tuple[int, list[str]]:
    normalized = 0
    unresolved: list[str] = []
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        attribute_index = next(
            (index for index, attribute in enumerate(node.attribute) if attribute.name == "allowzero"),
            None,
        )
        if attribute_index is None or int(node.attribute[attribute_index].i) == 0:
            continue
        shape = resolve(node.input[1]) if len(node.input) > 1 else None
        if shape is None or 0 in shape.reshape(-1).tolist():
            unresolved.append(node.name)
            continue
        del node.attribute[attribute_index]
        normalized += 1
    return normalized, unresolved


def _shape_map(graph) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = tuple(int(dim.dim_value) for dim in dims)
    for tensor in graph.initializer:
        shapes[tensor.name] = tuple(int(dim) for dim in tensor.dims)
    return shapes


def _element_type_map(graph) -> dict[str, int]:
    element_types: dict[str, int] = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        element_types[value.name] = int(value.type.tensor_type.elem_type)
    for tensor in graph.initializer:
        element_types[tensor.name] = int(tensor.data_type)
    return element_types


def _materialize_static_mul_broadcasts(graph, onnx) -> list[dict[str, object]]:
    """Replace a single singleton-axis Mul broadcast with an explicit Expand.

    OMG can lose the final feature width while constant-folding RoPE position
    tensors.  Restrict this normalization to equal-rank, fully static operands
    where exactly one axis expands from 1; those are the cases in which Expand
    is exactly the ONNX Mul broadcast, without inventing or padding values.
    """

    import numpy as np

    shapes = _shape_map(graph)
    names = {
        name
        for node in graph.node
        for name in [node.name, *node.input, *node.output]
        if name
    }
    names.update(tensor.name for tensor in graph.initializer)

    def unique(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names.add(candidate)
        return candidate

    materialized: list[dict[str, object]] = []
    for node in list(graph.node):
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        left_shape = shapes.get(node.input[0])
        right_shape = shapes.get(node.input[1])
        if (
            left_shape is None
            or right_shape is None
            or len(left_shape) < 2
            or len(left_shape) != len(right_shape)
            or left_shape == right_shape
        ):
            continue

        for input_index, (source_shape, target_shape) in enumerate(
            ((left_shape, right_shape), (right_shape, left_shape))
        ):
            expanded_axes = [
                axis
                for axis, (source_dim, target_dim) in enumerate(zip(source_shape, target_shape))
                if source_dim == 1 and target_dim > 1
            ]
            if len(expanded_axes) != 1 or any(
                source_dim != target_dim and axis not in expanded_axes
                for axis, (source_dim, target_dim) in enumerate(zip(source_shape, target_shape))
            ):
                continue

            original_input = node.input[input_index]
            prefix = node.name or node.output[0] or "mul"
            shape_name = unique(f"{prefix}_broadcast_shape")
            expanded_name = unique(f"{original_input}_expanded_for_{prefix}")
            expand_name = unique(f"{prefix}_materialize_broadcast")
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(target_shape, dtype=np.int64), name=shape_name
                )
            )
            graph.node.insert(
                list(graph.node).index(node),
                onnx.helper.make_node(
                    "Expand",
                    [original_input, shape_name],
                    [expanded_name],
                    name=expand_name,
                ),
            )
            node.input[input_index] = expanded_name
            materialized.append(
                {
                    "mul_name": node.name,
                    "input_index": input_index,
                    "source_tensor": original_input,
                    "source_shape": list(source_shape),
                    "target_shape": list(target_shape),
                    "expanded_axis": expanded_axes[0],
                    "expand_name": expand_name,
                }
            )
            break
    return materialized


def _lower_fixed_gather_nd(graph, resolve, onnx) -> tuple[int, int, list[str]]:
    """Lower fixed GatherND to Slice or flattened Slice+Concat only."""
    import numpy as np

    shapes = _shape_map(graph)
    element_types = _element_type_map(graph)
    names = {
        name
        for node in graph.node
        for name in [node.name, *node.input, *node.output]
        if name
    }
    names.update(tensor.name for tensor in graph.initializer)
    initializer_cache: dict[tuple[int, ...], str] = {}

    def unique(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names.add(candidate)
        return candidate

    def i64(values: tuple[int, ...], label: str) -> str:
        cached = initializer_cache.get(values)
        if cached is not None:
            return cached
        name = unique(label)
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
        )
        initializer_cache[values] = name
        return name

    axes = i64((0,), "omg_slice_axis_0")
    steps = i64((1,), "omg_slice_step_1")
    flat_shape = i64((-1,), "omg_flat_shape")
    contiguous_count = 0
    segmented_count = 0
    unresolved: list[str] = []

    for node in list(graph.node):
        if node.op_type != "GatherND" or len(node.input) < 2 or not node.output:
            continue
        indices = resolve(node.input[1])
        data_shape = shapes.get(node.input[0])
        if indices is None or data_shape is None or indices.ndim != 2:
            unresolved.append(node.name)
            continue
        indices = indices.astype(np.int64)
        if indices.shape[1] == 1:
            rows = indices[:, 0]
            if rows.size and np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size)):
                data_input = node.input[0]
                starts = i64((int(rows[0]),), f"{node.name}_starts")
                ends = i64((int(rows[-1]) + 1,), f"{node.name}_ends")
                del node.attribute[:]
                del node.input[:]
                node.input.extend([data_input, starts, ends, axes, steps])
                node.op_type = "Slice"
                contiguous_count += 1
                continue
        if indices.shape[1] != len(data_shape):
            unresolved.append(node.name)
            continue
        try:
            flat_indices = np.ravel_multi_index(indices.T, data_shape)
        except ValueError:
            unresolved.append(node.name)
            continue
        if flat_indices.size == 0 or np.any(np.diff(flat_indices) <= 0):
            unresolved.append(node.name)
            continue

        runs: list[tuple[int, int]] = []
        start = previous = int(flat_indices[0])
        for value in flat_indices[1:]:
            value = int(value)
            if value != previous + 1:
                runs.append((start, previous + 1))
                start = value
            previous = value
        runs.append((start, previous + 1))

        node_index = list(graph.node).index(node)
        data_input = node.input[0]
        output_name = node.output[0]
        element_type = element_types.get(data_input)
        if element_type is None:
            unresolved.append(node.name)
            continue
        prefix = node.name or f"gathernd_{node_index}"
        flattened = unique(f"{output_name}_flat")
        graph.value_info.append(
            onnx.helper.make_tensor_value_info(flattened, element_type, [int(np.prod(data_shape))])
        )
        replacement = [
            onnx.helper.make_node(
                "Reshape", [data_input, flat_shape], [flattened], name=unique(f"{prefix}_flatten")
            )
        ]
        pieces: list[str] = []
        piece_sizes: dict[str, int] = {}
        for run_index, (run_start, run_end) in enumerate(runs):
            piece = unique(f"{output_name}_slice_{run_index}")
            replacement.append(
                onnx.helper.make_node(
                    "Slice",
                    [
                        flattened,
                        i64((run_start,), f"omg_slice_start_{run_start}"),
                        i64((run_end,), f"omg_slice_end_{run_end}"),
                        axes,
                        steps,
                    ],
                    [piece],
                    name=unique(f"{prefix}_slice_{run_index}"),
                )
            )
            pieces.append(piece)
            piece_sizes[piece] = run_end - run_start
            graph.value_info.append(
                onnx.helper.make_tensor_value_info(piece, element_type, [run_end - run_start])
            )
        level = 0
        while len(pieces) > 64:
            grouped: list[str] = []
            for group_index in range(0, len(pieces), 64):
                group = pieces[group_index : group_index + 64]
                merged = unique(f"{output_name}_concat_{level}_{group_index // 64}")
                replacement.append(
                    onnx.helper.make_node(
                        "Concat", group, [merged], axis=0,
                        name=unique(f"{prefix}_concat_{level}_{group_index // 64}"),
                    )
                )
                group_size = sum(piece_sizes[item] for item in group)
                piece_sizes[merged] = group_size
                graph.value_info.append(
                    onnx.helper.make_tensor_value_info(merged, element_type, [group_size])
                )
                grouped.append(merged)
            pieces = grouped
            level += 1
        replacement.append(
            onnx.helper.make_node(
                "Concat", pieces, [output_name], axis=0, name=unique(f"{prefix}_concat_final")
            )
        )
        graph.node.remove(node)
        for offset, replacement_node in enumerate(replacement):
            graph.node.insert(node_index + offset, replacement_node)
        segmented_count += 1
    return contiguous_count, segmented_count, unresolved


def _normalize_scatter_nd_for_omg(
    graph, resolve, onnx
) -> tuple[int, int, int, list[str], list[dict[str, object]]]:
    """Lower validated row-wise ScatterND patterns for OMG compatibility."""
    import numpy as np

    shapes = _shape_map(graph)
    element_types = _element_type_map(graph)
    names = {
        name
        for node in graph.node
        for name in [node.name, *node.input, *node.output]
        if name
    }
    names.update(tensor.name for tensor in graph.initializer)

    def unique(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names.add(candidate)
        return candidate

    scalar_cache: dict[int, str] = {}

    def scalar(value: int) -> str:
        if value in scalar_cache:
            return scalar_cache[value]
        name = unique(f"omg_scatter_slice_{value}")
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name)
        )
        scalar_cache[value] = name
        return name

    axis = scalar(0)
    step = scalar(1)
    removed_none = 0
    lowered_add = 0
    lowered_overwrite = 0
    unresolved: list[str] = []

    for node in list(graph.node):
        if node.op_type != "ScatterND":
            continue
        reduction_index = next(
            (index for index, attribute in enumerate(node.attribute) if attribute.name == "reduction"),
            None,
        )
        reduction = (
            "none"
            if reduction_index is None
            else node.attribute[reduction_index].s.decode("utf-8")
        )
        if reduction in {"", "none"}:
            if reduction_index is not None:
                del node.attribute[reduction_index]
                removed_none += 1
            reduction = "none"
        if reduction not in {"add", "none"}:
            unresolved.append(node.name)
            continue
        if len(node.input) < 3 or not node.output:
            if reduction == "add":
                unresolved.append(node.name)
            continue

        data_name, indices_name, updates_name = node.input[:3]
        data_shape = shapes.get(data_name)
        updates_shape = shapes.get(updates_name)
        element_type = element_types.get(data_name)
        indices = resolve(indices_name)
        invalid_common_layout = (
            data_shape is None
            or updates_shape is None
            or element_type is None
            or indices is None
            or indices.ndim != 2
            or indices.shape[1] != 1
        )
        if invalid_common_layout:
            if reduction == "add":
                unresolved.append(node.name)
            continue
        rows = indices[:, 0].astype(np.int64)
        if (
            rows.size == 0
            or tuple(updates_shape) != (int(rows.size), *data_shape[1:])
            or int(rows[0]) < 0
            or int(rows[-1]) >= data_shape[0]
        ):
            if reduction == "add":
                unresolved.append(node.name)
            continue

        if reduction == "none":
            if rows.size > 1 and np.any(np.diff(rows) <= 0):
                continue

            prefix = node.name or f"scatternd_overwrite_{list(graph.node).index(node)}"
            node_index = list(graph.node).index(node)
            replacement = []
            concat_inputs: list[str] = []
            piece_sizes: dict[str, int] = {}
            cursor = 0

            for update_index, row_value in enumerate(rows):
                row = int(row_value)
                if cursor < row:
                    data_piece = unique(f"{node.output[0]}_data_{cursor}_{row}")
                    replacement.append(
                        onnx.helper.make_node(
                            "Slice",
                            [data_name, scalar(cursor), scalar(row), axis, step],
                            [data_piece],
                            name=unique(f"{prefix}_data_slice_{cursor}_{row}"),
                        )
                    )
                    piece_sizes[data_piece] = row - cursor
                    graph.value_info.append(
                        onnx.helper.make_tensor_value_info(
                            data_piece, element_type, [row - cursor, *data_shape[1:]]
                        )
                    )
                    concat_inputs.append(data_piece)

                update_piece = unique(f"{node.output[0]}_update_{update_index}")
                replacement.append(
                    onnx.helper.make_node(
                        "Slice",
                        [
                            updates_name,
                            scalar(update_index),
                            scalar(update_index + 1),
                            axis,
                            step,
                        ],
                        [update_piece],
                        name=unique(f"{prefix}_update_slice_{update_index}"),
                    )
                )
                piece_sizes[update_piece] = 1
                graph.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        update_piece, element_type, [1, *data_shape[1:]]
                    )
                )
                concat_inputs.append(update_piece)
                cursor = row + 1

            if cursor < data_shape[0]:
                data_piece = unique(f"{node.output[0]}_data_{cursor}_{data_shape[0]}")
                replacement.append(
                    onnx.helper.make_node(
                        "Slice",
                        [data_name, scalar(cursor), scalar(data_shape[0]), axis, step],
                        [data_piece],
                        name=unique(f"{prefix}_data_slice_{cursor}_{data_shape[0]}"),
                    )
                )
                piece_sizes[data_piece] = data_shape[0] - cursor
                graph.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        data_piece,
                        element_type,
                        [data_shape[0] - cursor, *data_shape[1:]],
                    )
                )
                concat_inputs.append(data_piece)

            level = 0
            while len(concat_inputs) > 64:
                grouped: list[str] = []
                for group_index in range(0, len(concat_inputs), 64):
                    group = concat_inputs[group_index : group_index + 64]
                    merged = unique(
                        f"{node.output[0]}_concat_{level}_{group_index // 64}"
                    )
                    replacement.append(
                        onnx.helper.make_node(
                            "Concat",
                            group,
                            [merged],
                            axis=0,
                            name=unique(
                                f"{prefix}_concat_{level}_{group_index // 64}"
                            ),
                        )
                    )
                    merged_size = sum(piece_sizes[item] for item in group)
                    piece_sizes[merged] = merged_size
                    graph.value_info.append(
                        onnx.helper.make_tensor_value_info(
                            merged, element_type, [merged_size, *data_shape[1:]]
                        )
                    )
                    grouped.append(merged)
                concat_inputs = grouped
                level += 1

            replacement.append(
                onnx.helper.make_node(
                    "Concat",
                    concat_inputs,
                    [node.output[0]],
                    axis=0,
                    name=unique(f"{prefix}_concat_final"),
                )
            )
            graph.node.remove(node)
            for offset, replacement_node in enumerate(replacement):
                graph.node.insert(node_index + offset, replacement_node)
            lowered_overwrite += 1
            continue

        if not np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size)):
            unresolved.append(node.name)
            continue

        start = int(rows[0])
        end = int(rows[-1]) + 1
        prefix = node.name or f"scatternd_add_{list(graph.node).index(node)}"
        node_index = list(graph.node).index(node)
        replacement = []
        concat_inputs: list[str] = []

        if start:
            prefix_output = unique(f"{node.output[0]}_prefix")
            replacement.append(
                onnx.helper.make_node(
                    "Slice",
                    [data_name, scalar(0), scalar(start), axis, step],
                    [prefix_output],
                    name=unique(f"{prefix}_prefix_slice"),
                )
            )
            graph.value_info.append(
                onnx.helper.make_tensor_value_info(
                    prefix_output, element_type, [start, *data_shape[1:]]
                )
            )
            concat_inputs.append(prefix_output)

        selected = unique(f"{node.output[0]}_selected_base")
        updated = unique(f"{node.output[0]}_updated")
        replacement.extend(
            [
                onnx.helper.make_node(
                    "Slice",
                    [data_name, scalar(start), scalar(end), axis, step],
                    [selected],
                    name=unique(f"{prefix}_selected_slice"),
                ),
                onnx.helper.make_node(
                    "Add",
                    [selected, updates_name],
                    [updated],
                    name=unique(f"{prefix}_selected_add"),
                ),
            ]
        )
        selected_shape = [end - start, *data_shape[1:]]
        graph.value_info.extend(
            [
                onnx.helper.make_tensor_value_info(selected, element_type, selected_shape),
                onnx.helper.make_tensor_value_info(updated, element_type, selected_shape),
            ]
        )
        concat_inputs.append(updated)

        if end < data_shape[0]:
            suffix_output = unique(f"{node.output[0]}_suffix")
            replacement.append(
                onnx.helper.make_node(
                    "Slice",
                    [data_name, scalar(end), scalar(data_shape[0]), axis, step],
                    [suffix_output],
                    name=unique(f"{prefix}_suffix_slice"),
                )
            )
            graph.value_info.append(
                onnx.helper.make_tensor_value_info(
                    suffix_output, element_type, [data_shape[0] - end, *data_shape[1:]]
                )
            )
            concat_inputs.append(suffix_output)

        replacement.append(
            onnx.helper.make_node(
                "Concat",
                concat_inputs,
                [node.output[0]],
                axis=0,
                name=unique(f"{prefix}_concat"),
            )
        )
        graph.node.remove(node)
        for offset, replacement_node in enumerate(replacement):
            graph.node.insert(node_index + offset, replacement_node)
        lowered_add += 1
    remaining = [
        {
            "name": node.name,
            "reduction": next(
                (
                    attribute.s.decode("utf-8")
                    for attribute in node.attribute
                    if attribute.name == "reduction"
                ),
                "none",
            ),
            "inputs": list(node.input),
            "input_shapes": [shapes.get(input_name) for input_name in node.input],
        }
        for node in graph.node
        if node.op_type == "ScatterND"
    ]
    return removed_none, lowered_add, lowered_overwrite, unresolved, remaining


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
    parser.add_argument(
        "--materialize-static-mul-broadcasts",
        action="store_true",
        help=(
            "Insert explicit Expand nodes for eligible static Mul broadcasts. "
            "Disabled by default to preserve the historical implicit-broadcast graph."
        ),
    )
    args = parser.parse_args()
    if args.max_rank < 0:
        parser.error("--max-rank must be non-negative")
    if args.input.parent.resolve() != args.output.parent.resolve():
        raise ValueError("Input and output must share a directory so external-data references remain valid")

    import onnx

    model = onnx.load_model(str(args.input), load_external_data=False)
    eliminated_graph_output_identities = _eliminate_graph_output_identities(model.graph)
    resolve = _constant_resolver(model.graph, onnx)
    normalized_reshape_allowzero, unresolved_reshape_allowzero = _normalize_safe_reshape_allowzero(
        model.graph, resolve
    )
    materialized_mul_broadcasts = (
        _materialize_static_mul_broadcasts(model.graph, onnx)
        if args.materialize_static_mul_broadcasts
        else []
    )
    lowered_contiguous_gather_nd, lowered_segmented_gather_nd, unresolved_gather_nd = (
        _lower_fixed_gather_nd(model.graph, resolve, onnx)
    )
    (
        removed_scatter_nd_none,
        lowered_scatter_nd_add,
        lowered_scatter_nd_overwrite,
        unresolved_scatter_nd_reduction,
        remaining_scatter_nd,
    ) = _normalize_scatter_nd_for_omg(model.graph, resolve, onnx)
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
        "normalized_reshape_allowzero_count": normalized_reshape_allowzero,
        "unresolved_reshape_allowzero": unresolved_reshape_allowzero,
        "materialize_static_mul_broadcasts_enabled": args.materialize_static_mul_broadcasts,
        "materialized_mul_broadcast_count": len(materialized_mul_broadcasts),
        "materialized_mul_broadcasts": materialized_mul_broadcasts,
        "lowered_contiguous_gather_nd_count": lowered_contiguous_gather_nd,
        "lowered_segmented_gather_nd_count": lowered_segmented_gather_nd,
        "unresolved_gather_nd": unresolved_gather_nd,
        "removed_explicit_scatter_nd_none_count": removed_scatter_nd_none,
        "lowered_scatter_nd_add_count": lowered_scatter_nd_add,
        "lowered_scatter_nd_overwrite_count": lowered_scatter_nd_overwrite,
        "unresolved_scatter_nd_reduction": unresolved_scatter_nd_reduction,
        "remaining_scatter_nd_count": len(remaining_scatter_nd),
        "remaining_scatter_nd": remaining_scatter_nd,
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
        or report["unresolved_reshape_allowzero"]
        or report["unresolved_gather_nd"]
        or report["unresolved_scatter_nd_reduction"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
