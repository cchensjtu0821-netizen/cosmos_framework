# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rewrite a fixed-layout Cosmos3 Policy ONNX graph for a restricted backend.

This is an ONNX-only transformation and does not modify the PyTorch model or
its serving forward path. It targets the exact patterns emitted by
``export_action_policy_onnx``:

* unary Einsum permutations -> Transpose
* ConstantOfShape -> scalar initializer + Expand
* constant Gather -> initializer, scalar Gather -> Slice + Squeeze
* supported axis-0 ScatterElements layouts -> ScatterND
* fixed batch-1 vision I/O -> rank-4 vision I/O
* fixed rank-6 vision patchify/unpatchify -> rank <= 4 space/depth layouts
* causal Trilu + Where(mask, -inf, scores) -> Add(scores, finite causal bias)
* prompt token embedding Gather -> a ``prompt_embeddings`` graph input
* fixed DROID action domain -> constant domain ID 8

The finite additive causal-mask rewrite targets deployment scores that remain
finite before masking; verification differences are reported but are not a
structural compatibility failure. The prompt rewrite is interface-equivalent
when the host supplies embeddings produced by the exported model's original
embedding table.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.scripts.inspect_action_policy_onnx import (
    DEFAULT_TARGET_OPS,
    _value_info_shape,
    inspect_model,
)


def _initializer_map(graph: Any) -> dict[str, Any]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _producer_map(graph: Any) -> dict[str, Any]:
    return {output: node for node in graph.node for output in node.output if output}


def _static_tensor_shape(graph: Any, name: str) -> tuple[int, ...] | None:
    """Return a fully static tensor shape recorded in the ONNX graph."""
    initializer = _initializer_map(graph).get(name)
    if initializer is not None:
        return tuple(int(value) for value in initializer.dims)

    value = next(
        (
            candidate
            for candidate in [*graph.input, *graph.output, *graph.value_info]
            if candidate.name == name
        ),
        None,
    )
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def _resolve_static_tensor_shape(
    graph: Any,
    name: str,
    evaluator: "ConstantEvaluator",
    onnx: Any,
    seen: set[str] | None = None,
) -> tuple[int, ...] | None:
    """Recover static shapes through common legacy-export shape-preserving ops."""
    direct = _static_tensor_shape(graph, name)
    if direct is not None:
        return direct
    seen = set() if seen is None else seen
    if name in seen:
        return None
    seen.add(name)
    node = evaluator.producers.get(name)
    if node is None:
        return None

    if node.op_type == "Reshape" and len(node.input) >= 2:
        target = evaluator.evaluate(node.input[1])
        if target is None:
            return None
        shape = [int(value) for value in target.reshape(-1)]
        if shape.count(-1) > 1:
            return None
        if -1 in shape:
            source = _resolve_static_tensor_shape(
                graph, node.input[0], evaluator, onnx, seen
            )
            if source is None:
                return None
            known = int(np.prod([value for value in shape if value != -1], dtype=np.int64))
            total = int(np.prod(source, dtype=np.int64))
            if known == 0 or total % known:
                return None
            shape[shape.index(-1)] = total // known
        return tuple(shape)

    if node.op_type == "Transpose" and node.input:
        source = _resolve_static_tensor_shape(graph, node.input[0], evaluator, onnx, seen)
        if source is None:
            return None
        permutation = tuple(int(value) for value in _attribute(node, "perm", onnx, range(len(source))))
        return tuple(source[index] for index in permutation)

    if node.op_type in {"Cast", "Identity", "Mul", "Div"}:
        candidates = [
            _resolve_static_tensor_shape(graph, input_name, evaluator, onnx, seen.copy())
            for input_name in node.input
            if input_name
        ]
        candidates = [shape for shape in candidates if shape is not None]
        return max(candidates, key=len) if candidates else None

    if node.op_type == "MatMul" and len(node.input) == 2:
        left = _resolve_static_tensor_shape(graph, node.input[0], evaluator, onnx, seen.copy())
        right = _resolve_static_tensor_shape(graph, node.input[1], evaluator, onnx, seen.copy())
        if left is None or right is None or len(left) < 2 or len(right) < 2:
            return None
        try:
            prefix = np.broadcast_shapes(left[:-2], right[:-2])
        except ValueError:
            return None
        return (*prefix, left[-2], right[-1])

    return None


def _attribute(node: Any, name: str, onnx: Any, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


class ConstantEvaluator:
    def __init__(self, model_path: Path, graph: Any, onnx: Any) -> None:
        self.base_dir = str(model_path.parent)
        self.graph = graph
        self.onnx = onnx
        self.initializers = _initializer_map(graph)
        self.producers = _producer_map(graph)
        self.cache: dict[str, np.ndarray] = {}

    def evaluate(self, name: str) -> np.ndarray | None:
        if name in self.cache:
            return self.cache[name]
        initializer = self.initializers.get(name)
        if initializer is not None:
            value = self.onnx.numpy_helper.to_array(initializer, base_dir=self.base_dir)
            self.cache[name] = value
            return value

        node = self.producers.get(name)
        if node is None:
            return None
        if node.op_type == "Shape" and node.input:
            shape = _static_tensor_shape(self.graph, node.input[0])
            if shape is None:
                return None
            start = int(_attribute(node, "start", self.onnx, 0))
            end = int(_attribute(node, "end", self.onnx, len(shape)))
            result = np.asarray(shape[start:end], dtype=np.int64)
            self.cache[name] = result
            return result
        inputs = [self.evaluate(input_name) for input_name in node.input if input_name]
        if any(value is None for value in inputs):
            return None
        values = [value for value in inputs if value is not None]

        try:
            if node.op_type == "Constant":
                tensor = _attribute(node, "value", self.onnx)
                if tensor is None:
                    return None
                result = self.onnx.numpy_helper.to_array(tensor, base_dir=self.base_dir)
            elif node.op_type == "Cast":
                dtype = self.onnx.helper.tensor_dtype_to_np_dtype(int(_attribute(node, "to", self.onnx)))
                result = values[0].astype(dtype)
            elif node.op_type == "Expand":
                result = np.broadcast_to(values[0], tuple(int(x) for x in values[1].reshape(-1)))
            elif node.op_type == "Reshape":
                result = np.reshape(values[0], tuple(int(x) for x in values[1].reshape(-1)))
            elif node.op_type == "Unsqueeze":
                axes = values[1].reshape(-1) if len(values) > 1 else _attribute(node, "axes", self.onnx)
                result = values[0]
                for axis in sorted(int(x) for x in axes):
                    result = np.expand_dims(result, axis)
            elif node.op_type == "Squeeze":
                axes = values[1].reshape(-1) if len(values) > 1 else _attribute(node, "axes", self.onnx)
                result = np.squeeze(values[0], axis=tuple(int(x) for x in axes))
            elif node.op_type == "Trilu":
                # ONNX Trilu's optional second input is the diagonal offset k;
                # triangle direction is controlled independently by `upper`.
                k = int(values[1].reshape(-1)[0]) if len(values) > 1 else 0
                upper = int(_attribute(node, "upper", self.onnx, 1))
                result = np.triu(values[0], k=k) if upper else np.tril(values[0], k=k)
            elif node.op_type == "Concat":
                result = np.concatenate(values, axis=int(_attribute(node, "axis", self.onnx, 0)))
            elif node.op_type == "Equal":
                result = np.equal(values[0], values[1])
            elif node.op_type == "Where":
                result = np.where(values[0], values[1], values[2])
            elif node.op_type == "Gather":
                axis = int(_attribute(node, "axis", self.onnx, 0))
                result = np.take(values[0], values[1].astype(np.int64), axis=axis)
            elif node.op_type == "ConstantOfShape":
                tensor = _attribute(node, "value", self.onnx)
                if tensor is None:
                    fill = np.asarray([0], dtype=np.float32)
                else:
                    fill = self.onnx.numpy_helper.to_array(tensor, base_dir=self.base_dir)
                result = np.full(
                    tuple(int(x) for x in values[0].reshape(-1)),
                    fill.reshape(-1)[0],
                    dtype=fill.dtype,
                )
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None

        self.cache[name] = np.asarray(result)
        return self.cache[name]


def _replace_all_inputs(graph: Any, old_name: str, new_name: str, *, skip_node: Any | None = None) -> None:
    for node in graph.node:
        if node is skip_node:
            continue
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
    for output in graph.output:
        if output.name == old_name:
            output.name = new_name


def _unique_name(graph: Any, prefix: str) -> str:
    names = {
        name
        for node in graph.node
        for name in [*node.input, *node.output]
        if name
    }
    names.update(value.name for value in [*graph.input, *graph.output, *graph.value_info, *graph.initializer])
    if prefix not in names:
        return prefix
    index = 1
    while f"{prefix}_{index}" in names:
        index += 1
    return f"{prefix}_{index}"


def _add_initializer(graph: Any, name: str, value: np.ndarray, onnx: Any) -> str:
    tensor = onnx.numpy_helper.from_array(np.ascontiguousarray(value), name=name)
    graph.initializer.append(tensor)
    return name


def _inline_small_external_constants(
    model_path: Path,
    graph: Any,
    onnx: Any,
    *,
    max_bytes: int = 1024 * 1024,
) -> int:
    """Materialize small Constant attributes needed by path-based shape inference."""
    count = 0
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value" or not attribute.HasField("t"):
                continue
            tensor = attribute.t
            if not onnx.external_data_helper.uses_external_data(tensor):
                continue
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
            element_count = int(np.prod(tensor.dims, dtype=np.int64)) if tensor.dims else 1
            if element_count * np.dtype(dtype).itemsize > max_bytes:
                continue
            value = onnx.numpy_helper.to_array(
                tensor,
                base_dir=str(model_path.parent),
            )
            replacement = onnx.numpy_helper.from_array(
                np.ascontiguousarray(value),
                name=tensor.name,
            )
            tensor.CopyFrom(replacement)
            count += 1
    return count


def _rewrite_unary_einsum(graph: Any, onnx: Any, changes: list[dict[str, Any]]) -> int:
    count = 0
    for node in graph.node:
        if node.op_type != "Einsum" or len(node.input) != 1:
            continue
        equation = _attribute(node, "equation", onnx)
        if isinstance(equation, bytes):
            equation = equation.decode()
        if not isinstance(equation, str) or "->" not in equation:
            continue
        source, target = equation.replace(" ", "").split("->", 1)
        if len(source) != len(target) or set(source) != set(target) or len(set(source)) != len(source):
            continue
        permutation = [source.index(label) for label in target]
        old_name = node.name
        del node.attribute[:]
        node.op_type = "Transpose"
        node.name = old_name.replace("einsum", "transpose") if old_name else old_name
        node.attribute.append(onnx.helper.make_attribute("perm", permutation))
        changes.append({"kind": "exact", "node": old_name, "rewrite": "Einsum->Transpose", "perm": permutation})
        count += 1
    return count


def _decompose_gemm_nodes(
    graph: Any,
    onnx: Any,
    changes: list[dict[str, Any]],
    materialized_transposed_weights: set[str] | None = None,
) -> int:
    """Decompose default-scale Gemm into Transpose/MatMul and optional Add."""
    materialized_transposed_weights = materialized_transposed_weights or set()
    rewritten_nodes: list[Any] = []
    count = 0
    for node in graph.node:
        if node.op_type != "Gemm":
            rewritten_nodes.append(node)
            continue
        if len(node.input) not in {2, 3} or len(node.output) != 1:
            raise ValueError(
                f"Cannot decompose Gemm {node.name!r}: expected 2 or 3 inputs and one output, "
                f"got inputs={list(node.input)!r}, outputs={list(node.output)!r}"
            )

        has_bias = len(node.input) == 3 and bool(node.input[2])
        alpha = float(_attribute(node, "alpha", onnx, 1.0))
        beta = float(_attribute(node, "beta", onnx, 1.0))
        trans_a = int(_attribute(node, "transA", onnx, 0))
        trans_b = int(_attribute(node, "transB", onnx, 0))
        if alpha != 1.0 or (has_bias and beta != 1.0):
            raise ValueError(
                f"Cannot decompose Gemm {node.name!r} without inserting dtype-sensitive scales: "
                f"alpha={alpha}, beta={beta}"
            )
        if trans_a not in {0, 1} or trans_b not in {0, 1}:
            raise ValueError(
                f"Cannot decompose Gemm {node.name!r}: transA={trans_a}, transB={trans_b}"
            )

        left = node.input[0]
        right = node.input[1]
        inserted_transposes: list[str] = []
        if trans_a:
            transposed_left = _unique_name(graph, f"{node.output[0]}_gemm_transpose_a")
            transpose_a = onnx.helper.make_node(
                "Transpose",
                [left],
                [transposed_left],
                name=f"{node.name or node.output[0]}__transpose_a",
                perm=[1, 0],
            )
            rewritten_nodes.append(transpose_a)
            left = transposed_left
            inserted_transposes.append("A")
        materialized_weight = trans_b and right in materialized_transposed_weights
        if trans_b and not materialized_weight:
            transposed_right = _unique_name(graph, f"{node.output[0]}_gemm_transpose_b")
            transpose_b = onnx.helper.make_node(
                "Transpose",
                [right],
                [transposed_right],
                name=f"{node.name or node.output[0]}__transpose_b",
                perm=[1, 0],
            )
            rewritten_nodes.append(transpose_b)
            right = transposed_right
            inserted_transposes.append("B")

        matmul_output = (
            _unique_name(graph, f"{node.output[0]}_gemm_matmul") if has_bias else node.output[0]
        )
        matmul = onnx.helper.make_node(
            "MatMul",
            [left, right],
            [matmul_output],
            # Keep the original quantized linear node name on MatMul. Step 6
            # can then recover its module name from the direct weight input.
            name=node.name,
        )
        rewritten_nodes.append(matmul)
        if has_bias:
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Add",
                    [matmul_output, node.input[2]],
                    [node.output[0]],
                    name=f"{node.name or node.output[0]}__bias_add",
                )
            )

        changes.append(
            {
                "kind": "exact",
                "node": node.name,
                "rewrite": "Gemm->MatMul" + ("+Add" if has_bias else ""),
                "has_bias": has_bias,
                "transposed_inputs": inserted_transposes,
                "materialized_transposed_weight": right if materialized_weight else None,
                "alpha": alpha,
                "beta": beta if has_bias else None,
            }
        )
        count += 1

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    return count


def _external_data_values(tensor: Any) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _set_external_data(
    tensor: Any,
    onnx: Any,
    *,
    location: str,
    offset: int,
    length: int,
) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = onnx.TensorProto.EXTERNAL
    if hasattr(tensor, "ClearField"):
        tensor.ClearField("raw_data")


def _materialize_gemm_weight_transposes(
    input_path: Path,
    model: Any,
    output_external_data_path: Path,
    onnx: Any,
) -> tuple[set[str], dict[str, Any]]:
    """Transpose Gemm B initializers inside a separate external-data copy.

    The input model must keep its original weights for the Step-5 numerical
    reference. The candidate therefore receives a distinct external-data file;
    transposed tensors overwrite the same byte ranges in that copy because a
    rank-2 transpose preserves byte length.
    """
    if output_external_data_path.parent.resolve() != input_path.parent.resolve():
        raise ValueError(
            "Transposed external data must be beside the ONNX model: "
            f"model={input_path}, external_data={output_external_data_path}"
        )

    all_tensors = list(onnx.external_data_helper._get_all_tensors(model))
    external_tensors = [
        tensor
        for tensor in all_tensors
        if onnx.external_data_helper.uses_external_data(tensor)
    ]
    source_locations = {
        values["location"]
        for tensor in external_tensors
        if "location" in (values := _external_data_values(tensor))
    }
    if len(source_locations) != 1:
        raise ValueError(
            "Physical Gemm weight transposition requires exactly one source external-data file; "
            f"found {sorted(source_locations)}"
        )
    source_location = next(iter(source_locations))
    source_external_data_path = (input_path.parent / source_location).resolve()
    if not source_external_data_path.is_file():
        raise FileNotFoundError(f"Source ONNX external data does not exist: {source_external_data_path}")
    if source_external_data_path == output_external_data_path.resolve():
        raise ValueError(
            "Transposed external data must differ from the source so the original Gemm graph "
            "remains valid for numerical comparison"
        )

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    weights: dict[str, Any] = {}
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2:
            continue
        if int(_attribute(node, "transA", onnx, 0)) != 0:
            raise ValueError(
                f"Cannot guarantee transpose-free Gemm decomposition for {node.name!r}: "
                "transA=1 would require a runtime Transpose on the activation input"
            )
        if int(_attribute(node, "transB", onnx, 0)) != 1:
            continue
        weight_name = node.input[1]
        weight = initializers.get(weight_name)
        if weight is None:
            raise ValueError(
                f"Cannot physically transpose Gemm {node.name!r}: "
                f"B input {weight_name!r} is not an initializer"
            )
        if len(weight.dims) != 2:
            raise ValueError(
                f"Cannot physically transpose Gemm {node.name!r}: "
                f"weight {weight_name!r} has rank {len(weight.dims)}, expected 2"
            )
        incompatible_consumers = [
            consumer.name or consumer.op_type
            for consumer in consumers.get(weight_name, [])
            if not (
                consumer.op_type == "Gemm"
                and len(consumer.input) >= 2
                and consumer.input[1] == weight_name
                and int(_attribute(consumer, "transB", onnx, 0)) == 1
            )
        ]
        if incompatible_consumers:
            raise ValueError(
                f"Cannot physically transpose shared weight {weight_name!r}; "
                f"incompatible consumers={incompatible_consumers}"
            )
        weights[weight_name] = weight

    if not weights:
        raise ValueError(
            "Physical Gemm weight transposition was requested, but no "
            "Gemm(transB=1) initializer weights were found"
        )

    external_ranges: list[tuple[int, int, str]] = []
    for tensor in external_tensors:
        metadata = _external_data_values(tensor)
        if metadata.get("location") != source_location or "length" not in metadata:
            continue
        start = int(metadata.get("offset", "0"))
        external_ranges.append((start, start + int(metadata["length"]), tensor.name))

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_external_data_path.name}.transpose-",
        dir=output_external_data_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_external_data_path = Path(temporary_file.name)
    try:
        shutil.copyfile(source_external_data_path, temporary_external_data_path)
        records: list[dict[str, Any]] = []
        with temporary_external_data_path.open("r+b") as output_file:
            for weight_name, weight in weights.items():
                original_shape = tuple(int(dim) for dim in weight.dims)
                value = np.asarray(
                    onnx.numpy_helper.to_array(weight, base_dir=str(input_path.parent))
                )
                if tuple(value.shape) != original_shape:
                    raise ValueError(
                        f"Loaded weight shape mismatch for {weight_name!r}: "
                        f"metadata={original_shape}, value={value.shape}"
                    )
                transposed = np.ascontiguousarray(value.T)
                metadata = _external_data_values(weight)
                if metadata.get("location") == source_location:
                    offset = int(metadata.get("offset", "0"))
                    declared_length = int(metadata.get("length", str(value.nbytes)))
                    if declared_length != value.nbytes or transposed.nbytes != value.nbytes:
                        raise ValueError(
                            f"External-data byte length mismatch for {weight_name!r}: "
                            f"declared={declared_length}, original={value.nbytes}, "
                            f"transposed={transposed.nbytes}"
                        )
                    overlaps = [
                        name
                        for start, end, name in external_ranges
                        if name != weight_name
                        and offset < end
                        and start < offset + declared_length
                    ]
                    if overlaps:
                        raise ValueError(
                            f"Cannot overwrite external range for {weight_name!r}; "
                            f"it overlaps tensors={overlaps[:20]}"
                        )
                else:
                    output_file.seek(0, os.SEEK_END)
                    offset = output_file.tell()
                output_file.seek(offset)
                output_file.write(memoryview(transposed).cast("B"))
                _set_external_data(
                    weight,
                    onnx,
                    location=output_external_data_path.name,
                    offset=offset,
                    length=transposed.nbytes,
                )
                del weight.dims[:]
                weight.dims.extend(int(dim) for dim in transposed.shape)
                records.append(
                    {
                        "weight": weight_name,
                        "original_shape": list(original_shape),
                        "transposed_shape": list(transposed.shape),
                        "offset": offset,
                        "length": transposed.nbytes,
                    }
                )
            output_file.flush()
            os.fsync(output_file.fileno())

        for tensor in external_tensors:
            if tensor.name in weights:
                continue
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = output_external_data_path.name
        temporary_external_data_path.replace(output_external_data_path)
    finally:
        temporary_external_data_path.unlink(missing_ok=True)

    return set(weights), {
        "enabled": True,
        "source_external_data_path": str(source_external_data_path),
        "output_external_data_path": str(output_external_data_path.resolve()),
        "transposed_weight_count": len(weights),
        "weights": records,
    }


def _rewrite_constant_of_shape(
    model_path: Path,
    graph: Any,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    count = 0
    for node in graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        tensor = _attribute(node, "value", onnx)
        if tensor is None:
            scalar = np.asarray([0], dtype=np.float32)
        else:
            scalar = onnx.numpy_helper.to_array(
                tensor,
                base_dir=str(model_path.parent),
            )
        scalar_name = _unique_name(graph, f"{node.output[0]}_fill_value")
        _add_initializer(graph, scalar_name, scalar.reshape(1), onnx)
        old_name = node.name
        del node.attribute[:]
        node.op_type = "Expand"
        node.name = old_name.replace("ConstantOfShape", "Expand") if old_name else old_name
        node.input.insert(0, scalar_name)
        changes.append({"kind": "exact", "node": old_name, "rewrite": "ConstantOfShape->Expand"})
        count += 1
    return count


def _externalize_prompt_embedding(
    model_path: Path,
    graph: Any,
    table_path: Path,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) < 2 or node.input[1] != "prompt_token_ids":
            continue
        embedding_initializer = _initializer_map(graph).get(node.input[0])
        if embedding_initializer is None:
            raise RuntimeError(f"Prompt embedding table {node.input[0]!r} is not an initializer")
        embedding_table = onnx.numpy_helper.to_array(
            embedding_initializer,
            base_dir=str(model_path.parent),
        )

        output_name = node.output[0]
        output_info = next((value for value in graph.value_info if value.name == output_name), None)
        if output_info is None:
            prompt_input = next(
                (value for value in graph.input if value.name == "prompt_token_ids"),
                None,
            )
            if prompt_input is None:
                raise RuntimeError("Missing graph input 'prompt_token_ids'")
            prompt_shape = _value_info_shape(prompt_input)
            embedding_shape = tuple(int(dimension) for dimension in embedding_initializer.dims)
            if prompt_shape is None or len(embedding_shape) < 2:
                raise RuntimeError(
                    f"Cannot infer prompt embedding output metadata: "
                    f"prompt_shape={prompt_shape}, embedding_shape={embedding_shape}"
                )
            output_info = onnx.helper.make_tensor_value_info(
                output_name,
                embedding_initializer.data_type,
                [*prompt_shape, *embedding_shape[1:]],
            )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with table_path.open("wb") as table_file:
            np.save(table_file, embedding_table, allow_pickle=False)
        prompt_embeddings = "prompt_embeddings"
        new_input = copy.deepcopy(output_info)
        new_input.name = prompt_embeddings
        graph.input.append(new_input)
        _replace_all_inputs(graph, output_name, prompt_embeddings, skip_node=node)
        graph.node.remove(node)
        for index, graph_input in enumerate(graph.input):
            if graph_input.name == "prompt_token_ids":
                del graph.input[index]
                break
        changes.append(
            {
                "kind": "interface_equivalent",
                "node": node.name,
                "rewrite": "prompt_token_ids embedding Gather->prompt_embeddings input",
                "requirement": "Host must use the original embedding table to produce prompt_embeddings.",
                "embedding_table_path": str(table_path.resolve()),
            }
        )
        return 1
    return 0


def _freeze_action_domain(graph: Any, domain_id: int, onnx: Any, changes: list[dict[str, Any]]) -> None:
    for index, graph_input in enumerate(graph.input):
        if graph_input.name == "action_domain_id":
            del graph.input[index]
            break
    initializer = _initializer_map(graph).get("action_domain_id")
    if initializer is not None:
        graph.initializer.remove(initializer)
    _add_initializer(graph, "action_domain_id", np.asarray([domain_id], dtype=np.int64), onnx)
    changes.append(
        {
            "kind": "exact_for_fixed_configuration",
            "rewrite": "freeze action_domain_id",
            "domain_id": domain_id,
        }
    )


def _fold_constant_gathers(
    model_path: Path,
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    del model_path
    count = 0
    initializers = _initializer_map(graph)
    for node in list(graph.node):
        if node.op_type != "Gather" or len(node.input) < 2 or node.input[0] not in initializers:
            continue
        data = evaluator.evaluate(node.input[0])
        indices = evaluator.evaluate(node.input[1])
        if data is None or indices is None:
            continue
        axis = int(_attribute(node, "axis", onnx, 0))
        result = np.take(data, indices.astype(np.int64), axis=axis)
        output_name = node.output[0]
        _add_initializer(graph, output_name, result, onnx)
        graph.node.remove(node)
        changes.append({"kind": "exact", "node": node.name, "rewrite": "constant Gather->initializer"})
        count += 1
    return count


def _rewrite_scalar_gathers(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    count = 0
    replacement_nodes: list[tuple[Any, list[Any]]] = []
    for node in list(graph.node):
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        indices = evaluator.evaluate(node.input[1])
        if indices is None or indices.size != 1:
            continue
        axis = int(_attribute(node, "axis", onnx, 0))
        index = int(indices.reshape(-1)[0])
        starts = _add_initializer(graph, _unique_name(graph, f"{node.output[0]}_starts"), np.asarray([index]), onnx)
        ends = _add_initializer(graph, _unique_name(graph, f"{node.output[0]}_ends"), np.asarray([index + 1]), onnx)
        axes = _add_initializer(graph, _unique_name(graph, f"{node.output[0]}_axes"), np.asarray([axis]), onnx)
        steps = _add_initializer(graph, _unique_name(graph, f"{node.output[0]}_steps"), np.asarray([1]), onnx)
        sliced = _unique_name(graph, f"{node.output[0]}_slice")
        slice_node = onnx.helper.make_node(
            "Slice",
            [node.input[0], starts, ends, axes, steps],
            [sliced],
            name=f"{node.name}_slice",
        )
        squeeze_node = onnx.helper.make_node(
            "Squeeze",
            [sliced, axes],
            [node.output[0]],
            name=f"{node.name}_squeeze",
        )
        replacement_nodes.append((node, [slice_node, squeeze_node]))
        changes.append({"kind": "exact", "node": node.name, "rewrite": "scalar Gather->Slice+Squeeze"})
        count += 1

    for old_node, new_nodes in replacement_nodes:
        index = list(graph.node).index(old_node)
        graph.node.remove(old_node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(index + offset, new_node)
    return count


def _rewrite_scatter_elements(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    producers = _producer_map(graph)
    count = 0
    for node in list(graph.node):
        if node.op_type != "ScatterElements" or int(_attribute(node, "axis", onnx, 0)) != 0:
            continue
        indices = evaluator.evaluate(node.input[1])
        reduction = _attribute(node, "reduction", onnx, b"none")
        if isinstance(reduction, bytes):
            reduction = reduction.decode()

        compact: np.ndarray | None = None
        compact_name: str | None = None
        rewrite = "ScatterElements(axis=0)->ScatterND"
        if indices is not None and indices.ndim == 2 and np.all(indices == indices[:, :1]):
            compact = indices[:, :1]
        elif indices is not None and indices.ndim >= 2 and np.all(indices == indices.reshape(-1)[0]):
            compact = np.asarray([[indices.reshape(-1)[0]]], dtype=indices.dtype)

        if compact is not None:
            compact_name = _add_initializer(
                graph,
                _unique_name(graph, f"{node.output[0]}_scatternd_indices"),
                compact.astype(np.int64),
                onnx,
            )
        else:
            # torch.scatter_add(dim=0) exports a 1-D row index as
            # Unsqueeze(index, 1) -> Expand([N, hidden]) -> ScatterElements.
            # ScatterND accepts the compact [N, 1] index directly and applies
            # each [hidden] update to the selected row, so dropping only the
            # redundant Expand is exactly equivalent.
            expand_node = producers.get(node.input[1])
            if expand_node is None or expand_node.op_type != "Expand" or not expand_node.input:
                continue
            candidate_name = expand_node.input[0]
            candidate_value = evaluator.evaluate(candidate_name)
            if candidate_value is not None and candidate_value.ndim == 2 and candidate_value.shape[1] == 1:
                compact_name = _add_initializer(
                    graph,
                    _unique_name(graph, f"{node.output[0]}_scatternd_indices"),
                    candidate_value.astype(np.int64),
                    onnx,
                )
                rewrite = "ScatterElements(axis=0, expanded constant row indices)->ScatterND"
            else:
                unsqueeze_node = producers.get(candidate_name)
                if unsqueeze_node is None or unsqueeze_node.op_type != "Unsqueeze" or not unsqueeze_node.input:
                    continue
                axes = (
                    evaluator.evaluate(unsqueeze_node.input[1])
                    if len(unsqueeze_node.input) > 1
                    else np.asarray(_attribute(unsqueeze_node, "axes", onnx, []), dtype=np.int64)
                )
                if axes is None or axes.size != 1 or int(axes.reshape(-1)[0]) not in (1, -1):
                    continue
                compact_name = candidate_name
                rewrite = "ScatterElements(axis=0, expanded row indices)->ScatterND"

        old_name = node.name
        node.op_type = "ScatterND"
        node.name = old_name.replace("scatter", "scatternd") if old_name else old_name
        node.input[1] = compact_name
        del node.attribute[:]
        if reduction != "none":
            node.attribute.append(onnx.helper.make_attribute("reduction", reduction))
        changes.append(
            {
                "kind": "exact",
                "node": old_name,
                "rewrite": rewrite,
                "reduction": reduction,
            }
        )
        count += 1
    return count


def _normalize_nd_indices(graph: Any, onnx: Any, changes: list[dict[str, Any]]) -> int:
    """Cast GatherND/ScatterND indices to the int64 type required by ONNX Runtime."""
    count = 0
    for node in list(graph.node):
        if node.op_type not in {"GatherND", "ScatterND"} or len(node.input) < 2:
            continue
        original_indices = node.input[1]
        op_name = node.op_type.lower()
        cast_output = _unique_name(graph, f"{node.output[0]}_{op_name}_indices_int64")
        cast_node = onnx.helper.make_node(
            "Cast",
            [original_indices],
            [cast_output],
            name=f"{node.name}_indices_int64",
            to=onnx.TensorProto.INT64,
        )
        insert_at = list(graph.node).index(node)
        graph.node.insert(insert_at, cast_node)
        node.input[1] = cast_output
        changes.append(
            {
                "kind": "exact",
                "node": node.name,
                "rewrite": f"{node.op_type} indices->int64",
                "original_indices": original_indices,
            }
        )
        count += 1
    return count


def _replace_node(graph: Any, old_nodes: list[Any], new_nodes: list[Any]) -> None:
    indexes = [list(graph.node).index(node) for node in old_nodes]
    insert_at = min(indexes)
    for node in old_nodes:
        graph.node.remove(node)
    for offset, node in enumerate(new_nodes):
        graph.node.insert(insert_at + offset, node)


def _rewrite_fixed_vision_batch_io(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    """Remove the fixed batch=1 axis from the exported vision input/output."""
    count = 0

    video_input = next((value for value in graph.input if value.name == "video_latent"), None)
    if video_input is not None:
        dims = video_input.type.tensor_type.shape.dim
        consumers = [node for node in graph.node if "video_latent" in node.input]
        if (
            len(dims) == 5
            and dims[0].HasField("dim_value")
            and int(dims[0].dim_value) == 1
            and len(consumers) == 1
            and consumers[0].op_type == "Squeeze"
            and len(consumers[0].input) > 1
        ):
            squeeze = consumers[0]
            axes = evaluator.evaluate(squeeze.input[1])
            if axes is not None and axes.size == 1 and int(axes.reshape(-1)[0]) == 0:
                squeezed_name = squeeze.output[0]
                for node in graph.node:
                    if node is not squeeze:
                        for index, input_name in enumerate(node.input):
                            if input_name == squeezed_name:
                                node.input[index] = "video_latent"
                del dims[0]
                graph.node.remove(squeeze)
                changes.append(
                    {
                        "kind": "exact_for_fixed_configuration",
                        "node": squeeze.name,
                        "rewrite": "remove fixed batch=1 vision input Squeeze",
                    }
                )
                count += 1

    vision_output = next((value for value in graph.output if value.name == "vision_velocity"), None)
    producer = _producer_map(graph).get("vision_velocity")
    if vision_output is not None and producer is not None and producer.op_type == "Unsqueeze":
        dims = vision_output.type.tensor_type.shape.dim
        axes = evaluator.evaluate(producer.input[1]) if len(producer.input) > 1 else None
        if (
            len(dims) == 5
            and dims[0].HasField("dim_value")
            and int(dims[0].dim_value) == 1
            and axes is not None
            and axes.size == 1
            and int(axes.reshape(-1)[0]) == 0
        ):
            source_name = producer.input[0]
            del producer.attribute[:]
            del producer.input[:]
            producer.input.append(source_name)
            producer.op_type = "Identity"
            producer.name = producer.name.replace("unsqueeze", "identity")
            del dims[0]
            changes.append(
                {
                    "kind": "exact_for_fixed_configuration",
                    "node": producer.name,
                    "rewrite": "remove fixed batch=1 vision output Unsqueeze",
                }
            )
            count += 1

    return count


def _rewrite_patchify_rank6(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    """Lower fixed 6-D vision patchify/unpatchify through rank <= 4 ops."""
    producers = _producer_map(graph)
    count = 0

    for final_reshape in list(graph.node):
        if final_reshape.op_type != "Reshape" or len(final_reshape.input) < 2:
            continue
        transpose = producers.get(final_reshape.input[0])
        if transpose is None or transpose.op_type != "Transpose" or not transpose.input:
            continue
        first_reshape = producers.get(transpose.input[0])
        if first_reshape is None or first_reshape.op_type != "Reshape" or len(first_reshape.input) < 2:
            continue

        first_shape_value = evaluator.evaluate(first_reshape.input[1])
        final_shape_value = evaluator.evaluate(final_reshape.input[1])
        if first_shape_value is None or final_shape_value is None:
            continue
        first_shape = tuple(int(value) for value in first_shape_value.reshape(-1))
        final_shape = tuple(int(value) for value in final_shape_value.reshape(-1))
        permutation = tuple(int(value) for value in _attribute(transpose, "perm", onnx, []))

        source_name = first_reshape.input[0]
        output_name = final_reshape.output[0]

        # Patchify:
        # [C,T,H*p,W*p] -> reshape [C,T,H,p,W,p]
        # -> transpose [T,H,W,p,p,C] -> [T*H*W,p*p*C].
        if (
            len(first_shape) == 6
            and len(final_shape) == 2
            and permutation == (1, 2, 4, 3, 5, 0)
            and first_shape[3] == first_shape[5]
            and first_shape[3] > 0
            and final_shape[0] == first_shape[1] * first_shape[2] * first_shape[4]
            and final_shape[1] == first_shape[3] * first_shape[5] * first_shape[0]
        ):
            channels, frames, patch_h_count, patch_h, patch_w_count, patch_w = first_shape
            block_size = patch_h
            channel_patch = channels * block_size * block_size
            prefix = output_name
            to_nchw = _unique_name(graph, f"{prefix}_patchify_nchw")
            space_to_depth = _unique_name(graph, f"{prefix}_space_to_depth")
            channels_last = _unique_name(graph, f"{prefix}_patchify_channels_last")
            final_shape_name = _add_initializer(
                graph,
                _unique_name(graph, f"{prefix}_patchify_final_shape"),
                np.asarray(final_shape, dtype=np.int64),
                onnx,
            )
            new_nodes = [
                onnx.helper.make_node(
                    "Transpose",
                    [source_name],
                    [to_nchw],
                    name=f"{first_reshape.name}_rank4_transpose",
                    perm=[1, 0, 2, 3],
                ),
                onnx.helper.make_node(
                    "SpaceToDepth",
                    [to_nchw],
                    [space_to_depth],
                    name=f"{first_reshape.name}_space_to_depth",
                    blocksize=block_size,
                ),
                # ONNX SpaceToDepth orders its depth axis as [p, q, C],
                # exactly matching forward's "cthpwq->thwpqc".
                onnx.helper.make_node(
                    "Transpose",
                    [space_to_depth],
                    [channels_last],
                    name=f"{transpose.name}_rank4_channels_last",
                    perm=[0, 2, 3, 1],
                ),
                onnx.helper.make_node(
                    "Reshape",
                    [channels_last, final_shape_name],
                    [output_name],
                    name=final_reshape.name,
                    allowzero=1,
                ),
            ]
            _replace_node(graph, [first_reshape, transpose, final_reshape], new_nodes)
            changes.append(
                {
                    "kind": "exact",
                    "nodes": [first_reshape.name, transpose.name, final_reshape.name],
                    "rewrite": "rank-6 patchify->rank<=4 SpaceToDepth pipeline",
                    "block_size": block_size,
                    "channel_patch": channel_patch,
                }
            )
            count += 1
            continue

        # Unpatchify, the exact inverse:
        # [T*H*W,p*p*C] -> reshape [T,H,W,p,p,C]
        # -> transpose [C,T,H,p,W,p] -> [C,T,H*p,W*p].
        if (
            len(first_shape) == 6
            and len(final_shape) == 4
            and permutation == (5, 0, 1, 3, 2, 4)
            and first_shape[3] == first_shape[4]
            and first_shape[3] > 0
            and final_shape
            == (
                first_shape[5],
                first_shape[0],
                first_shape[1] * first_shape[3],
                first_shape[2] * first_shape[4],
            )
        ):
            frames, patch_h_count, patch_w_count, patch_h, patch_w, channels = first_shape
            block_size = patch_h
            prefix = output_name
            rank4_channels_last = _unique_name(graph, f"{prefix}_unpatchify_rank4_channels_last")
            nchw_patches = _unique_name(graph, f"{prefix}_unpatchify_nchw_patches")
            depth_to_space = _unique_name(graph, f"{prefix}_depth_to_space")
            rank4_shape = _add_initializer(
                graph,
                _unique_name(graph, f"{prefix}_unpatchify_rank4_shape"),
                np.asarray(
                    [frames, patch_h_count, patch_w_count, channels * block_size * block_size],
                    dtype=np.int64,
                ),
                onnx,
            )
            new_nodes = [
                onnx.helper.make_node(
                    "Reshape",
                    [source_name, rank4_shape],
                    [rank4_channels_last],
                    name=f"{first_reshape.name}_rank4_channels_last",
                    allowzero=1,
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [rank4_channels_last],
                    [nchw_patches],
                    name=f"{transpose.name}_rank4_nchw",
                    perm=[0, 3, 1, 2],
                ),
                onnx.helper.make_node(
                    "DepthToSpace",
                    [nchw_patches],
                    [depth_to_space],
                    name=f"{final_reshape.name}_depth_to_space",
                    blocksize=block_size,
                    # DCR consumes depth as [p, q, C], the exact inverse of
                    # forward's "thwpqc->cthpwq".
                    mode="DCR",
                ),
                onnx.helper.make_node(
                    "Transpose",
                    [depth_to_space],
                    [output_name],
                    name=final_reshape.name,
                    perm=[1, 0, 2, 3],
                ),
            ]
            _replace_node(graph, [first_reshape, transpose, final_reshape], new_nodes)
            changes.append(
                {
                    "kind": "exact",
                    "nodes": [first_reshape.name, transpose.name, final_reshape.name],
                    "rewrite": "rank-6 unpatchify->rank<=4 DepthToSpace pipeline",
                    "block_size": block_size,
                }
            )
            count += 1

    return count


def _rewrite_causal_where_additive(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    """Replace fixed causal Where masking with a finite additive FP16-safe bias."""
    mask_fill = -10000.0
    if not np.isfinite(np.asarray(mask_fill, dtype=np.float16)):
        raise ValueError(f"Causal mask fill {mask_fill} is not finite in float16")
    count = 0
    producers = _producer_map(graph)
    bias_cache: dict[tuple[tuple[int, ...], str, bytes], str] = {}
    for node in list(graph.node):
        if node.op_type != "Where" or len(node.input) != 3:
            continue
        condition_node = producers.get(node.input[0])
        if condition_node is None:
            continue
        causal_condition = condition_node
        while (
            causal_condition.op_type in {"Cast", "Identity"}
            and causal_condition.input
            and producers.get(causal_condition.input[0]) is not None
        ):
            causal_condition = producers[causal_condition.input[0]]
        if causal_condition.op_type != "Trilu":
            continue
        condition = evaluator.evaluate(node.input[0])
        fill_value = evaluator.evaluate(node.input[1])
        if condition is None or fill_value is None or fill_value.size != 1:
            continue
        fill = fill_value.reshape(-1)[0]
        if not np.isneginf(fill):
            continue
        scores_name = node.input[2]
        score_dtype = np.dtype(fill_value.dtype)
        if not np.issubdtype(score_dtype, np.floating):
            continue

        scores_shape = _resolve_static_tensor_shape(
            graph, scores_name, evaluator, onnx
        )
        if scores_shape is None:
            continue
        try:
            broadcast_mask = np.broadcast_to(condition.astype(bool), scores_shape)
        except ValueError:
            continue
        causal_bias = np.where(broadcast_mask, mask_fill, 0.0).astype(score_dtype)
        bias_key = (scores_shape, score_dtype.str, broadcast_mask.tobytes())
        bias_name = bias_cache.get(bias_key)
        if bias_name is None:
            bias_name = _add_initializer(
                graph,
                _unique_name(graph, f"{node.output[0]}_causal_bias"),
                causal_bias,
                onnx,
            )
            bias_cache[bias_key] = bias_name
        old_name = node.name
        add_node = onnx.helper.make_node(
            "Add",
            [scores_name, bias_name],
            [node.output[0]],
            name=old_name.replace("masked_fill", "causal_add") if old_name else old_name,
        )
        _replace_node(graph, [node], [add_node])
        changes.append(
            {
                "kind": "finite_score_softmax_equivalent_for_fixed_configuration",
                "node": old_name,
                "rewrite": "Trilu+Where->Add(scores, finite causal bias)",
                "scores_shape": list(scores_shape),
                "mask_fill_value": mask_fill,
                "masked_element_count": int(np.count_nonzero(broadcast_mask)),
                "shared_bias_initializer": bias_name,
            }
        )
        count += 1
    return count


def _prune_unused(graph: Any) -> None:
    producer_indices = {
        output: index
        for index, node in enumerate(graph.node)
        for output in node.output
        if output
    }
    required_tensors = {output.name for output in graph.output}
    required_nodes: set[int] = set()
    stack = list(required_tensors)
    while stack:
        tensor = stack.pop()
        node_index = producer_indices.get(tensor)
        if node_index is None or node_index in required_nodes:
            continue
        required_nodes.add(node_index)
        node = graph.node[node_index]
        for input_name in node.input:
            if input_name and input_name not in required_tensors:
                required_tensors.add(input_name)
                stack.append(input_name)
    kept_nodes = [node for index, node in enumerate(graph.node) if index in required_nodes]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    kept_initializers = [initializer for initializer in graph.initializer if initializer.name in required_tensors]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    kept_value_info = [value for value in graph.value_info if value.name in required_tensors]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)


def _deduplicate_node_names(graph: Any) -> int:
    """Make diagnostic node names unique without changing graph semantics."""
    used: set[str] = set()
    renamed = 0
    for index, node in enumerate(graph.node):
        base = node.name or f"node_{index}"
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        if node.name != candidate:
            node.name = candidate
            renamed += 1
        used.add(candidate)
    return renamed


def _infer_shapes_in_place(model_path: Path, onnx: Any) -> None:
    """Infer intermediate tensor metadata without loading external tensor data."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{model_path.name}.shape-inference-",
        suffix=model_path.suffix,
        dir=model_path.parent,
        delete=False,
    ) as temporary_file:
        inferred_path = Path(temporary_file.name)
    inferred_path.unlink()
    try:
        onnx.shape_inference.infer_shapes_path(
            str(model_path),
            str(inferred_path),
            check_type=True,
            strict_mode=True,
            data_prop=False,
        )
        onnx.checker.check_model(str(inferred_path))
        inferred_path.replace(model_path)
    finally:
        inferred_path.unlink(missing_ok=True)


def _ort_input_dtype(type_name: str) -> Any:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    if type_name == "tensor(bfloat16)":
        try:
            import ml_dtypes
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install `ml_dtypes` to verify a BF16 ONNX model") from exc
        return ml_dtypes.bfloat16
    if type_name not in mapping:
        raise TypeError(f"Unsupported ONNX Runtime verification input type: {type_name}")
    return mapping[type_name]


def _fixed_ort_shape(value: Any) -> tuple[int, ...]:
    shape = value.shape
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError(f"Equivalence verification requires fixed positive input shape, got {value.name}: {shape}")
    return tuple(int(dim) for dim in shape)


def _make_verification_inputs(
    session: Any,
    *,
    domain_id: int,
    seed: int,
    prompt_embedding_table_path: Path,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        shape = _fixed_ort_shape(value)
        dtype = _ort_input_dtype(value.type)
        if value.name == "prompt_embeddings":
            table = np.load(prompt_embedding_table_path, mmap_mode="r")
            if table.ndim != 2 or len(shape) != 2 or table.shape[1] != shape[1]:
                raise ValueError(
                    "Prompt embedding table/input shape mismatch: "
                    f"table={table.shape}, input={shape}"
                )
            # The pre-externalization verifier used all-zero prompt token IDs.
            # Repeating table row zero preserves that exact valid interface.
            inputs[value.name] = np.broadcast_to(table[0], shape).astype(dtype, copy=True)
        elif value.name == "prompt_token_ids":
            inputs[value.name] = np.zeros(shape, dtype=dtype)
        elif value.name == "action_domain_id":
            inputs[value.name] = np.full(shape, domain_id, dtype=dtype)
        elif "timestep" in value.name:
            inputs[value.name] = np.full(shape, 0.5, dtype=dtype)
        elif np.issubdtype(np.dtype(dtype), np.integer):
            inputs[value.name] = np.zeros(shape, dtype=dtype)
        else:
            inputs[value.name] = rng.standard_normal(shape).astype(dtype)
    return inputs


def _verify_rewrite_equivalence(
    reference_path: Path,
    output_path: Path,
    *,
    domain_id: int,
    prompt_embedding_table_path: Path,
    providers: list[str] | None,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `onnxruntime` or `onnxruntime-gpu` for equivalence verification") from exc

    available = set(ort.get_available_providers())
    requested = providers or (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    missing = [provider for provider in requested if provider not in available]
    if missing:
        raise RuntimeError(
            f"Requested ONNX Runtime providers are unavailable: {missing}; "
            f"available={sorted(available)}"
        )

    cuda_library_preload = None
    if "CUDAExecutionProvider" in requested:
        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is not None:
            preload_dlls(directory="")
            cuda_library_preload = "onnxruntime.preload_dlls(directory='')"
        else:
            try:
                import torch  # noqa: F401
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "CUDA equivalence verification requires discoverable CUDA/cuDNN "
                    "libraries. Install PyTorch or use ONNX Runtime >= 1.21 with "
                    "preload_dlls support."
                ) from exc
            cuda_library_preload = "import torch"

    original_session = ort.InferenceSession(str(reference_path), providers=requested)
    rewritten_session = ort.InferenceSession(str(output_path), providers=requested)
    original_inputs = _make_verification_inputs(
        original_session,
        domain_id=domain_id,
        seed=seed,
        prompt_embedding_table_path=prompt_embedding_table_path,
    )

    rewritten_inputs: dict[str, np.ndarray] = {}
    rewritten_input_meta = {
        value.name: value for value in rewritten_session.get_inputs()
    }
    for name, rewritten_value in rewritten_input_meta.items():
        if name == "prompt_embeddings":
            embeddings = original_inputs.get(name)
            if embeddings is None:
                raise ValueError("Verification reference graph has no prompt_embeddings input")
            rewritten_inputs[name] = embeddings
        elif name == "video_latent":
            video = original_inputs[name]
            if video.ndim != 5 or video.shape[0] != 1:
                raise ValueError(f"Expected original video_latent [1,C,T,H,W], got {video.shape}")
            expected_shape = _fixed_ort_shape(rewritten_value)
            if len(expected_shape) == 5:
                rewritten_inputs[name] = video
            elif len(expected_shape) == 4:
                rewritten_inputs[name] = video[0]
            else:
                raise ValueError(
                    f"Expected rewritten video_latent rank 4 or 5, got {expected_shape}"
                )
        elif name in original_inputs:
            rewritten_inputs[name] = original_inputs[name]
        else:
            raise KeyError(f"Cannot map rewritten ONNX input {name!r} to the original graph")

    original_names = [value.name for value in original_session.get_outputs()]
    rewritten_names_ordered = [value.name for value in rewritten_session.get_outputs()]
    original_outputs = dict(
        zip(original_names, original_session.run(original_names, original_inputs))
    )
    rewritten_outputs = dict(
        zip(rewritten_names_ordered, rewritten_session.run(rewritten_names_ordered, rewritten_inputs))
    )

    output_metrics: dict[str, Any] = {}
    all_close = True
    for name, original in original_outputs.items():
        if name not in rewritten_outputs:
            raise KeyError(f"Rewritten graph is missing output {name!r}")
        rewritten = rewritten_outputs[name]
        comparable_original = original
        if original.ndim == rewritten.ndim + 1 and original.shape[0] == 1:
            comparable_original = original[0]
        if comparable_original.shape != rewritten.shape:
            raise ValueError(
                f"Output shape mismatch for {name}: original={original.shape}, "
                f"comparable={comparable_original.shape}, rewritten={rewritten.shape}"
            )
        original_fp32 = comparable_original.astype(np.float32)
        rewritten_fp32 = rewritten.astype(np.float32)
        original_finite = np.isfinite(original_fp32)
        rewritten_finite = np.isfinite(rewritten_fp32)
        original_nonfinite_count = int(original_fp32.size - np.count_nonzero(original_finite))
        rewritten_nonfinite_count = int(rewritten_fp32.size - np.count_nonzero(rewritten_finite))
        finite = original_nonfinite_count == 0 and rewritten_nonfinite_count == 0
        difference = np.abs(original_fp32 - rewritten_fp32)
        finite_difference = difference[np.isfinite(difference)]
        close = finite and bool(np.allclose(original_fp32, rewritten_fp32, atol=atol, rtol=rtol))
        all_close &= close
        output_metrics[name] = {
            "original_shape": list(original.shape),
            "rewritten_shape": list(rewritten.shape),
            "original_nonfinite_count": original_nonfinite_count,
            "rewritten_nonfinite_count": rewritten_nonfinite_count,
            "finite": finite,
            "max_abs_error": float(finite_difference.max(initial=0.0)),
            "mean_abs_error": float(finite_difference.mean()) if finite_difference.size else None,
            "allclose": close,
        }

    result = {
        "enabled": True,
        "providers": requested,
        "cuda_library_preload": cuda_library_preload,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "allclose": all_close,
        "blocking": False,
        "outputs": output_metrics,
    }
    return result


def _apply_rewrites(
    input_path: Path,
    graph: Any,
    *,
    domain_id: int,
    prompt_embedding_table_path: Path,
    onnx: Any,
    lower_vision_ranks: bool,
    decompose_gemm: bool = False,
    materialized_transposed_weights: set[str] | None = None,
    keep_causal_where: bool = False,
    keep_scatter_elements: bool = False,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    inlined_external_constants = _inline_small_external_constants(input_path, graph, onnx)
    _freeze_action_domain(graph, domain_id, onnx, changes)
    _externalize_prompt_embedding(input_path, graph, prompt_embedding_table_path, onnx, changes)
    evaluator = ConstantEvaluator(input_path, graph, onnx)
    counts = {
        "inlined_external_constants": inlined_external_constants,
        "fixed_vision_batch_io": (
            _rewrite_fixed_vision_batch_io(graph, evaluator, onnx, changes)
            if lower_vision_ranks
            else 0
        ),
        # The dynamo exporter emits these fixed permutations as unary Einsum,
        # while the rank-lowering matcher operates on the equivalent Transpose.
        "einsum": _rewrite_unary_einsum(graph, onnx, changes),
        "rank6_patchify": (
            _rewrite_patchify_rank6(graph, evaluator, onnx, changes)
            if lower_vision_ranks
            else 0
        ),
        "constant_of_shape": _rewrite_constant_of_shape(input_path, graph, onnx, changes),
        "constant_gather": _fold_constant_gathers(input_path, graph, evaluator, onnx, changes),
        "scalar_gather": _rewrite_scalar_gathers(graph, evaluator, onnx, changes),
        "scatter_elements": (
            0 if keep_scatter_elements else _rewrite_scatter_elements(graph, evaluator, onnx, changes)
        ),
        "causal_where": (
            0 if keep_causal_where else _rewrite_causal_where_additive(graph, evaluator, onnx, changes)
        ),
        "gemm": (
            _decompose_gemm_nodes(
                graph,
                onnx,
                changes,
                materialized_transposed_weights=materialized_transposed_weights,
            )
            if decompose_gemm
            else 0
        ),
    }
    counts["nd_indices"] = _normalize_nd_indices(graph, onnx, changes)
    _prune_unused(graph)
    counts["deduplicated_node_names"] = _deduplicate_node_names(graph)
    return counts, changes


def _prepare_verification_reference(
    input_path: Path,
    graph: Any,
    *,
    domain_id: int,
    prompt_embedding_table_path: Path,
    onnx: Any,
) -> list[dict[str, Any]]:
    """Align graph interfaces while preserving the original compute graph.

    ONNX Runtime requires int64 GatherND/ScatterND indices and unique diagnostic
    node names. Those changes are exact and do not replace any target backend
    operator, so the resulting graph remains a useful reference for all actual
    compatibility rewrites.
    """
    changes: list[dict[str, Any]] = []
    _freeze_action_domain(graph, domain_id, onnx, changes)
    _externalize_prompt_embedding(input_path, graph, prompt_embedding_table_path, onnx, changes)
    _normalize_nd_indices(graph, onnx, changes)
    _prune_unused(graph)
    _deduplicate_node_names(graph)
    return changes


def rewrite_model(
    input_path: Path,
    output_path: Path,
    *,
    domain_id: int,
    prompt_embedding_table_path: Path,
    verify_equivalence: bool = False,
    verification_providers: list[str] | None = None,
    verification_seed: int = 0,
    verification_atol: float = 1e-2,
    verification_rtol: float = 1e-2,
    keep_causal_where: bool = False,
    keep_vision_ranks: bool = False,
    keep_scatter_elements: bool = False,
    decompose_gemm: bool = False,
    materialize_gemm_weight_transposes: bool = False,
    transposed_external_data_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `onnx` to rewrite an ONNX model") from exc

    if input_path.parent.resolve() != output_path.parent.resolve():
        raise ValueError("Input and output must share a directory so existing external-data references remain valid")
    if materialize_gemm_weight_transposes and not decompose_gemm:
        raise ValueError("Physical Gemm weight transposition requires decompose_gemm=True")
    if materialize_gemm_weight_transposes and transposed_external_data_path is None:
        raise ValueError(
            "Physical Gemm weight transposition requires a distinct transposed_external_data_path"
        )
    if not materialize_gemm_weight_transposes and transposed_external_data_path is not None:
        raise ValueError(
            "transposed_external_data_path is only valid with materialize_gemm_weight_transposes=True"
        )
    model = onnx.load_model(str(input_path), load_external_data=False)
    graph = model.graph
    materialized_transposed_weights: set[str] = set()
    materialized_weight_report: dict[str, Any] = {"enabled": False}
    if materialize_gemm_weight_transposes:
        assert transposed_external_data_path is not None
        materialized_transposed_weights, materialized_weight_report = (
            _materialize_gemm_weight_transposes(
                input_path,
                model,
                transposed_external_data_path,
                onnx,
            )
        )
    counts, changes = _apply_rewrites(
        input_path,
        graph,
        domain_id=domain_id,
        prompt_embedding_table_path=prompt_embedding_table_path,
        onnx=onnx,
        lower_vision_ranks=not keep_vision_ranks,
        keep_causal_where=keep_causal_where,
        keep_scatter_elements=keep_scatter_elements,
        decompose_gemm=decompose_gemm,
        materialized_transposed_weights=materialized_transposed_weights,
    )
    onnx.save_model(model, str(output_path))
    _infer_shapes_in_place(output_path, onnx)
    onnx.checker.check_model(str(output_path))

    compatibility = inspect_model(output_path, DEFAULT_TARGET_OPS, max_rank=4)
    verification: dict[str, Any] = {"enabled": False}
    if verify_equivalence:
        reference_path = output_path.with_name(f"{output_path.stem}.verification_reference{output_path.suffix}")
        reference_model = onnx.load_model(str(input_path), load_external_data=False)
        _prepare_verification_reference(
            input_path,
            reference_model.graph,
            domain_id=domain_id,
            prompt_embedding_table_path=prompt_embedding_table_path,
            onnx=onnx,
        )
        onnx.save_model(reference_model, str(reference_path))
        onnx.checker.check_model(str(reference_path))
        verification_completed = False
        try:
            verification = _verify_rewrite_equivalence(
                reference_path,
                output_path,
                domain_id=domain_id,
                prompt_embedding_table_path=prompt_embedding_table_path,
                providers=verification_providers,
                seed=verification_seed,
                atol=verification_atol,
                rtol=verification_rtol,
            )
            verification["reference_model"] = str(reference_path.resolve())
            verification["reference_scope"] = (
                "Original compute graph with only fixed-interface alignment, "
                "ORT-required ND index casts, pruning, and unique node names."
            )
            verification_completed = True
        finally:
            if verification_completed:
                reference_path.unlink(missing_ok=True)
    remaining = compatibility["summary"]["target_node_count"]
    remaining_high_rank_nodes = compatibility["summary"]["high_rank_node_count"]
    remaining_high_rank_graph_io = compatibility["summary"]["high_rank_graph_io_count"]
    report = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "prompt_embedding_table_path": str(prompt_embedding_table_path.resolve()),
        "domain_id": domain_id,
        "diagnostic_keep_causal_where": keep_causal_where,
        "diagnostic_keep_vision_ranks": keep_vision_ranks,
        "diagnostic_keep_scatter_elements": keep_scatter_elements,
        "decompose_gemm": decompose_gemm,
        "materialize_gemm_weight_transposes": materialize_gemm_weight_transposes,
        "materialized_gemm_weights": materialized_weight_report,
        "counts": counts,
        "remaining_target_nodes": remaining,
        "remaining_high_rank_nodes": remaining_high_rank_nodes,
        "remaining_high_rank_graph_io": remaining_high_rank_graph_io,
        "changes": changes,
        "compatibility_summary": compatibility["summary"],
        "verification": verification,
        "equivalence": {
            "exact_rewrites": (
                "Einsum, ConstantOfShape, constant/scalar Gather, validated ScatterElements patterns, "
                "and enabled default-scale Gemm decomposition."
            ),
            "fixed_configuration": f"action_domain_id is frozen to {domain_id}.",
            "interface_requirement": (
                "prompt_embeddings must equal the original embedding table lookup for prompt_token_ids."
            ),
            "causal_mask": (
                "For the fixed causal layout, Add applies a shared 0/-10000 finite bias. "
                "This matches masked softmax for finite deployment scores but does not preserve "
                "Where branch isolation for NaN/Inf masked scores."
            ),
            "numerical_validation": (
                "Optional ORT comparison is recorded as a non-blocking diagnostic; "
                "deployment validation remains required."
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--domain-id", type=int, default=8, help="Fixed action domain; DROID is 8.")
    parser.add_argument("--prompt-embedding-table-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument(
        "--decompose-gemm",
        action="store_true",
        help="Rewrite default-scale Gemm as MatMul and add bias only when a third input exists.",
    )
    parser.add_argument(
        "--materialize-gemm-weight-transposes",
        action="store_true",
        help=(
            "Physically transpose transB=1 Gemm weight initializers in a separate "
            "external-data file so the replacement MatMul has no weight Transpose node."
        ),
    )
    parser.add_argument(
        "--transposed-external-data-path",
        type=Path,
        help=(
            "Distinct external-data output for physically transposed Gemm weights; "
            "required with --materialize-gemm-weight-transposes."
        ),
    )
    parser.add_argument(
        "--verification-provider",
        action="append",
        dest="verification_providers",
        default=None,
        help="ONNX Runtime provider in priority order; repeat for fallbacks.",
    )
    parser.add_argument("--verification-seed", type=int, default=0)
    parser.add_argument("--verification-atol", type=float, default=1e-2)
    parser.add_argument("--verification-rtol", type=float, default=1e-2)
    parser.add_argument(
        "--keep-causal-where",
        action="store_true",
        help=(
            "Diagnostic mode: retain causal Where nodes while applying all other rewrites. "
            "The output is not target-compatible while Where remains."
        ),
    )
    parser.add_argument(
        "--keep-vision-ranks",
        action="store_true",
        help=(
            "Diagnostic mode: retain rank-5 vision I/O and rank-6 patch layouts "
            "while applying other rewrites."
        ),
    )
    parser.add_argument(
        "--keep-scatter-elements",
        action="store_true",
        help="Diagnostic mode: retain ScatterElements nodes while applying other rewrites.",
    )
    args = parser.parse_args()

    if not args.input_path.is_file():
        parser.error(f"input model does not exist: {args.input_path}")
    if args.input_path.resolve() == args.output_path.resolve():
        parser.error("output_path must differ from input_path")
    if args.materialize_gemm_weight_transposes and not args.decompose_gemm:
        parser.error("--materialize-gemm-weight-transposes requires --decompose-gemm")
    if args.materialize_gemm_weight_transposes and args.transposed_external_data_path is None:
        parser.error(
            "--materialize-gemm-weight-transposes requires --transposed-external-data-path"
        )
    if not args.materialize_gemm_weight_transposes and args.transposed_external_data_path is not None:
        parser.error(
            "--transposed-external-data-path requires --materialize-gemm-weight-transposes"
        )

    prompt_embedding_table_path = args.prompt_embedding_table_path or args.output_path.with_suffix(
        args.output_path.suffix + ".prompt_embedding.npy"
    )
    try:
        report = rewrite_model(
            args.input_path,
            args.output_path,
            domain_id=args.domain_id,
            prompt_embedding_table_path=prompt_embedding_table_path,
            verify_equivalence=args.verify_equivalence,
            verification_providers=args.verification_providers,
            verification_seed=args.verification_seed,
            verification_atol=args.verification_atol,
            verification_rtol=args.verification_rtol,
            keep_causal_where=args.keep_causal_where,
            keep_vision_ranks=args.keep_vision_ranks,
            keep_scatter_elements=args.keep_scatter_elements,
            decompose_gemm=args.decompose_gemm,
            materialize_gemm_weight_transposes=args.materialize_gemm_weight_transposes,
            transposed_external_data_path=args.transposed_external_data_path,
        )
    except Exception:
        if args.materialize_gemm_weight_transposes and args.transposed_external_data_path is not None:
            args.transposed_external_data_path.unlink(missing_ok=True)
        raise
    report_path = args.report_path or args.output_path.with_suffix(args.output_path.suffix + ".rewrite.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Rewritten ONNX saved to: {args.output_path}")
    print(f"Prompt embedding table saved to: {prompt_embedding_table_path}")
    print(f"Rewrite report saved to: {report_path}")
    print("Equivalence conditions:")
    print(f"  action_domain_id is fixed to {args.domain_id}")
    print("  prompt_embeddings must come from the original prompt embedding table")
    print("  causal masking uses a shared finite 0/-10000 additive bias")
    print("  masked attention scores must be finite before the additive bias")
    if report["verification"]["enabled"]:
        print("ONNX Runtime equivalence verification:")
        for name, metrics in report["verification"]["outputs"].items():
            print(
                f"  {name}: allclose={metrics['allclose']} "
                f"finite={metrics['finite']} "
                f"nonfinite=({metrics['original_nonfinite_count']},"
                f"{metrics['rewritten_nonfinite_count']}) "
                f"max_abs_error={metrics['max_abs_error']:.8g} "
                f"mean_abs_error={metrics['mean_abs_error']}"
            )
        if not report["verification"]["allclose"]:
            print(
                "WARNING: Step 5 numerical equivalence verification did not pass; "
                "the finite additive causal mask is an intentional approximation. "
                "The rewritten ONNX is retained and downstream steps will continue. "
                f"See {report_path} for error metrics.",
                file=sys.stderr,
            )
    if args.keep_causal_where:
        print(
            "Diagnostic mode: causal Where nodes were retained; "
            "this output is not target-compatible."
        )
    if args.keep_vision_ranks:
        print(
            "Diagnostic mode: high-rank vision layouts were retained; "
            "this output is not target-compatible."
        )
    if args.keep_scatter_elements:
        print(
            "Diagnostic mode: ScatterElements nodes were retained; "
            "this output is not target-compatible."
        )
    op_counts = report["compatibility_summary"]["op_counts"]
    blocking_target_nodes = report["remaining_target_nodes"]
    if args.keep_causal_where:
        blocking_target_nodes -= op_counts.get("Where", 0)
    if args.keep_scatter_elements:
        blocking_target_nodes -= op_counts.get("ScatterElements", 0)
    blocking_high_rank_nodes = (
        0 if args.keep_vision_ranks else report["remaining_high_rank_nodes"]
    )
    blocking_high_rank_graph_io = (
        0 if args.keep_vision_ranks else report["remaining_high_rank_graph_io"]
    )
    if (
        blocking_target_nodes
        or blocking_high_rank_nodes
        or blocking_high_rank_graph_io
    ):
        remaining_counts = {op: op_counts.get(op, 0) for op in DEFAULT_TARGET_OPS}
        raise RuntimeError(
            f"Rewritten model still has {report['remaining_target_nodes']} target nodes "
            f"{remaining_counts}, {report['remaining_high_rank_nodes']} high-rank nodes, and "
            f"{report['remaining_high_rank_graph_io']} high-rank graph I/O tensors; "
            f"equivalence_allclose={report['verification'].get('allclose', 'not-run')}. "
            f"See {report_path}."
        )


if __name__ == "__main__":
    main()
