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
* causal Trilu + Where(mask, -inf, scores) -> Add(scores, causal_bias)
* prompt token embedding Gather -> a ``prompt_embeddings`` graph input
* fixed DROID action domain -> constant domain ID 8

The causal-mask rewrite is equivalent for finite attention scores. The prompt
rewrite is interface-equivalent when the host supplies embeddings produced by
the exported model's original embedding table.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.scripts.inspect_action_policy_onnx import DEFAULT_TARGET_OPS, inspect_model


def _initializer_map(graph: Any) -> dict[str, Any]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _producer_map(graph: Any) -> dict[str, Any]:
    return {output: node for node in graph.node for output in node.output if output}


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
                upper = (
                    int(values[1].reshape(-1)[0])
                    if len(values) > 1
                    else int(_attribute(node, "upper", self.onnx, 1))
                )
                result = np.triu(values[0]) if upper else np.tril(values[0])
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


def _rewrite_constant_of_shape(graph: Any, onnx: Any, changes: list[dict[str, Any]]) -> int:
    count = 0
    for node in graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        tensor = _attribute(node, "value", onnx)
        if tensor is None:
            scalar = np.asarray([0], dtype=np.float32)
        else:
            scalar = onnx.numpy_helper.to_array(tensor)
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
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with table_path.open("wb") as table_file:
            np.save(table_file, embedding_table, allow_pickle=False)

        output_name = node.output[0]
        output_info = next((value for value in graph.value_info if value.name == output_name), None)
        if output_info is None:
            raise RuntimeError(f"Missing value_info for prompt embedding output {output_name!r}")
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


def _rewrite_causal_where_safe(
    graph: Any,
    evaluator: ConstantEvaluator,
    onnx: Any,
    changes: list[dict[str, Any]],
) -> int:
    """Wrapper retaining score input while replacing Where nodes."""
    count = 0
    producers = _producer_map(graph)
    for node in list(graph.node):
        if node.op_type != "Where" or len(node.input) != 3:
            continue
        condition_node = producers.get(node.input[0])
        if condition_node is None or condition_node.op_type != "Trilu":
            continue
        condition = evaluator.evaluate(node.input[0])
        fill_value = evaluator.evaluate(node.input[1])
        if condition is None or fill_value is None or fill_value.size != 1:
            continue
        fill = fill_value.reshape(-1)[0]
        if not np.isneginf(fill):
            continue
        scores_name = node.input[2]
        bias = np.where(condition.astype(bool), fill, np.asarray(0, dtype=fill_value.dtype))
        bias_name = _add_initializer(graph, _unique_name(graph, f"{node.output[0]}_causal_bias"), bias, onnx)
        old_name = node.name
        node.op_type = "Add"
        node.name = old_name.replace("masked_fill", "causal_add") if old_name else old_name
        del node.input[:]
        node.input.extend([scores_name, bias_name])
        changes.append(
            {
                "kind": "finite_input_equivalent",
                "node": old_name,
                "rewrite": "Trilu+Where->Add(causal_bias)",
                "requirement": "Attention scores must be finite before masking.",
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


def rewrite_model(
    input_path: Path,
    output_path: Path,
    *,
    domain_id: int,
    prompt_embedding_table_path: Path,
) -> dict[str, Any]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `onnx` to rewrite an ONNX model") from exc

    if input_path.parent.resolve() != output_path.parent.resolve():
        raise ValueError("Input and output must share a directory so existing external-data references remain valid")
    model = onnx.load_model(str(input_path), load_external_data=False)
    graph = model.graph
    changes: list[dict[str, Any]] = []

    _freeze_action_domain(graph, domain_id, onnx, changes)
    _externalize_prompt_embedding(input_path, graph, prompt_embedding_table_path, onnx, changes)
    evaluator = ConstantEvaluator(input_path, graph, onnx)
    counts = {
        "einsum": _rewrite_unary_einsum(graph, onnx, changes),
        "constant_of_shape": _rewrite_constant_of_shape(graph, onnx, changes),
        "constant_gather": _fold_constant_gathers(input_path, graph, evaluator, onnx, changes),
        "scalar_gather": _rewrite_scalar_gathers(graph, evaluator, onnx, changes),
        "scatter_elements": _rewrite_scatter_elements(graph, evaluator, onnx, changes),
        "causal_where": _rewrite_causal_where_safe(graph, evaluator, onnx, changes),
    }
    _prune_unused(graph)
    onnx.save_model(model, str(output_path))
    onnx.checker.check_model(str(output_path))

    compatibility = inspect_model(output_path, DEFAULT_TARGET_OPS, max_rank=4)
    remaining = compatibility["summary"]["target_node_count"]
    report = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "prompt_embedding_table_path": str(prompt_embedding_table_path.resolve()),
        "domain_id": domain_id,
        "counts": counts,
        "remaining_target_nodes": remaining,
        "changes": changes,
        "compatibility_summary": compatibility["summary"],
        "equivalence": {
            "exact_rewrites": (
                "Einsum, ConstantOfShape, constant/scalar Gather, and validated ScatterElements patterns."
            ),
            "fixed_configuration": f"action_domain_id is frozen to {domain_id}.",
            "interface_requirement": (
                "prompt_embeddings must equal the original embedding table lookup for prompt_token_ids."
            ),
            "finite_input_requirement": "Attention scores must be finite before additive causal masking.",
            "numerical_validation": "Not performed by this structural rewrite; compare outputs on deployment inputs.",
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
    args = parser.parse_args()

    if not args.input_path.is_file():
        parser.error(f"input model does not exist: {args.input_path}")
    if args.input_path.resolve() == args.output_path.resolve():
        parser.error("output_path must differ from input_path")

    prompt_embedding_table_path = args.prompt_embedding_table_path or args.output_path.with_suffix(
        args.output_path.suffix + ".prompt_embedding.npy"
    )
    report = rewrite_model(
        args.input_path,
        args.output_path,
        domain_id=args.domain_id,
        prompt_embedding_table_path=prompt_embedding_table_path,
    )
    report_path = args.report_path or args.output_path.with_suffix(args.output_path.suffix + ".rewrite.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Rewritten ONNX saved to: {args.output_path}")
    print(f"Prompt embedding table saved to: {prompt_embedding_table_path}")
    print(f"Rewrite report saved to: {report_path}")
    print("Equivalence conditions:")
    print(f"  action_domain_id is fixed to {args.domain_id}")
    print("  prompt_embeddings must come from the original prompt embedding table")
    print("  attention scores must be finite before causal masking")
    if report["remaining_target_nodes"]:
        op_counts = report["compatibility_summary"]["op_counts"]
        remaining_counts = {op: op_counts.get(op, 0) for op in DEFAULT_TARGET_OPS}
        raise RuntimeError(
            f"Rewritten model still has {report['remaining_target_nodes']} target nodes: {remaining_counts}. "
            f"See {report_path}."
        )


if __name__ == "__main__":
    main()
