#!/usr/bin/env python3
"""Audit ONNX names, shape metadata, ranks, graph I/O, and target operators."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from cosmos_framework.scripts.inspect_action_policy_onnx import DEFAULT_TARGET_OPS, inspect_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--allow-unknown-rank", action="store_true")
    parser.add_argument("--allow-target-ops", action="store_true")
    args = parser.parse_args()

    import onnx

    onnx.checker.check_model(str(args.model_path))
    model = onnx.load_model(str(args.model_path), load_external_data=False)
    node_names = [node.name for node in model.graph.node]
    name_counts = Counter(node_names)
    empty_node_indices = [index for index, name in enumerate(node_names) if not name]
    duplicate_node_names = sorted(name for name, count in name_counts.items() if name and count > 1)
    producers: dict[str, list[str]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        node_label = node.name or f"{node.op_type}_{index}"
        for output in node.output:
            if output:
                producers[output].append(node_label)
    duplicate_tensor_producers = {
        name: node_labels for name, node_labels in sorted(producers.items()) if len(node_labels) > 1
    }
    compatibility = inspect_model(args.model_path, DEFAULT_TARGET_OPS, args.max_rank)
    summary = compatibility["summary"]
    report = {
        "model_path": str(args.model_path.resolve()),
        "onnx_checker": "passed",
        "node_count": len(model.graph.node),
        "empty_node_name_count": len(empty_node_indices),
        "empty_node_indices": empty_node_indices,
        "duplicate_node_name_count": len(duplicate_node_names),
        "duplicate_node_names": duplicate_node_names,
        "duplicate_tensor_producer_count": len(duplicate_tensor_producers),
        "duplicate_tensor_producers": duplicate_tensor_producers,
        "graph_io": compatibility["graph_io"],
        "target_node_count": summary["target_node_count"],
        "target_nodes": compatibility["target_nodes"],
        "high_rank_node_count": summary["high_rank_node_count"],
        "high_rank_nodes": compatibility["high_rank_nodes"],
        "high_rank_graph_io_count": summary["high_rank_graph_io_count"],
        "high_rank_graph_io": compatibility["high_rank_graph_io"],
        "unknown_rank_tensor_count": summary["unknown_rank_tensor_count"],
        "unknown_rank_tensors": compatibility["unknown_rank_tensors"],
        "op_counts": summary["op_counts"],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"nodes={report['node_count']} empty_names={report['empty_node_name_count']} "
        f"duplicate_names={report['duplicate_node_name_count']} "
        f"duplicate_producers={report['duplicate_tensor_producer_count']}"
    )
    print(
        f"target_nodes={report['target_node_count']} high_rank_nodes={report['high_rank_node_count']} "
        f"high_rank_io={report['high_rank_graph_io_count']} "
        f"unknown_rank={report['unknown_rank_tensor_count']}"
    )
    print(f"Report: {args.report_path}")

    blocking = (
        report["empty_node_name_count"]
        or report["duplicate_node_name_count"]
        or report["duplicate_tensor_producer_count"]
        or report["high_rank_node_count"]
        or report["high_rank_graph_io_count"]
        or (report["unknown_rank_tensor_count"] and not args.allow_unknown_rank)
        or (report["target_node_count"] and not args.allow_target_ops)
    )
    if blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
