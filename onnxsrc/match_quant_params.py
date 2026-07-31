#!/usr/bin/env python3
"""Compare encrypted DOPT quant-parameter entry names with ONNX node names."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def decrypt(path: Path) -> dict:
    with os.fdopen(os.open(path, os.O_RDONLY, 0o700), "rb") as file:
        encrypted = np.frombuffer(file.read(), dtype=np.uint8)
    mask = np.frombuffer(b"ddkv320", dtype=np.uint8)
    decrypted = np.bitwise_xor(encrypted, np.resize(mask, encrypted.shape)).tobytes()
    return json.loads(decrypted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--quant-params", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    import onnx

    model = onnx.load_model(str(args.onnx), load_external_data=False)
    node_names = {node.name for node in model.graph.node if node.name}
    params = decrypt(args.quant_params).get("ModelLightWeightParameter", [])
    param_names = {str(entry.get("name", "")) for entry in params if entry.get("name")}
    missing_from_onnx = sorted(param_names - node_names)
    onnx_names_without_params = sorted(node_names - param_names)
    report = {
        "onnx": str(args.onnx.resolve()),
        "quant_params": str(args.quant_params.resolve()),
        "onnx_named_node_count": len(node_names),
        "quant_parameter_entry_count": len(param_names),
        "matched_count": len(node_names & param_names),
        "missing_from_onnx_count": len(missing_from_onnx),
        "missing_from_onnx": missing_from_onnx,
        "onnx_names_without_quant_params_count": len(onnx_names_without_params),
        "onnx_names_without_quant_params": onnx_names_without_params,
    }
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"matched={report['matched_count']} "
        f"missing_from_onnx={report['missing_from_onnx_count']} "
        f"onnx_without_params={report['onnx_names_without_quant_params_count']}"
    )
    if missing_from_onnx:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
