#!/usr/bin/env python3
"""Export a calibrated DOPT Cosmos3 Policy denoiser to fixed-layout FP32 ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.onnxsrc.cosmos3_quantize import _import_dopt, _load_fixed_policy
from cosmos_framework.scripts.export_action_policy_onnx import (
    PolicyDenoiserOnnxWrapper,
    _install_onnx_attention,
    _shape_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dopt-sim-path", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, required=True)
    parser.add_argument("--fakequant-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--condition-vision-frames", default="0")
    parser.add_argument("--condition-action-frames", default="0")
    parser.add_argument("--prompt", default="Pick up the object and place it in the target location.")
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--conditioning-fps", type=float, default=5.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--domain-name", default="droid_lerobot")
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument(
        "--exporter",
        choices=("legacy", "dynamo"),
        default="legacy",
        help="DOPT uses tensor-valued Python control flags, so legacy tracing is the default.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Cosmos3 Policy ONNX export requires CUDA")
    if not args.quant_config.is_file():
        raise FileNotFoundError(f"DOPT config does not exist: {args.quant_config}")
    if not args.fakequant_weight.is_file():
        raise FileNotFoundError(f"Fake-quant state dict does not exist: {args.fakequant_weight}")

    _generate_config, _generate_params, optimize_model, set_calibrate_state, set_quant_state = _import_dopt(
        args.dopt_sim_path
    )
    service, export_args, packed, inputs = _load_fixed_policy(args)
    quant_net = optimize_model(service.model.net.float().eval(), str(args.quant_config)).eval()
    state_dict = torch.load(args.fakequant_weight, map_location="cpu")
    incompatible = quant_net.load_state_dict(state_dict, strict=False)
    missing_model_keys = [
        key for key in incompatible.missing_keys if ".quant_op." not in key
    ]
    if missing_model_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Fake-quant state dict does not match the reconstructed model weights: "
            f"missing_model_keys={missing_model_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    if incompatible.missing_keys:
        print(
            "DOPT quantizer state is absent from the fake-quant checkpoint and will "
            f"be reconstructed from the quant config and model weights: {len(incompatible.missing_keys)} key(s)"
        )
    set_quant_state(quant_net, weight_state=True, input_state=True)
    set_calibrate_state(quant_net, False)
    quant_net = quant_net.to(device=torch.device("cuda"), dtype=torch.float32).eval()
    _install_onnx_attention(quant_net)
    wrapper = PolicyDenoiserOnnxWrapper(quant_net, packed).float().eval()
    float_inputs = tuple(value.float() if value.is_floating_point() else value for value in inputs)
    input_names = (
        "prompt_token_ids",
        "video_latent",
        "action_latent",
        "vision_timestep",
        "action_timestep",
        "action_domain_id",
    )
    output_names = ("vision_velocity", "action_velocity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    use_dynamo = args.exporter == "dynamo"
    print(f"Using ONNX exporter: {args.exporter}")

    with torch.inference_mode():
        reference_outputs = wrapper(*float_inputs)
        torch.onnx.export(
            wrapper,
            float_inputs,
            str(args.output),
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=args.opset_version,
            dynamo=use_dynamo,
            external_data=True,
            report=use_dynamo,
        )

    import onnx

    onnx.checker.check_model(str(args.output))
    manifest = {
        "scope": "DOPT fake-quant fixed-layout Cosmos3 Policy MoT denoiser",
        "checkpoint_path": args.checkpoint_path,
        "dopt_config": str(args.quant_config.resolve()),
        "fakequant_weight": str(args.fakequant_weight.resolve()),
        "inputs": _shape_manifest(input_names, float_inputs),
        "outputs": _shape_manifest(output_names, reference_outputs),
        "settings": export_args.model_dump(mode="json"),
        "onnx_exporter": args.exporter,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported fake-quant ONNX: {args.output}")
    print(f"Export manifest: {manifest_path}")


if __name__ == "__main__":
    main()
