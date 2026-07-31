#!/usr/bin/env python3
"""Export a calibrated DOPT Cosmos3 Policy denoiser to fixed-layout FP32 ONNX."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.onnxsrc.cosmos3_quantize import _import_dopt, _load_fixed_policy
from cosmos_framework.scripts.export_action_policy_onnx import (
    PolicyDenoiserOnnxWrapper,
    _install_onnx_attention,
    _shape_manifest,
    _verify_float32_onnx,
)


def _audit_export_tensor_devices(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    expected_device: torch.device,
) -> None:
    offenders: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.device != expected_device:
            offenders.append(f"parameter:{name}={parameter.device}")
    for name, buffer in model.named_buffers():
        if buffer.device != expected_device:
            offenders.append(f"buffer:{name}={buffer.device}")
    for module_name, module in model.named_modules():
        parameter_names = set(module._parameters)
        buffer_names = set(module._buffers)
        for attribute_name, value in vars(module).items():
            if (
                isinstance(value, torch.Tensor)
                and attribute_name not in parameter_names
                and attribute_name not in buffer_names
                and value.device != expected_device
            ):
                qualified_name = f"{module_name}.{attribute_name}" if module_name else attribute_name
                offenders.append(f"tensor_attribute:{qualified_name}={value.device}")
    for index, value in enumerate(inputs):
        if value.device != expected_device:
            offenders.append(f"input:{index}={value.device}")
    if offenders:
        preview = ", ".join(offenders[:50])
        suffix = f" (+{len(offenders) - 50} more)" if len(offenders) > 50 else ""
        raise RuntimeError(
            f"ONNX export tensors must all be on {expected_device}; found {preview}{suffix}"
        )
    print(
        f"Export device audit passed: parameters, buffers, tensor attributes, "
        f"and {len(inputs)} inputs are on {expected_device}"
    )


def _verify_fp32_graph_io(model_path: Path, onnx: Any) -> dict[str, int]:
    model = onnx.load_model(str(model_path), load_external_data=False)
    floating_types = {
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    }
    counts = {"float32": 0, "non_float": 0}
    offenders: list[str] = []
    for value in [*model.graph.input, *model.graph.output]:
        element_type = value.type.tensor_type.elem_type
        if element_type == onnx.TensorProto.FLOAT:
            counts["float32"] += 1
        elif element_type in floating_types:
            offenders.append(
                f"{value.name}:{onnx.TensorProto.DataType.Name(element_type)}"
            )
        else:
            counts["non_float"] += 1
    if offenders:
        raise RuntimeError(
            "Fake-quant ONNX floating graph I/O must be FP32, found "
            + ", ".join(offenders)
        )
    return counts


def _consolidate_external_data(model_path: Path, onnx: Any) -> Path:
    lightweight_model = onnx.load_model(str(model_path), load_external_data=False)
    old_locations = {
        entry.value
        for tensor in onnx.external_data_helper._get_all_tensors(lightweight_model)
        for entry in tensor.external_data
        if entry.key == "location"
    }
    data_path = model_path.with_suffix(model_path.suffix + ".data")
    data_path.unlink(missing_ok=True)

    model = onnx.load_model(str(model_path), load_external_data=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{model_path.name}.consolidated-",
        suffix=model_path.suffix,
        dir=model_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_model_path = Path(temporary_file.name)
    temporary_model_path.unlink()
    try:
        onnx.save_model(
            model,
            str(temporary_model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            # Keep scalar/shape/axis constants in the ModelProto so ONNX path-
            # based shape inference can read them without loading external data.
            size_threshold=1024 * 1024,
            convert_attribute=True,
        )
        onnx.checker.check_model(str(temporary_model_path))
        temporary_model_path.replace(model_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)

    output_directory = model_path.parent.resolve()
    removed = 0
    for location in old_locations:
        old_path = (model_path.parent / location).resolve()
        if old_path == data_path.resolve() or old_path.parent != output_directory:
            continue
        if old_path.is_file():
            old_path.unlink()
            removed += 1
    print(f"Consolidated ONNX external data: {data_path} (removed {removed} old shard file(s))")
    return data_path


def _simplify_onnx_in_place(model_path: Path, onnx: Any) -> None:
    """Fold constants and simplify a large external-data ONNX in place."""
    try:
        import onnxslim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install onnxslim to fold constants and simplify the exported ONNX"
        ) from exc

    with tempfile.NamedTemporaryFile(
        prefix=f".{model_path.name}.simplified-",
        suffix=model_path.suffix,
        dir=model_path.parent,
        delete=False,
    ) as temporary_file:
        simplified_path = Path(temporary_file.name)
    simplified_path.unlink()
    try:
        onnxslim.slim(
            str(model_path),
            str(simplified_path),
            save_as_external_data=True,
            # In-memory ONNX shape inference cannot serialize this multi-GB
            # ModelProto. Shape metadata is handled by the rewrite/audit stage.
            no_shape_infer=True,
            # Fold ordinary scalar/shape constants without materializing very
            # large derived tensors inside the ModelProto.
            size_threshold=1024 * 1024,
        )
        onnx.checker.check_model(str(simplified_path))
        simplified_path.replace(model_path)
        # onnxslim names external data after its temporary output. Repack it
        # immediately so the final artifact again has one stable .onnx.data.
        _consolidate_external_data(model_path, onnx)
    finally:
        simplified_path.unlink(missing_ok=True)
        for temporary_artifact in model_path.parent.glob(f"{simplified_path.name}*"):
            if temporary_artifact.is_file():
                temporary_artifact.unlink()
    print(f"Folded constants and simplified ONNX in place: {model_path}")


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
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument(
        "--external-data-mode",
        choices=("single", "sharded"),
        default="single",
        help="Consolidate all external tensors into one .onnx.data file by default.",
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
    _audit_export_tensor_devices(wrapper, float_inputs, torch.device("cuda", torch.cuda.current_device()))
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
    print("Using ONNX exporter: legacy (dynamo=False)")

    with torch.no_grad():
        reference_outputs = wrapper(*float_inputs)
        torch.onnx.export(
            wrapper,
            float_inputs,
            str(args.output),
            export_params=True,
            # Cosmos3's traced graph contains CUDA tensors plus CPU shape/axis
            # constants synthesized by the legacy exporter. Its JIT constant
            # folding pass attempts to evaluate them together and fails with a
            # mixed-device error. Keep folding disabled during export; later
            # compatibility rewrites handle graph constants explicitly.
            do_constant_folding=False,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=args.opset_version,
            training=torch.onnx.TrainingMode.EVAL,
            # Keep DA3's TorchScript/legacy tracing behavior across PyTorch
            # versions whose default exporter may differ.
            dynamo=False,
        )

    import onnx

    external_data_path = None
    if args.external_data_mode == "single":
        external_data_path = _consolidate_external_data(args.output, onnx)
    _simplify_onnx_in_place(args.output, onnx)
    if args.external_data_mode == "single":
        external_data_path = args.output.with_suffix(args.output.suffix + ".data")
    onnx.checker.check_model(str(args.output))
    forbidden_float_initializer_counts = _verify_float32_onnx(args.output, onnx)
    graph_io_dtype_counts = _verify_fp32_graph_io(args.output, onnx)
    print(
        "Verified FP32 fake-quant ONNX initializers: "
        f"forbidden dtype counts={forbidden_float_initializer_counts}"
    )
    print(f"Verified fake-quant ONNX graph I/O dtypes: {graph_io_dtype_counts}")
    manifest = {
        "scope": "DOPT fake-quant fixed-layout Cosmos3 Policy MoT denoiser",
        "checkpoint_path": args.checkpoint_path,
        "dopt_config": str(args.quant_config.resolve()),
        "fakequant_weight": str(args.fakequant_weight.resolve()),
        "inputs": _shape_manifest(input_names, float_inputs),
        "outputs": _shape_manifest(output_names, reference_outputs),
        "settings": export_args.model_dump(mode="json"),
        "onnx_exporter": "legacy",
        "onnx_simplified": True,
        "constant_folding": "onnxslim",
        "external_data_mode": args.external_data_mode,
        "external_data_path": str(external_data_path.resolve()) if external_data_path is not None else None,
        "forbidden_float_initializer_counts": forbidden_float_initializer_counts,
        "graph_io_dtype_counts": graph_io_dtype_counts,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported fake-quant ONNX: {args.output}")
    print(f"Export manifest: {manifest_path}")


if __name__ == "__main__":
    main()
