# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export the fixed-shape RoboLab/DROID Cosmos3 Policy denoiser to ONNX.

This exports one MoT denoise network call, not the Python tokenizer, VAE,
SequencePlan builder, UniPC loop, or action postprocessing.  A real packed
sequence is captured from the normal RoboLab preparation path so attention
indexes and modality metadata exactly match the selected request shape.

Example:
  python -m cosmos_framework.scripts.export_action_policy_onnx \
    --checkpoint-path /checkpoints/Cosmos3-Nano-Policy-DROID \
    --output-path /tmp/cosmos3_policy_denoiser.onnx
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import copy
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch

from cosmos_framework.model.generator.mot.attention import dispatch_onnx_attention
from cosmos_framework.scripts.action_policy_server_robolab import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
)


class ExportArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    checkpoint_path: str = "nvidia/Cosmos3-Nano-Policy-DROID"
    output_path: Path = Path("cosmos3_policy_denoiser.onnx")
    prompt: str = "Pick up the object and place it in the target location."
    image_height: int = 540
    image_width: int = 640
    action_chunk_size: int = 32
    action_dim: int = 8
    conditioning_fps: float = 15.0
    resolution: str | None = "480"
    domain_name: str = "droid_lerobot"
    export_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    """Floating-point dtype for the exported denoiser weights, inputs, and outputs."""
    opset_version: int = 18
    verify_onnx: bool = True
    simplify_onnx: bool = True
    simplified_output_path: Path | None = None
    export_report: bool = True


class _PackedSequenceCaptured(RuntimeError):
    pass


def _clone_modality(modality: Any) -> Any:
    return copy.copy(modality) if modality is not None else None


class PolicyDenoiserOnnxWrapper(torch.nn.Module):
    """Tensor-only ONNX boundary around one fixed-layout MoT forward."""

    def __init__(self, net: torch.nn.Module, packed_template: Any) -> None:
        super().__init__()
        self.net = net
        self.packed_template = packed_template

    def forward(
        self,
        prompt_token_ids: torch.Tensor,
        video_latent: torch.Tensor,
        action_latent: torch.Tensor,
        vision_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        action_domain_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = copy.copy(self.packed_template)
        packed.vision = _clone_modality(self.packed_template.vision)
        packed.action = _clone_modality(self.packed_template.action)
        packed.text_ids = prompt_token_ids

        assert packed.vision is not None
        packed.vision.tokens = [video_latent]
        packed.vision.timesteps = vision_timestep

        assert packed.action is not None
        packed.action.tokens = [action_latent]
        packed.action.timesteps = action_timestep
        packed.action.domain_id = [action_domain_id]

        outputs = self.net(packed_seq=packed)
        return outputs["preds_vision"][0], outputs["preds_action"][0]


def _dummy_observation(args: ExportArgs) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "observation/image": np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
        "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
    }


def _capture_first_packed_sequence(service: RobolabPolicyService, data_batch: dict[str, Any]) -> Any:
    captured: dict[str, Any] = {}

    def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        packed = kwargs.get("packed_seq")
        if packed is None and args:
            packed = args[0]
        captured["packed"] = packed
        raise _PackedSequenceCaptured

    handle = service.model.net.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            service.model.generate_samples_from_batch(
                data_batch,
                guidance=1.0,
                seed=[0],
                num_steps=1,
                shift=5.0,
            )
    except _PackedSequenceCaptured:
        pass
    finally:
        handle.remove()
    if captured.get("packed") is None:
        raise RuntimeError("Failed to capture the packed Policy input before the MoT network forward")
    return captured["packed"]


def _example_inputs(packed: Any) -> tuple[torch.Tensor, ...]:
    if packed.vision is None or not packed.vision.tokens:
        raise ValueError("Captured Policy input has no vision tokens")
    if packed.action is None or not packed.action.tokens or not packed.action.domain_id:
        raise ValueError("Captured Policy input has no action tokens/domain ID")
    return (
        packed.text_ids,
        packed.vision.tokens[0],
        packed.action.tokens[0],
        packed.vision.timesteps,
        packed.action.timesteps,
        packed.action.domain_id[0],
    )


def _cast_export_inputs(inputs: tuple[torch.Tensor, ...], export_dtype: str) -> tuple[torch.Tensor, ...]:
    if export_dtype == "bfloat16":
        return inputs
    return tuple(tensor.to(dtype=torch.float32) if tensor.is_floating_point() else tensor for tensor in inputs)


def _install_onnx_attention(net: torch.nn.Module) -> int:
    replaced = 0
    for module in net.modules():
        if hasattr(module, "dispatch_attention_fn"):
            module.dispatch_attention_fn = dispatch_onnx_attention
            replaced += 1
    if replaced == 0:
        raise RuntimeError("Failed to find any attention modules for ONNX export")
    return replaced


def _shape_manifest(names: tuple[str, ...], tensors: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}
        for name, tensor in zip(names, tensors)
    }


def _verify_float32_onnx(model_path: Path, onnx: Any) -> dict[str, int]:
    model = onnx.load_model(str(model_path), load_external_data=False)
    forbidden = {
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.BFLOAT16: "bfloat16",
    }
    counts = {name: 0 for name in forbidden.values()}
    offenders: list[str] = []
    for initializer in model.graph.initializer:
        dtype_name = forbidden.get(initializer.data_type)
        if dtype_name is not None:
            counts[dtype_name] += 1
            offenders.append(f"{initializer.name}:{dtype_name}")
    if offenders:
        preview = ", ".join(offenders[:20])
        suffix = f" (+{len(offenders) - 20} more)" if len(offenders) > 20 else ""
        raise RuntimeError(
            f"Float32 export still contains {len(offenders)} FP16/BF16 initializers: {preview}{suffix}"
        )
    return counts


def _simplified_output_path(args: ExportArgs) -> Path:
    if args.simplified_output_path is not None:
        return args.simplified_output_path
    return args.output_path.with_name(f"{args.output_path.stem}.simplified{args.output_path.suffix}")


def _simplify_onnx(source_path: Path, output_path: Path) -> None:
    try:
        import onnx
        import onnxslim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `onnxslim` to simplify the exported ONNX model") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnxslim.slim(
        str(source_path),
        str(output_path),
        save_as_external_data=True,
        # ONNX shape inference serializes an in-memory ModelProto and therefore
        # cannot process this multi-GB model even when its weights originated
        # as external data.
        no_shape_infer=True,
        # Avoid constant folding that materializes very large derived tensors.
        size_threshold=1024 * 1024,
    )
    onnx.checker.check_model(str(output_path))


def export_action_policy_onnx(args: ExportArgs) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The Cosmos3 Policy exporter requires a CUDA environment")

    server_args = RobolabServerArgs(
        checkpoint_path=args.checkpoint_path,
        image_height=args.image_height,
        image_width=args.image_width,
        action_chunk_size=args.action_chunk_size,
        action_dim=args.action_dim,
        conditioning_fps=args.conditioning_fps,
        resolution=args.resolution,
        domain_name=args.domain_name,
        guidance=1.0,
        num_steps=1,
        use_torch_compile=False,
    )
    service = RobolabPolicyService(server_args)
    sample = service._build_sample(_dummy_observation(args))
    data_batch = _build_data_batch_from_sample(sample)
    packed = _capture_first_packed_sequence(service, data_batch)
    replaced_attention_modules = _install_onnx_attention(service.model.net)
    print(f"Using ONNX dense attention in {replaced_attention_modules} module(s)")

    export_dtype = torch.float32 if args.export_dtype == "float32" else torch.bfloat16
    service.model.net.to(dtype=export_dtype)
    wrapper = PolicyDenoiserOnnxWrapper(service.model.net, packed).eval()
    inputs = _cast_export_inputs(_example_inputs(packed), args.export_dtype)
    input_names = (
        "prompt_token_ids",
        "video_latent",
        "action_latent",
        "vision_timestep",
        "action_timestep",
        "action_domain_id",
    )
    output_names = ("vision_velocity", "action_velocity")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference_outputs = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            str(args.output_path),
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=args.opset_version,
            dynamo=True,
            external_data=True,
            report=args.export_report,
        )

    manifest = {
        "scope": "one fixed-layout Cosmos3 Policy MoT denoise forward",
        "checkpoint_path": args.checkpoint_path,
        "inputs": _shape_manifest(input_names, inputs),
        "outputs": _shape_manifest(output_names, reference_outputs),
        "settings": {
            "prompt": args.prompt,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "action_chunk_size": args.action_chunk_size,
            "action_dim": args.action_dim,
            "conditioning_fps": args.conditioning_fps,
            "resolution": args.resolution,
            "domain_name": args.domain_name,
            "export_dtype": args.export_dtype,
            "opset_version": args.opset_version,
        },
    }
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if args.verify_onnx or args.simplify_onnx:
        try:
            import onnx
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install `onnx` to verify or simplify the exported model") from exc

    if args.verify_onnx:
        onnx.checker.check_model(str(args.output_path))
        if args.export_dtype == "float32":
            dtype_counts = _verify_float32_onnx(args.output_path, onnx)
            print(f"Verified float32 ONNX initializers: forbidden dtype counts={dtype_counts}")

    simplified_path = None
    if args.simplify_onnx:
        simplified_path = _simplified_output_path(args)
        print(f"Simplifying ONNX to: {simplified_path}")
        _simplify_onnx(args.output_path, simplified_path)

    print(f"ONNX saved to: {args.output_path}")
    if simplified_path is not None:
        print(f"Simplified ONNX saved to: {simplified_path}")
    print(f"Shape manifest saved to: {manifest_path}")


def main() -> None:
    import tyro

    export_action_policy_onnx(tyro.cli(ExportArgs, description=__doc__))


if __name__ == "__main__":
    main()
