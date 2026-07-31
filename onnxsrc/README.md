# Cosmos3 Policy DOPT ONNX tools

This directory adapts the former DA3 DOPT workflow to the fixed-layout
Cosmos3 Edge Policy denoiser. It quantizes and exports only
`RobolabPolicyService.model.net`, matching
`scripts/export_action_policy_onnx.py`; it does not export the tokenizer, VAE,
sampler, CFG loop, or action postprocessing.

## Pipeline

1. `cosmos3_quantize.py --stage gen-config` creates the DOPT configuration.
2. `modify_dopt_config.py` selects 8-bit weight + `dyn_s8` input quantization
   for `torch.nn.Linear` layers. Non-Linear layers remain float.
3. `cosmos3_quantize.py --stage quant` prepares the fixed Policy denoiser with
   synthetic tensors, saves the fake-quant state dict, and generates the
   encrypted quant file.
4. `convert_cosmos3_to_onnx.py` reconstructs the same DOPT graph, loads the
   fake-quant state dict, and exports ONNX with the same dynamo exporter used
   by the validated Policy rewrite. DOPT's tensor-valued `bit` configuration
   used by Python branches is frozen into a Python scalar before capture;
   other DOPT buffers remain tensors and fake-quant arithmetic remains in the
   graph.
   A pre-export audit rejects parameters, buffers, unregistered tensor
   attributes, or inputs found on the wrong device. The resulting fake-quant ONNX keeps the
   original export's FP32 floating boundary and fails if any FP16/BF16
   initializer remains; INT8 deployment parameters stay in the DOPT quant
   files rather than changing ONNX graph I/O to INT8. Legacy export shards are
   consolidated by default into one `<model>.onnx.data` file; the source shard
   files are removed only after the consolidated model passes ONNX checker.
   Constants below 1 MiB remain inline so Slice axes, Reshape shapes, and other
   control tensors are available to path-based ONNX shape inference. The
   compatibility rewrite also repairs older exports by inlining small external
   Constant attributes before rewriting.
5. The existing Policy compatibility rewrite removes target-unsupported
   operators and lowers vision ranks.
6. `finalize_onnx.py` assigns every node a unique non-empty name.
7. `match_quant_params.py` compares quant-file entry names with final ONNX
   node names.
8. `audit_onnx.py` checks ONNX validity, duplicate/empty names, rank limits,
   missing shape metadata, graph I/O, and operator counts.

All paths are supplied through environment variables in
`run_cosmos3_quant_onnx_full.sh`. Copy that file or override its variables;
do not commit private checkpoints, generated models, quant files, or reports.

The synthetic layout is reconstructed from the shape manifest written beside
an existing FP32 Policy ONNX. No image is read, and the tokenizer, VAE encode,
sampler, CFG, and action postprocessing paths are not executed. The generated
six input shapes must exactly match the manifest before DOPT preparation can
continue.

The DOPT simulator is a private server dependency. These scripts intentionally
import it only after adding `COSMOS3_DOPT_SIM_PATH` to `sys.path`.

## Dynamic Linear policy

The default modifier selects every config entry whose type contains
`torch.nn.modules.linear`:

- weight: signed INT8, per-channel, `min_max`;
- input: INT8 dynamic activation, `dyn_s8`;
- all other layer types: float.

Use repeatable `--exclude-regex` arguments for backend-sensitive Linear
modules. The generated JSON report records every selected and excluded layer.
Start without exclusions, then exclude only layers demonstrated unsupported or
numerically sensitive by server validation.

`COSMOS3_EMBEDDING_SEPARATE=1` asks DOPT to emit embedding weights and
dequantization scales as sidecar files. These include the language-model token
embedding and the action-to/from-LLM projection adapters; they are still inside
the VFM denoiser boundary. The workflow defaults to `0`, producing only the
main quant-parameter file. Enable separation only when required by the target
DOPT deployment.

The compatibility rewrite emits one prompt embedding table even when DOPT
embedding separation is disabled. This table is a required deployment input:
the restricted graph replaces the unsupported token-embedding `Gather` with a
`prompt_embeddings` graph input, so the host must perform that lookup.
