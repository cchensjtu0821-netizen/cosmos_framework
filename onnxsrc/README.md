# Cosmos3 Policy DOPT ONNX tools

This directory adapts the former DA3 DOPT workflow to the fixed-layout
Cosmos3 Edge Policy denoiser. It quantizes and exports only
`RobolabPolicyService.model.net`, matching
`scripts/export_action_policy_onnx.py`; it does not export the tokenizer, VAE,
sampler, CFG loop, or action postprocessing.

## Pipeline

1. `cosmos3_quantize.py --stage gen-config` loads the checkpoint, clones the
   shared two-Linear timestep MLP into independent
   `vision_time_embedder`/`action_time_embedder` modules, then creates the DOPT
   configuration. The clones start with identical values but own distinct
   Parameters, ONNX initializers, and quantization entries.
2. `modify_dopt_config.py` selects static 8-bit weight + input quantization
   for Policy `torch.nn.Linear` layers and removes non-quantized entries from
   the deployment config.
3. `cosmos3_quantize.py --stage quant` prepares the fixed Policy denoiser with
   synthetic tensors, saves the fake-quant state dict, and generates the
   encrypted quant file.
4. `convert_cosmos3_to_onnx.py` builds the clean FP32 Policy graph, loads only
   the matching model weights from the fake-quant state dict, ignores
   DOPT-only `.quant_op.*` state, and exports ONNX using PyTorch's dynamo
   exporter in evaluation mode with opset 18. No fake-quant arithmetic,
   Quant, or Dequant nodes are intentionally
   included in the ONNX; deployment quantization remains in `quant_params_v2`
   and is applied by the downstream ONNX-to-OM flow. After serialization,
   `onnxslim` performs additional constant folding and graph simplification.
   External tensors are then repacked into one `<model>.onnx.pb` file by
   default. Set `COSMOS3_EXTERNAL_DATA_USE_PB=0` when running the full pipeline
   to retain the former `<model>.onnx.data` naming. The main ONNX `location`
   metadata and OMG `--weight` path are updated together. Any
   FP16/BF16 initializer fails validation.
   A pre-export audit rejects parameters, buffers, unregistered tensor
   attributes, or inputs found on the wrong device. The resulting clean ONNX keeps the
   original export's FP32 floating boundary and fails if any FP16/BF16
   initializer remains; INT8 deployment parameters stay in the DOPT quant
   files rather than changing ONNX graph I/O to INT8. Legacy export shards are
   consolidated by default into one `<model>.onnx.pb` file (or
   `<model>.onnx.data` with the switch disabled); the source shard
   files are removed only after the consolidated model passes ONNX checker.
   Constants below 1 MiB remain inline so Slice axes, Reshape shapes, and other
   control tensors are available to path-based ONNX shape inference. The
   compatibility rewrite also repairs older exports by inlining small external
   Constant attributes before rewriting.
5. The existing Policy compatibility rewrite removes target-unsupported
   operators and lowers vision ranks. `COSMOS3_DECOMPOSE_GEMM=1` optionally
   lowers default-scale `Gemm` to `MatMul` plus a bias `Add` when needed.
   Adding `COSMOS3_TRANSPOSE_GEMM_WEIGHTS=1` physically transposes every
   `transB=1` Linear weight in a distinct copy of the external-data file, so
   each replacement `MatMul` reads the transposed initializer directly and no
   weight `Transpose` node is emitted. The original external data is preserved
   for Step-5 ORT comparison. This mode requires enough free disk space for a
   second full external-data file; override its path with
   `COSMOS3_TRANSPOSED_ONNX_WEIGHT`.
6. `finalize_onnx.py` assigns every node a unique non-empty name, bypasses
   redundant `Identity` nodes immediately before graph outputs for OMG
   compatibility, normalizes safe nonzero `Reshape.allowzero` attributes, and
   materializes fully static single-axis singleton broadcasts before `Mul` as
   explicit `Expand` nodes. This preserves ONNX broadcast semantics while
   preventing OMG constant folding from losing the RoPE feature width. It also
   lowers fixed `GatherND` patterns to `Slice`/`Concat` and removes explicit
   default `ScatterND(reduction=none)` attributes and lowers validated
   contiguous `ScatterND(reduction=add)` patterns to `Slice`/`Add`/`Concat`.
   It also lowers default-overwrite `ScatterND` with static, sorted, unique
   `[N, 1]` row indices to interleaved data/update `Slice` nodes and bounded
   fan-in `Concat` nodes. After these lowerings it removes graph-unreachable
   nodes, including dead index Casts whose GatherND/ScatterND consumers were
   replaced, without folding live Casts or renaming live compute nodes. The
   report lists every remaining `ScatterND` so later
   OMG failures can be correlated without assuming all ScatterND layouts are
   interchangeable.
   It also rejects node tensors or graph I/O whose rank exceeds 4. Findings are
   reported but high-rank nodes are never deleted merely to pass the audit.
7. `match_quant_params.py` compares quant-file entry names with final ONNX
   node names and requires all four modality-specific timestep Linear names in
   both artifacts. This deliberately rejects a Step-5 resume from a legacy
   shared-weight ONNX/quant-file pair even when its older 340 names match each
   other.
8. `audit_onnx.py` checks ONNX validity, duplicate/empty names, rank limits,
   missing shape metadata, graph I/O, and operator counts.
9. After the strict audit passes, `run_cosmos3_quant_onnx_full.sh` invokes OMG
   to convert the named ONNX and its external weight data to OMC. Set
   `COSMOS3_RUN_OMG=0` to stop after ONNX generation and audit. OMG stdout and
   stderr remain visible in the terminal and are also saved to
   `${COSMOS3_QUANT_ROOT}/omg.log`; override `COSMOS3_OMG_LOG` to select another
   path. The pipeline retains OMG's nonzero exit status when logging through
   `tee`.

All paths are supplied through environment variables in
`run_cosmos3_quant_onnx_full.sh`. Copy that file or override its variables;
do not commit private checkpoints, generated models, quant files, or reports.

The physically transposed-weight mode is an explicit backend experiment. The
rewrite report must show `materialize_gemm_weight_transposes=true`, every
transposed weight and its old/new shape, and ORT output metrics. Step 7 still
requires complete quant-name coverage, but name coverage alone does not prove
that a backend interprets per-channel INT8 scales along the intended axis after
the weight layout changes. Checker, original-versus-rewritten output comparison,
and OMG/OMC validation therefore remain mandatory.

The synthetic layout is reconstructed from the shape manifest written beside
an existing FP32 Policy ONNX. No image is read, and the tokenizer, VAE encode,
sampler, CFG, and action postprocessing paths are not executed. The generated
six input shapes must exactly match the manifest before DOPT preparation can
continue.

The DOPT simulator is a private server dependency. These scripts intentionally
import it only after adding `COSMOS3_DOPT_SIM_PATH` to `sys.path`.

## Static Linear policy

The default modifier selects every config entry whose type contains
`torch.nn.modules.linear`, except `language_model.lm_head`, which is not
executed by the exported vision/action denoiser boundary:

- weight: signed INT8, per-channel, `min_max`;
- input: static INT8 activation, per-tensor, `min_max`;
- all non-quantized entries are omitted from the final DOPT config.

For the current Edge Policy graph, the two timestep Linear layers execute once
for vision and once for action. The ONNX/DOPT preparation path therefore
replaces the checkpoint's two shared `time_embedder.mlp.{0,2}` parameter sets
with four independent entries:

```text
vision_time_embedder.mlp.0
vision_time_embedder.mlp.2
action_time_embedder.mlp.0
action_time_embedder.mlp.2
```

The clean exporter audits the raw/simplified graph and fails unless all four
weight initializers remain distinct. For the recorded Edge layout this closes
the historical gap between 342 exported Linear executions and 340 selected
parameter sets; a fresh DOPT configuration is expected to select 342 Linear
entries. Reusing an older two-entry timestep quant file with the new graph is
not supported.

Because the module split happens immediately after checkpoint loading, the
first run after this change must start at Step 1. `COSMOS3_START_STEP5=1`
cannot retrofit separate weights or quantization entries into an existing raw
ONNX and will fail the required-name check for legacy artifacts.

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

## Synthetic OMC input binaries

Generate deterministic smoke-test inputs for the fixed rank-lowered Edge
Policy OMC boundary:

```bash
python -m cosmos_framework.onnxsrc.generate_omc_input_bins \
  --output-dir outputs/omc_inputs \
  --seed 0 \
  --timestep 500
```

The command writes one headerless, C-order, little-endian FP16 `.bin` per
input plus `manifest.json` using only the Python standard library. The fixed
order and shapes are
`video_latent:[48,9,33,40]`, `action_latent:[33,64]`,
`vision_timestep:[2720]`, `action_timestep:[32]`, and
`prompt_embeddings:[108,2048]`. Existing files are preserved unless
`--overwrite` is supplied. These random tensors are only for OMC loading and
interface smoke tests; they are not substitutes for tokenizer, VAE, sampler,
or host-side prompt-embedding preprocessing outputs.
