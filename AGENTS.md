# AGENTS.md — Cosmos Framework Package

This file applies to work under `cosmos_framework/`. Read the repository-root
`AGENTS.md` first, then use this file for package-specific context.

## Working Agreement

- Preserve unrelated user changes and untracked files.
- Stage and commit only files that belong to the requested change; never use
  `git add -A` in a mixed worktree.
- Before publishing, show or inspect the scoped diff and use a terse,
  descriptive commit message.
- When the user asks to publish directly, update `main`, push `origin/main`,
  and remove any temporary branch created for that change.
- Do not commit generated ONNX models, external weight files, profiling output,
  caches, credentials, tokens, or machine-specific artifacts unless explicitly
  requested.
- The local Codex workspace does not have the user's ONNX/CUDA runtime. Do not
  claim GPU or ONNX Runtime validation was performed locally. Make code changes
  here and provide exact validation commands for the user's GPU server.
- Prefer diagnosis before changing model semantics. ONNX graph rewrites must
  preserve numerical behavior and require server-side comparison against the
  original graph.

## Package Map

| Area | Location | Purpose |
| --- | --- | --- |
| Policy service | `scripts/action_policy_server_robolab.py` | Builds RoboLab observations, runs Policy inference, profiles requests, and returns robot actions. |
| ONNX export | `scripts/export_action_policy_onnx.py` | Captures one fixed-layout `Cosmos3VFMNetwork.forward()` and exports the denoiser boundary. |
| ONNX rewrite | `scripts/rewrite_action_policy_onnx.py` | Rewrites selected unsupported ONNX patterns for a restricted backend. |
| ONNX audit | `scripts/inspect_action_policy_onnx.py` | Reports target operators, high-rank tensors, graph I/O, and unknown ranks without modifying the model. |
| Policy model orchestration | `model/generator/omni_mot_model.py` | Prepares conditions/noise, runs CFG and the sampler, and combines velocity outputs. |
| VFM network | `model/generator/mot/cosmos3_vfm_network.py` | Encodes text/vision/action, runs MoT layers, and decodes vision/action velocity. |
| Attention backend | `model/generator/mot/attention.py` | Packed attention and ONNX-compatible attention dispatch. |
| Profiling utilities | `utils/module_profiler.py` | CPU/CUDA timings, shapes, FLOPs, counters, and memory reporting. |
| Policy documentation | `docs/ROBOlab_POLICY_*.md`, `docs/ROBOLAB_POLICY_*.md` | Code walkthrough, profiling, inputs, and ONNX export notes. |

## Action Policy Execution Boundary

The full RoboLab Policy path is:

```text
observation
→ sample/data transform
→ tokenizer + VAE encode
→ condition/noise preparation
→ UniPC sampler
→ conditional and unconditional denoiser calls
→ CFG velocity
→ latent updates
→ action postprocessing
→ returned robot action
```

The exported Policy ONNX contains only one fixed packed-layout VFM denoiser
forward:

```text
text/vision/action encoding
→ MoT Transformer
→ vision head + action head
→ vision_velocity + action_velocity
```

It does not contain the tokenizer, VAE encoder, sampler loop, CFG, latent
updates, or final action postprocessing.

## Current Edge Policy Facts

- The active deployment target is `Cosmos3-Edge-Policy-DROID`, not the generic
  Cosmos3 Edge reasoner.
- Edge uses hidden size `2048` and intermediate size `9216`; Nano Policy uses
  hidden size `4096` and intermediate size `12288`.
- A model can load Edge weights while still using a non-canonical 32-step
  request configuration. Model architecture and action chunk length are
  independent settings.
- The observed 32-step layout is action latent `[33, 64]` and returned action
  `[32, 8]` (one state row plus 32 future actions).
- The canonical Edge Policy layout uses 16 future actions, 17 video frames,
  conditioning FPS 5, and JSON-formatted prompts. Confirm actual shapes from
  profiling rather than hard-coding inferred vision shapes.
- Conditional and unconditional packed text layouts differ. A fixed-layout
  conditional ONNX must not be assumed to accept unconditional tokens or an
  arbitrary prompt length.

## ONNX Rewrite Rules

- Never delete unsupported nodes merely to pass an operator audit.
- Rewrite only patterns whose equivalence conditions are validated.
- `ScatterElements(axis=0, reduction=add)` produced by
  `Unsqueeze → Expand → scatter_add` may be rewritten to
  `ScatterND(reduction=add)` using compact `[N, 1]` row indices.
- An output file may exist even when the rewrite command ultimately fails:
  `rewrite_action_policy_onnx.py` saves the model before its final compatibility
  audit. Always inspect the command exit status and generated rewrite report.
- After every rewrite, the GPU server must run:
  1. `onnx.checker`;
  2. `inspect_action_policy_onnx.py`;
  3. numerical comparison of original and rewritten vision/action outputs.

## Server-Side Validation

Commands involving private checkpoints, multi-GB ONNX external data, CUDA,
ONNX Runtime, or profiling run on the user's server. Keep server paths and
credentials out of commits unless the user explicitly requests otherwise.

When handing off a change, report:

- changed files and commit;
- expected graph transformation;
- exact server command to run;
- expected audit result;
- numerical equivalence checks still required.

## Active ONNX Worklog — 2026-07-30

The FP32 Edge Policy export and structural rewrites currently reach:

- zero target-unsupported nodes when all rewrites are enabled;
- zero tensors above rank 4, including graph I/O;
- finite, exactly equal outputs when both causal `Where` rewriting and vision
  rank lowering are disabled with `--keep-causal-where --keep-vision-ranks`.

Two numerical-equivalence issues remain:

1. **Causal mask rewrite**
   - Original: `Where(mask, -Inf, attention_scores)`.
   - Current attempted replacement: `Clip(attention_scores) + causal_bias`.
   - With the replacement enabled, both rewritten outputs become entirely
     non-finite while the reference outputs remain finite.
   - Keeping the original 28 `Where` nodes removes the non-finite outputs.
   - Do not treat `Clip + Add` as an accepted solution. A future replacement
     must preserve overwrite semantics even when masked source values are NaN
     or Inf; fixed unique-index `ScatterND` is the leading candidate.

2. **Vision rank lowering**
   - The rank-6 patchify/unpatchify and rank-5 I/O lowering passes structural
     inspection but is not numerically equivalent.
   - With causal `Where` retained but vision rank lowering enabled, observed
     errors were approximately:
     `vision_velocity max_abs_error=8.5640602`,
     `action_velocity max_abs_error=0.082353115`.
   - With `--keep-vision-ranks`, both outputs are finite and exactly equal:
     `allclose=True`, `max_abs_error=0`.
   - Recheck `SpaceToDepth`/`DepthToSpace` channel and patch-element ordering
     against the original six-dimensional Reshape/Transpose layouts.

This isolation run also left the `ScatterElements -> ScatterND` rewrite enabled
and still produced exact outputs when the two problematic passes were disabled.
That is evidence that ScatterElements rewriting is not responsible for the
currently observed mismatch, though final full-model validation remains
required after the two issues above are fixed.

Useful diagnostic flags:

```text
--keep-causal-where
--keep-vision-ranks
--keep-scatter-elements
```

Diagnostic outputs are not deployment-compatible. A saved ONNX file is usable
only after the command exits successfully with `finite=True`,
`nonfinite=(0,0)`, `allclose=True`, zero unsupported target nodes, and zero
high-rank nodes/I/O.
