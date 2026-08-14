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
- This repository's `origin` uses Git over SSH. Direct commits and pushes to
  `origin/main` do not require GitHub CLI authentication, a pull request, or a
  `gh auth status` prerequisite. Do not block a direct publish because a
  cached GitHub API token is absent or stale; attempt the requested Git SSH
  operation and ask about authentication only if that operation itself reports
  an SSH/permission failure. Use `gh` authentication only for workflows that
  actually need the GitHub API, such as creating or editing a pull request.
- For a direct publish from the mixed worktree: fetch `origin/main`; confirm
  local `main` is current; stage only explicit requested paths; inspect
  `git diff --cached` and run relevant checks; commit tersely; run
  `git push origin main`; then verify the remote SHA with
  `git ls-remote origin refs/heads/main`. Preserve all unrelated staged,
  unstaged, and untracked files throughout.
- Do not commit generated ONNX models, external weight files, profiling output,
  caches, credentials, tokens, or machine-specific artifacts unless explicitly
  requested.
- The local Codex workspace does not have the user's ONNX/CUDA runtime. Do not
  claim GPU or ONNX Runtime validation was performed locally. Make code changes
  here and provide exact validation commands for the user's GPU server.
- Prefer diagnosis before changing model semantics. ONNX graph rewrites must
  preserve numerical behavior and require server-side comparison against the
  original graph.

## Daily Work Logs

- On each calendar day when work is performed in this package, create or
  update `docs/agent_logs/YYYY-MM-DD.md` using the current local date. Do not
  create empty logs for days with no work.
- Keep the log concise and append durable context for each material issue:
  symptom/error, diagnosis, solution or next action, affected files/commands,
  validation evidence, and current status.
- Use explicit status labels: `OPEN`, `INVESTIGATING`, `BLOCKED`, or
  `RESOLVED`. When an issue is fixed, update its existing entry to `RESOLVED`
  rather than leaving stale text that still describes it as active.
- Preserve useful failed approaches when they prevent repeated work, but mark
  them clearly as rejected or superseded.
- At the start and end of a work turn, check the current daily log size. When
  it reaches or exceeds 128 KiB, notify the user and report its path and size.
  Do not delete, truncate, compress, split, or archive it without the user's
  decision.
- Daily logs are repository notes, not generated runtime artifacts. Never put
  credentials, tokens, private server paths, large command output, model data,
  or profiling dumps in them. Summarize such material and use placeholders for
  machine-specific paths.

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
- That intermediate `ScatterND(reduction=add)` is still not accepted by OMG.
  During `onnxsrc/finalize_onnx.py`, a validated contiguous axis-0 update is
  lowered again to `Slice(data) → Add(updates) → Concat(prefix, updated,
  suffix)`. Only static `[N, 1]` consecutive row indices with matching update
  shapes are eligible; unresolved reductions must fail finalization.
- For default overwrite `ScatterND`, remove an explicit `reduction=none`
  attribute when present. Static, sorted, unique `[N, 1]` row-index patterns
  with matching update shapes must be lowered to interleaved data/update
  `Slice` nodes followed by bounded fan-in `Concat`; this avoids OMG's faulty
  shape propagation for strided-Slice-fed updates. Report every remaining
  `ScatterND`; do not infer that arbitrary ScatterND layouts are supported.
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

## Active ONNX Worklog — 2026-08-03

The FP32 Edge Policy export and structural rewrites currently reach:

- zero target-unsupported nodes when all rewrites are enabled;
- zero tensors above rank 4, including graph I/O;
- historical finite, numerically equivalent outputs with the superseded exact
  causal overwrite rewrite and vision rank lowering enabled; the current finite
  additive-mask candidate requires new server-side validation.

The two previously isolated numerical-equivalence issues are resolved:

1. **Causal mask rewrite**
   - Original: `Where(mask, -Inf, attention_scores)`.
   - The current edge candidate uses a shared fixed-layout additive bias with
     `0` at visible positions and finite FP16-safe `-10000` at masked positions.
   - This intentionally trades exact masked NaN/Inf branch isolation for a
     compact `Add` graph whose masked softmax probabilities underflow to zero
     for normal finite deployment scores.
   - ORT output comparison remains mandatory, but an `allclose=False` result is
     diagnostic and non-blocking so Step 6 and downstream OMG can run. Checker,
     structural compatibility, shape/rank, and runtime failures remain blocking.

2. **Vision rank lowering**
   - Fixed batch-1 rank-5 vision I/O is lowered to rank 4.
   - Rank-6 patchify is replaced by a rank-4 transpose,
     `SpaceToDepth`, channels-last transpose, and final reshape.
   - Rank-6 unpatchify uses the exact inverse with rank-4 reshape, transpose,
     `DepthToSpace(mode=DCR)`, and output transpose.
   - The corrected channel depth order is `[p, q, C]`, matching the original
     `cthpwq -> thwpqc` patchify permutation and its inverse.

`ScatterElements -> ScatterND` rewriting remains independently validated and
was not responsible for either historical mismatch. The produced
`ScatterND(reduction=add)` is only an intermediate ONNX form: finalization must
lower it to the validated `Slice`/`Add`/`Concat` sequence before OMG/OMC
conversion.

Useful diagnostic flags:

```text
--keep-causal-where
--keep-vision-ranks
--keep-scatter-elements
```

The `--keep-*` flags are retained for regression isolation; outputs produced
with unsupported operators retained are not deployment-compatible. A saved
ONNX file is usable only after the command exits successfully with `finite=True`,
`nonfinite=(0,0)`, `allclose=True`, zero unsupported target nodes, and zero
high-rank nodes/I/O.
