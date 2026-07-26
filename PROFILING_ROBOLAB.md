# RoboLab action-policy profiling

This copy is based on NVIDIA/cosmos-framework commit
`3017efb48826b69f9582ac6e3c001f50d1401067`.

## Lightweight module timings

Start the server with the original arguments plus:

```bash
--enable-module-profile \
--profile-every-n 1 \
--profile-warmup-requests 3 \
--profile-output-dir /tmp/cosmos3_action_profile
```

Each profiled request appends one JSON object to:

```text
/tmp/cosmos3_action_profile/module_profile.jsonl
```

The report includes:

- CPU: request, sample/batch construction, action/video post-processing.
- CUDA: generator, prompt reasoner when enabled, data preparation, VAE encode,
  sampler, conditional/unconditional CFG forwards, network, modality encoders,
  MoT joint forward, output heads, and optional VAE decode.
- Counters: VAE calls and diffusion-network calls.
- Peak allocated and reserved CUDA memory.

The first `profile_warmup_requests` entries are marked `"warmup": true`.
Do not include them in stable latency statistics. Parent timings contain their
children, so do not add nested regions together.

## One-request PyTorch Profiler

Add:

```bash
--torch-profiler-request 4
```

This profiles only the fourth request and writes:

```text
/tmp/cosmos3_action_profile/request_4_trace.json
/tmp/cosmos3_action_profile/request_4_summary.txt
```

Open the trace with Chrome/Perfetto. PyTorch `with_flops=True` reports FLOPs
only for supported operators; it is not a complete-model FLOPs measurement for
fused attention, Triton, NATTEN, or custom kernels.

## Measurement guidance

- Use module profiling with compilation disabled for clean attribution.
- Use compilation enabled for production end-to-end latency.
- Warm up at least three requests.
- Collect at least 20 stable requests and report median, p90, and p95.
- Profiling is disabled by default and does not change the WebSocket response.
- `reasoner_ar` appears only when `upsample_task` is actually enabled. The
  standard RoboLab call does not pass it; understanding/generation joint work is
  represented by `mot_joint_forward`.
