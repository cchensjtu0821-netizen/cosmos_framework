# Export a real MOZ1 recording for RoboLab replay

Run from the directory containing the `cosmos_framework/` package, using the existing dataset Python environment. No RoboLab checkout, Isaac Sim, checkpoint or model server is required. The exporter needs NumPy and PyArrow; video trimming additionally needs an existing ffmpeg with libx264.

```bash
python -m cosmos_framework.scripts.data_processing.export_moz1_recording \
  --root "$MOZ1_ROOT" \
  --output /tmp/moz_reference_30s \
  --duration 30 --joint-units rad --trim-videos
```

Set `MOZ1_ROOT` to the existing LeRobot-v3 training dataset, or supply its path explicitly with `--root`. When `MOZ1_ROOT` is set, `--root` may be omitted. To execute the file directly instead of as a module:

```bash
python cosmos_framework/scripts/data_processing/export_moz1_recording.py --help
```

The default selects the longest episode covering 30 seconds (900 frames at native 30 fps), then the continuous window with the largest sum of absolute joint increments. Ties select the smallest episode ID and earliest window. Use `--episode-id N`, `--start-seconds S`, and `--duration 60` to override. No episode boundaries or frame gaps are stitched together. An entirely stationary selected episode is rejected; choose another episode explicitly.

The `recording.json` bundle preserves original state20, action20 and measured left/right/torso joint states. No normalization, command-minus-state conversion, model inference or unit conversion is applied. `--joint-units rad` explicitly declares the raw joint units; left joint0..6, right joint0..6 and torso joint0..5 map to QX LeftArm_0..6, RightArm_0..6 and LegWaist_0..5. Verify the physical model and joint conventions before playback. Recorded joint-command fields may have been filled from measured states and are not treated as independent commands.

Each LeRobot-v3 video uses the selected episode's video chunk/file indices and from/to timestamps. With `--trim-videos`, matching `real_view_*.mp4` clips are written alongside the JSON. Without it, the JSON retains source video paths and offsets without copying video. Existing output directories are never overwritten. A specific Parquet histogram compatibility error tries an already-installed DuckDB fallback; the script never installs dependencies.

Copy the output directory to the RoboLab machine, for example `output/moz_reference_30s/`. RoboLab commit `e048f70` or later supports:

```bash
bash scripts/run_cosmos3_moz.sh --motion-smoke \
  --motion-replay output/moz_reference_30s/recording.json \
  --motion-replay-kind state \
  --motion-output output/moz_replay_state_30s --motion-keep-open
```

`state` restores the first measured joint pose and replays continuous measured joint targets as a motion baseline. Switching to `--motion-replay-kind action` instead feeds the original Cartesian commands through the current anchored IK diagnostic. That mode still has diagnostic limits and incomplete SDK/TCP calibration; it is not a faithful absolute-command reproduction. Both modes skip model requests and ignore the motion chunk count.

RoboLab saves `simulation.mp4`, `replay_control.jsonl` and a summary with source timestamps and joint errors. Pre-roll is excluded from the video. Compare source and simulation frame indices rather than wall-clock execution times. Current replay covers arms and torso, holds grippers/head/base, and does not restore real object poses or camera calibration, so visual/interaction differences do not by themselves prove an action mapping bug.

The standalone exporter and its synthetic-Parquet tests were ported from RoboLab `e048f7066d8b7c0375b5381d9fda5a214be83fb4`; their Apache-2.0 SPDX notices are retained. The implementation does not import RoboLab. Real dataset selection, ffmpeg encoding and Isaac replay still require server validation.
