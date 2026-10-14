#!/usr/bin/env bash
set -euo pipefail

# One-card, held-out MOZ1 open-loop evaluation. Set MOZ1_EVAL_COMPARE_BASE=1
# to run both checkpoints on the same samples. To compare videos already
# generated, set MOZ1_EVAL_OUT and MOZ1_EVAL_BASE_OUT to their result directories.
cd /srv/data2/c00932551/Nvidia_models/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=

export MOZ1_ROOT=/srv/data2/20260407_merged_123
export BASE_CHECKPOINT_PATH=/srv/data2/c00932551/Nvidia_models/Cosmos3-Edge-Policy-DROID-DCP
export EDGE_HF_CHECKPOINT=/srv/data2/c00932551/Nvidia_models/Cosmos3-Edge-Policy-DROID
export WAN_VAE_PATH=/srv/data2/c00932551/Nvidia_models/Wan2.2/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/srv/data2/c00932551/cosmos_outputs/test
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_root="${MOZ1_EVAL_CKPT:-/srv/data2/c00932551/cosmos_outputs/moz1_edge_smoke/cosmos3_action/action_sft/action_policy_moz1_edge_train}"
if [[ -f "$checkpoint_root/checkpoints/latest_checkpoint.txt" ]]; then
    checkpoint_name=$(<"$checkpoint_root/checkpoints/latest_checkpoint.txt")
    if [[ ! "$checkpoint_name" =~ ^iter_[0-9]{9}$ ]]; then
        echo "Invalid latest checkpoint name: $checkpoint_name" >&2
        exit 1
    fi
    checkpoint_dir="$checkpoint_root/checkpoints/$checkpoint_name"
else
    checkpoint_dir="$checkpoint_root"
    checkpoint_name=$(basename "$checkpoint_dir")
fi
compare_base="${MOZ1_EVAL_COMPARE_BASE:-0}"
existing_base_out="${MOZ1_EVAL_BASE_OUT:-}"
if [[ "$compare_base" != 0 && "$compare_base" != 1 ]]; then
    echo "MOZ1_EVAL_COMPARE_BASE must be 0 or 1" >&2
    exit 1
fi
if [[ -n "$existing_base_out" ]]; then
    compare_base=1
fi
if [[ -z "$existing_base_out" && ! -f "$checkpoint_dir/model/.metadata" ]]; then
    echo "No completed DCP model found at $checkpoint_dir/model/.metadata" >&2
    echo "Set MOZ1_EVAL_CKPT to a run root or a specific checkpoints/iter_... directory." >&2
    exit 1
fi
if [[ "$compare_base" == 1 ]]; then
    base_checkpoint_dir="${MOZ1_EVAL_BASE_CKPT:-$BASE_CHECKPOINT_PATH}"
    if [[ -z "$existing_base_out" && ! -f "$base_checkpoint_dir/model/.metadata" ]]; then
        echo "No completed base DCP model found at $base_checkpoint_dir/model/.metadata" >&2
        exit 1
    fi
    if [[ "$(readlink -f "$base_checkpoint_dir")" == "$(readlink -f "$checkpoint_dir")" ]]; then
        echo "Base and post-trained checkpoints must be different" >&2
        exit 1
    fi
fi

make_video_comparisons() {
    local base_predictions_dir="$1"
    local ffmpeg_bin comparison_dir sample_json sample_name base_video trained_video video
    ffmpeg_bin=$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
    comparison_dir="$eval_dir/video_comparison"
    mkdir -p "$comparison_dir"
    for sample_json in "$eval_dir"/inputs/*/sample.json; do
        sample_name=$(basename "$(dirname "$sample_json")")
        base_video="$base_predictions_dir/$sample_name/vision.mp4"
        trained_video="$eval_dir/predictions/$sample_name/vision.mp4"
        for video in "$base_video" "$trained_video"; do
            if [[ ! -s "$video" ]]; then
                echo "Missing or empty generated video: $video" >&2
                return 1
            fi
        done
        "$ffmpeg_bin" -hide_banner -loglevel error -y \
            -i "$base_video" -i "$trained_video" \
            -filter_complex '[0:v]setpts=PTS-STARTPTS[base];[1:v]setpts=PTS-STARTPTS[trained];[base][trained]hstack=inputs=2[v]' \
            -map '[v]' -an -c:v libx264 -pix_fmt yuv420p \
            "$comparison_dir/$sample_name.mp4"
    done
    echo "Paired videos (left: base; right: post-trained): $comparison_dir"
}

if [[ -n "$existing_base_out" ]]; then
    if [[ -z "${MOZ1_EVAL_OUT:-}" ]]; then
        echo "Set MOZ1_EVAL_OUT to the existing post-trained result directory" >&2
        exit 1
    fi
    eval_dir="$MOZ1_EVAL_OUT"
    python -m cosmos_framework.scripts.evaluate_moz1_open_loop validate \
        --out-dir "$eval_dir" --checkpoint-path "$checkpoint_dir"
    python -m cosmos_framework.scripts.evaluate_moz1_open_loop validate \
        --out-dir "$existing_base_out" --checkpoint-path "$base_checkpoint_dir"
    # Compare sample identities and actual conditioning/target bytes before
    # displaying two independently prepared evaluation runs side by side.
    python - "$eval_dir" "$existing_base_out" <<'PY'
import json
import sys
from pathlib import Path

trained, base = map(Path, sys.argv[1:])
trained_manifest = json.loads((trained / "manifest.json").read_text())
base_manifest = json.loads((base / "manifest.json").read_text())
for key in ("dataset_root", "split", "split_seed", "split_val_ratio", "sample_seed", "selected_windows"):
    if trained_manifest.get(key) != base_manifest.get(key):
        raise ValueError(f"Evaluation manifests differ at {key}")
trained_samples = trained_manifest["samples"]
base_samples = base_manifest["samples"]
if len(trained_samples) != len(base_samples):
    raise ValueError("Evaluation manifests have different sample counts")
for left, right in zip(trained_samples, base_samples, strict=True):
    for key in ("name", "dataset_index", "episode_id", "row_index", "frame_offset"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Evaluation samples differ at {key}: {left.get('name')}")
    name = left["name"]
    left_dir, right_dir = trained / "inputs" / name, base / "inputs" / name
    for filename in ("conditioning.png", "ground_truth_last.png", "state.json", "target_action.json"):
        if (left_dir / filename).read_bytes() != (right_dir / filename).read_bytes():
            raise ValueError(f"Evaluation samples differ at {name}/{filename}")
    left_input = json.loads((left_dir / "sample.json").read_text())
    right_input = json.loads((right_dir / "sample.json").read_text())
    for input_data in (left_input, right_input):
        input_data.pop("vision_path", None)
        input_data.pop("state_path", None)
    if left_input != right_input:
        raise ValueError(f"Evaluation inputs differ at {name}/sample.json")
    for output_dir in (trained, base):
        status_path = output_dir / "predictions" / name / "sample_outputs.json"
        if json.loads(status_path.read_text()).get("status") != "success":
            raise ValueError(f"Inference did not succeed for {name} in {output_dir}")
print(f"Verified {len(trained_samples)} paired samples")
PY
    make_video_comparisons "$existing_base_out/predictions"
    exit 0
fi

for required_path in "$MOZ1_ROOT/meta/info.json" "$MOZ1_ROOT/norm_stats.json" "$EDGE_HF_CHECKPOINT" "$WAN_VAE_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required input: $required_path" >&2
        exit 1
    fi
done

eval_gpu="${MOZ1_EVAL_GPU:-7}"
if [[ ! "$eval_gpu" =~ ^[0-9]+$ ]]; then
    echo "MOZ1_EVAL_GPU must be a physical GPU index" >&2
    exit 1
fi
if ! gpu_processes=$(nvidia-smi -i "$eval_gpu" --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo "Could not query physical GPU $eval_gpu" >&2
    exit 1
fi
if grep -Eq '^[[:space:]]*[0-9]+' <<<"$gpu_processes"; then
    echo "GPU $eval_gpu already has a compute process. Choose an idle card with MOZ1_EVAL_GPU." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$eval_gpu"

eval_dir="${MOZ1_EVAL_OUT:-$IMAGINAIRE_OUTPUT_ROOT/moz1_open_loop_eval/${checkpoint_name}_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$eval_dir" ]]; then
    python -m cosmos_framework.scripts.evaluate_moz1_open_loop validate \
        --out-dir "$eval_dir" --checkpoint-path "$checkpoint_dir"
    reuse_samples=1
else
    reuse_samples=0
fi
echo "Evaluating $checkpoint_dir on physical GPU $eval_gpu"
echo "Output: $eval_dir"
if [[ "$compare_base" == 1 ]]; then
    echo "Comparing against base checkpoint: $base_checkpoint_dir"
fi

# Catch the cuBLAS/BF16 failure seen during training before preparing samples.
python -c 'import torch; x=torch.randn((128,128),device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); print("BF16 matmul OK", tuple(y.shape))'

if [[ "$reuse_samples" -eq 0 ]]; then
    python -m cosmos_framework.scripts.evaluate_moz1_open_loop prepare \
        --dataset-root "$MOZ1_ROOT" \
        --checkpoint-path "$checkpoint_dir" \
        --out-dir "$eval_dir" \
        --max-samples "${MOZ1_EVAL_SAMPLES:-32}" \
        --seed 42
fi

torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --experiment=action_policy_moz1_edge \
    --checkpoint-path="$checkpoint_dir" \
    --no-use-ema-weights \
    --no-guardrails \
    -i "$eval_dir/inputs/*/sample.json" \
    -o "$eval_dir/predictions"

python -m cosmos_framework.scripts.evaluate_moz1_open_loop score --out-dir "$eval_dir"

if [[ "$compare_base" == 1 ]]; then
    # Record the base identity before inference, so a retry cannot silently mix
    # predictions from a different checkpoint with these prepared samples.
    base_checkpoint_file="$eval_dir/base_checkpoint_path.txt"
    base_checkpoint_resolved=$(readlink -f "$base_checkpoint_dir")
    if [[ -f "$base_checkpoint_file" ]]; then
        if [[ "$(<"$base_checkpoint_file")" != "$base_checkpoint_resolved" ]]; then
            echo "Existing base predictions belong to a different checkpoint" >&2
            exit 1
        fi
    elif [[ -e "$eval_dir/base_predictions" ]]; then
        echo "Unidentified base_predictions already exist in $eval_dir" >&2
        exit 1
    else
        printf '%s\n' "$base_checkpoint_resolved" > "$base_checkpoint_file"
    fi

    torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
        --parallelism-preset=throughput \
        --experiment=action_policy_moz1_edge \
        --checkpoint-path="$base_checkpoint_dir" \
        --no-use-ema-weights \
        --no-guardrails \
        -i "$eval_dir/inputs/*/sample.json" \
        -o "$eval_dir/base_predictions"

    make_video_comparisons "$eval_dir/base_predictions"
fi
