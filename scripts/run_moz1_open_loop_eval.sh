#!/usr/bin/env bash
set -euo pipefail

# One-card, held-out MOZ1 open-loop evaluation. Set MOZ1_EVAL_GPU to use an
# idle physical GPU other than 7, or MOZ1_EVAL_SAMPLES to change the sample count.
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
if [[ ! -f "$checkpoint_dir/model/.metadata" ]]; then
    echo "No completed DCP model found at $checkpoint_dir/model/.metadata" >&2
    echo "Set MOZ1_EVAL_CKPT to a run root or a specific checkpoints/iter_... directory." >&2
    exit 1
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
    echo "Evaluation directory already exists: $eval_dir" >&2
    exit 1
fi
echo "Evaluating $checkpoint_dir on physical GPU $eval_gpu"
echo "Output: $eval_dir"

# Catch the cuBLAS/BF16 failure seen during training before preparing samples.
python -c 'import torch; x=torch.randn((128,128),device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); print("BF16 matmul OK", tuple(y.shape))'

python -m cosmos_framework.scripts.evaluate_moz1_open_loop prepare \
    --dataset-root "$MOZ1_ROOT" \
    --checkpoint-path "$checkpoint_dir" \
    --out-dir "$eval_dir" \
    --max-samples "${MOZ1_EVAL_SAMPLES:-32}" \
    --seed 42

torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --experiment=action_policy_moz1_edge \
    --checkpoint-path="$checkpoint_dir" \
    --no-use-ema-weights \
    -i "$eval_dir/inputs/*/sample.json" \
    -o "$eval_dir/predictions"

python -m cosmos_framework.scripts.evaluate_moz1_open_loop score --out-dir "$eval_dir"
