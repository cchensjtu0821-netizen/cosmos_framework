#!/usr/bin/env bash
set -euo pipefail

# Run all 10,440 formal optimizer steps on physical GPU 7.
# Keep a separate job name so this run does not resume the 10-step smoke.
cd /srv/data2/c00932551/Nvidia_models/cosmos-framework
source .venv/bin/activate

export MOZ1_ROOT=/srv/data2/20260407_merged_123
export BASE_CHECKPOINT_PATH=/srv/data2/c00932551/Nvidia_models/Cosmos3-Edge-Policy-DROID-DCP
export EDGE_HF_CHECKPOINT=/srv/data2/c00932551/Nvidia_models/Cosmos3-Edge-Policy-DROID
export WAN_VAE_PATH=/srv/data2/c00932551/Nvidia_models/Wan2.2/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/srv/data2/c00932551/cosmos_outputs/test
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=7

for input_dir in "$MOZ1_ROOT" "$BASE_CHECKPOINT_PATH" "$EDGE_HF_CHECKPOINT"; do
    if [[ ! -d "$input_dir" ]]; then
        echo "Missing required directory: $input_dir" >&2
        exit 1
    fi
done
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    echo "Missing VAE file: $WAN_VAE_PATH" >&2
    exit 1
fi

exec torchrun \
    --standalone \
    --nproc-per-node=1 \
    -m cosmos_framework.scripts.train \
    --sft-toml=cosmos_framework/configs/toml_config/action_policy_moz1_edge_train.toml \
    -- \
    job.name=action_policy_moz1_edge_gpu7_train \
    model.config.parallelism.data_parallel_shard_degree=1
