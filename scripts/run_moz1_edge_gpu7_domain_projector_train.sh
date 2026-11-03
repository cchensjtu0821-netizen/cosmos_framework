#!/usr/bin/env bash
set -euo pipefail

# Required paths are supplied by the caller; defaults select physical GPU 7.
# Extra arguments are Hydra overrides, e.g. trainer.max_iter=10.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/../.."
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi
export LD_LIBRARY_PATH=

: "${MOZ1_ROOT:?Set MOZ1_ROOT to the LeRobot-v3 dataset root}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to the initial Policy DCP}"
: "${EDGE_HF_CHECKPOINT:?Set EDGE_HF_CHECKPOINT to the local Edge Policy snapshot}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to Wan2.2_VAE.pth}"
: "${IMAGINAIRE_OUTPUT_ROOT:?Set IMAGINAIRE_OUTPUT_ROOT to the training output root}"
export MOZ1_ROOT BASE_CHECKPOINT_PATH EDGE_HF_CHECKPOINT WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

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
    --sft-toml=cosmos_framework/configs/toml_config/action_policy_moz1_edge_domain_projector_train.toml \
    -- "$@"
