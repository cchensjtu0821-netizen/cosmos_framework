#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
if [[ ! -f .venv/bin/activate ]]; then
    echo "Missing $repo_root/.venv/bin/activate; use the training uv environment." >&2
    exit 1
fi
source .venv/bin/activate
export LD_LIBRARY_PATH=
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Defaults follow run_moz1_edge_gpu7_train.sh's server layout, without a fixed
# account/workspace ID. Override any location using the existing training envs.
model_root=$(cd "$repo_root/.." && pwd)
workspace_root=$(cd "$repo_root/../.." && pwd)
export MOZ1_ROOT="${MOZ1_ROOT:-$workspace_root/../20260407_merged_123}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$model_root/Cosmos3-Edge-Policy-DROID-DCP}"
export EDGE_HF_CHECKPOINT="${EDGE_HF_CHECKPOINT:-$model_root/Cosmos3-Edge-Policy-DROID}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$model_root/Wan2.2/Wan2.2_VAE.pth}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$workspace_root/cosmos_outputs/test}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${MOZ1_SERVER_GPU:-2}"
export MOZ1_SERVER_CKPT="${MOZ1_SERVER_CKPT:-$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action/action_sft/action_policy_moz1_edge_gpu7_train}"
export MOZ1_STATS_PATH="${MOZ1_STATS_PATH:-$MOZ1_ROOT/norm_stats.json}"

exec python -m cosmos_framework.scripts.action_policy_server_robolab_moz \
    --port "${MOZ1_SERVER_PORT:-8000}" "$@"
