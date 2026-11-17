#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Pin a specific post-training iteration for a reproducible task evaluation.
: "${MOZ1_SERVER_CKPT:?Set MOZ1_SERVER_CKPT to the actual post-trained DCP iteration}"
exec bash "$script_dir/run_moz1_robolab_server.sh" \
    --num-steps "${MOZ1_NUM_STEPS:-30}" --guidance "${MOZ1_GUIDANCE:-1.0}" \
    --shift "${MOZ1_SHIFT:-3.0}" --seed "${MOZ1_GENERATION_SEED:-0}" "$@"
