# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Provisional Cosmos3-Edge action-policy SFT recipe for the MOZ1 dataset.

This recipe deliberately assumes that the configured 6-D Cartesian action
columns are already relative.  It exists to validate the full data/model/train
path before the authoritative producer-side pose conversion is recovered.

Server smoke example::

    MOZ1_ROOT=/path/to/20260407_merged_123 \
    BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-DCP \
    EDGE_HF_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID \
    WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth \
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
      --sft-toml=cosmos_framework/configs/toml_config/action_policy_moz1_edge_smoke.toml
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_moz1_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


action_policy_moz1_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_moz1_edge["job"].update(
    project="cosmos3_action",
    group="action_sft",
    name="action_policy_moz1_edge",
    wandb_mode="disabled",
)
action_policy_moz1_edge["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)

# EDGE_MODEL_CONFIG points its processor at the public nvidia/Cosmos3-Edge
# repository.  This smoke is intended to run from a self-contained local Policy
# snapshot, so remove every Hub selector and dispatch build_processor_lazy via
# its local-directory mode instead.
edge_processor_config = action_policy_moz1_edge["model"]["config"]["vlm_config"]["tokenizer"]
edge_processor_config.pop("repository", None)
edge_processor_config.pop("revision", None)
edge_processor_config.pop("subdir", None)
edge_processor_config["tokenizer_type"] = "${oc.env:EDGE_HF_CHECKPOINT}"

# Start conservatively for the first data-path smoke.  Scale only after a finite
# 1--10 step run establishes the real memory and throughput envelope.
action_policy_moz1_edge["optimizer"].update(
    lr=1.0e-5,
    keys_to_select=[
        "moe_gen",
        "time_embedder",
        "vae2llm",
        "llm2vae",
        "k_norm_und_for_gen",
        "action2llm",
        "llm2action",
        "action_modality_embed",
    ],
)
action_policy_moz1_edge["scheduler"].update(cycle_lengths=[100], warm_up_steps=[5])
action_policy_moz1_edge["trainer"].update(max_iter=10, logging_iter=1, grad_accum_iter=1)
action_policy_moz1_edge["checkpoint"].update(
    load_path="???",
    save_iter=10,
    strict_resume=False,
    # The renewed Edge checkpoint contains action heads.  Reuse them for the
    # smoke instead of reinitializing the full action path.
    keys_to_skip_loading=["net_ema."],
)

action_policy_moz1_edge["dataloader_train"].update(
    dataset_name="action_moz1",
    max_samples_per_batch=1,
    max_sequence_length=None,
)
action_policy_moz1_edge["dataloader_train"]["dataloader"].update(
    batch_size=1,
    in_order=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    datasets=dict(
        moz1=dict(
            ratio=1,
            dataset=L(get_action_moz1_sft_dataset)(
                root="${oc.env:MOZ1_ROOT}",
                fps=30.0,
                chunk_length=16,
                mode="policy",
                split="train",
                split_seed=42,
                split_val_ratio=0.01,
                # norm_stats.json is expected at MOZ1_ROOT by default.
                action_normalization="quantile",
                stats_path=None,
                sample_stride=1,
                resolution="256",
                max_action_dim="${model.config.max_action_dim}",
                cfg_dropout_rate=0.1,
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                format_prompt_as_json=True,
                append_idle_frames=False,
                iterable_shuffle=True,
                episode_shuffle_seed=42,
                # Leaf names are resolved against meta/info.json. Override any
                # of these with a full feature key if the producer used a
                # different namespace.
                high_camera_feature="cam_high",
                left_wrist_camera_feature="cam_left_wrist",
                right_wrist_camera_feature="cam_right_wrist",
                left_state_feature="leftarm_state_cart_pos",
                left_gripper_state_feature="leftarm_gripper_state_pos",
                right_state_feature="rightarm_state_cart_pos",
                right_gripper_state_feature="rightarm_gripper_state_pos",
                torso_state_feature="torso_state_cart_pos",
                left_action_feature="leftarm_cmd_cart_pos",
                left_gripper_action_feature="leftarm_gripper_cmd_pos",
                right_action_feature="rightarm_cmd_cart_pos",
                right_gripper_action_feature="rightarm_gripper_cmd_pos",
                torso_action_feature="torso_cmd_cart_pos",
            ),
        )
    ),
)

# 16 future actions + one absolute-state row and 17 matching video frames.
action_policy_moz1_edge["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
action_policy_moz1_edge["model"]["config"]["max_num_tokens_after_packing"] = -1

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_moz1_edge",
    node=action_policy_moz1_edge,
)
