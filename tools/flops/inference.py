"""Theoretical forward FLOPs for request-scoped Cosmos policy profiling."""

from __future__ import annotations

from typing import Any

from cosmos_framework.tools.flops.omni_mot import (
    compute_omni_mot_flops_per_batch,
    get_omni_mot_model_descriptor,
)


def _descriptor(net: Any):
    vlm_cfg = net.language_model.config
    text_cfg = getattr(vlm_cfg, "text_config", vlm_cfg)
    net_cfg = net.config
    num_experts = int(getattr(text_cfg, "num_experts", 0))
    return get_omni_mot_model_descriptor(
        hidden_size=int(text_cfg.hidden_size),
        num_hidden_layers=int(text_cfg.num_hidden_layers),
        num_attention_heads=int(text_cfg.num_attention_heads),
        num_key_value_heads=int(text_cfg.num_key_value_heads),
        head_dim=getattr(text_cfg, "head_dim", None),
        intermediate_size=int(text_cfg.intermediate_size),
        vocab_size=int(text_cfg.vocab_size),
        use_moe=num_experts > 0,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text_cfg, "num_experts_per_tok", 0)),
        moe_intermediate_size=int(getattr(text_cfg, "moe_intermediate_size", 0)),
        decoder_sparse_step=int(getattr(text_cfg, "decoder_sparse_step", 1)),
        mlp_only_layers=list(getattr(text_cfg, "mlp_only_layers", [])),
        latent_patch_size=int(getattr(net_cfg, "latent_patch_size", 2)),
        latent_channel_size=int(getattr(net_cfg, "latent_channel_size", 48)),
        action_dim=int(getattr(net_cfg, "action_dim", 32)),
        sound_dim=int(getattr(net_cfg, "sound_dim", 64)),
        frequency_embedding_size=int(getattr(net_cfg, "frequency_embedding_size", 256)),
        predict_text_tokens=bool(getattr(net_cfg, "predict_text_tokens", False)),
    )


def compute_policy_forward_flops(net: Any, packed_seq: Any) -> dict[str, int]:
    """Return non-overlapping MoT forward components for one network call."""
    cfg = _descriptor(net)
    text_tokens = int(packed_seq.text_ids.numel())
    vision_tokens = (
        int(packed_seq.vision.sequence_indexes.numel())
        if packed_seq.vision is not None and packed_seq.vision.sequence_indexes is not None
        else 0
    )
    action_tokens = (
        int(packed_seq.action.sequence_indexes.numel())
        if packed_seq.action is not None and packed_seq.action.sequence_indexes is not None
        else 0
    )
    sound_tokens = (
        int(packed_seq.sound.sequence_indexes.numel())
        if packed_seq.sound is not None and packed_seq.sound.sequence_indexes is not None
        else 0
    )
    split_lens = [int(x) for x in packed_seq.split_lens]
    attn_modes = list(packed_seq.attn_modes)
    total = int(
        compute_omni_mot_flops_per_batch(
            cfg,
            B=1,
            text_tokens=text_tokens,
            vision_tokens=vision_tokens,
            action_tokens=action_tokens,
            sound_tokens=sound_tokens,
            vision_gen=bool(net.config.vision_gen),
            action_gen=bool(net.config.action_gen),
            sound_gen=bool(net.config.sound_gen),
            backwardpass_ratio=0.0,
            split_lens=split_lens,
            attn_modes=attn_modes,
            include_padding=True,
        )
    )
    d = cfg.hidden_size
    freq = cfg.frequency_embedding_size
    timestep = 2 * freq * d + 2 * d * d
    patch_dim = cfg.latent_patch_size**2 * cfg.latent_channel_size
    encode_vision = 2 * vision_tokens * patch_dim * d + (timestep if vision_tokens else 0)
    decode_vision = 2 * vision_tokens * d * patch_dim
    encode_action = 2 * action_tokens * cfg.action_dim * d + (timestep if action_tokens else 0)
    decode_action = 2 * action_tokens * d * cfg.action_dim
    encode_sound = 2 * sound_tokens * cfg.sound_dim * d + (timestep if sound_tokens else 0)
    decode_sound = 2 * sound_tokens * d * cfg.sound_dim
    components = {
        "encode_text": 0,
        "encode_vision": encode_vision,
        "encode_action": encode_action,
        "encode_sound": encode_sound,
        "vision_head": decode_vision,
        "action_head": decode_action,
        "sound_head": decode_sound,
    }
    components["mot_joint_forward"] = max(0, total - sum(components.values()))
    return components
