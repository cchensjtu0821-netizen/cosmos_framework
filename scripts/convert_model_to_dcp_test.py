# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for local checkpoint handling in the Hugging Face to DCP converter."""

from pathlib import Path
from unittest.mock import patch

import pytest

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    from cosmos_framework.scripts import convert_model_to_dcp

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_redirect_edge_processor_to_local_snapshot(tmp_path: Path) -> None:
    (tmp_path / "processor_config.json").write_text(
        '{"processor_class": "Cosmos3EdgeProcessor"}', encoding="utf-8"
    )
    tokenizer_config = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "repository": "nvidia/Cosmos3-Edge-Policy-DROID",
        "revision": "main",
        "subdir": "",
        "tokenizer_type": "nvidia/Cosmos3-Edge-Policy-DROID",
    }
    model_dict = {"config": {"vlm_config": {"tokenizer": tokenizer_config}}}

    redirected = convert_model_to_dcp._redirect_edge_processor_to_local(model_dict, tmp_path)

    assert redirected is True
    assert tokenizer_config == {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "tokenizer_type": str(tmp_path),
    }


def test_redirect_edge_processor_rejects_non_edge_snapshot(tmp_path: Path) -> None:
    tokenizer_config = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "tokenizer_type": "nvidia/Cosmos3-Edge-Policy-DROID",
    }
    model_dict = {"config": {"vlm_config": {"tokenizer": tokenizer_config}}}

    redirected = convert_model_to_dcp._redirect_edge_processor_to_local(model_dict, tmp_path)

    assert redirected is False
    assert tokenizer_config["tokenizer_type"] == "nvidia/Cosmos3-Edge-Policy-DROID"
