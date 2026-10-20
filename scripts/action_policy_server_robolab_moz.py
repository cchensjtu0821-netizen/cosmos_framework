# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility entry point for the MOZ1 RoboLab policy server."""

from __future__ import annotations

import importlib
import runpy
import sys

_TARGET = "cosmos_framework.scripts.inference_tools.action_policy_server_robolab_moz"

if __name__ == "__main__":
    runpy.run_module(_TARGET, run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module(_TARGET)
