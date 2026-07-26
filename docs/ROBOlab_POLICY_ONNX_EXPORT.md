# Cosmos3 RoboLab Policy ONNX 导出

## 导出范围

脚本导出一次固定 packed-layout 的 Cosmos3 Policy MoT denoise forward。

ONNX 包含：

- prompt token embedding；
- video latent 到 hidden 的编码；
- action 到 hidden 的编码；
- MoT Transformer；
- vision velocity head；
- action velocity head。

ONNX 不包含：

- 字符串 tokenizer；
- 原始 RGB 到 latent 的 VAE encoder；
- SequencePlan 和 packed sequence 元数据构造；
- UniPC/EDM 采样循环与 CFG；
- action 反归一化和机器人坐标转换；
- 可选 VAE decoder。

这些阶段含有 Python 控制流、字符串处理或动态数据结构，应保留在宿主程序中。部署时每个 sampler step 调用 ONNX 一次；CFG 开启时分别调用 conditional 和 unconditional forward。

## 默认 RoboLab/DROID 设置

| 参数 | 默认值 |
|---|---:|
| 原始 observation image | `[540, 640, 3]`, `uint8` |
| action chunk | `32` |
| 原始 action dim | `8` |
| conditioning FPS | `15` |
| transform resolution | `"480"` |
| domain | `droid_lerobot` |
| ONNX opset | `18` |

video latent、模型内部 action 宽度、prompt token 数和 timestep shape 不手工猜测。脚本会使用正常 RoboLab Policy 预处理执行一次请求，在 MoT forward 前捕获真实 `PackedSequence`，据此生成 ONNX example inputs，并输出同名 `.onnx.json` shape manifest。

## 环境要求

- 可正常运行 Cosmos3-Nano-Policy-DROID 推理的 CUDA 环境；
- PyTorch 及项目完整依赖；
- `onnx`；
- PyTorch dynamo ONNX exporter 所需依赖（通常包含 `onnxscript`）。

建议额外安装 `onnxruntime-gpu`，用于后续数值对比。

## 导出命令

在完整 `cosmos-framework` 项目根目录运行：

```bash
python -m cosmos_framework.scripts.export_action_policy_onnx \
  --checkpoint-path /实际路径/Cosmos3-Nano-Policy-DROID \
  --output-path /实际输出路径/cosmos3_policy_denoiser.onnx
```

也可以直接使用 Hugging Face checkpoint 名称：

```bash
python -m cosmos_framework.scripts.export_action_policy_onnx \
  --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
  --output-path outputs/onnx/cosmos3_policy_denoiser.onnx
```

默认会运行 `onnx.checker.check_model`。临时跳过：

```bash
python -m cosmos_framework.scripts.export_action_policy_onnx \
  --checkpoint-path /实际模型路径 \
  --output-path outputs/onnx/cosmos3_policy_denoiser.onnx \
  --no-verify-onnx
```

## ONNX 输入输出

输入名称：

| 名称 | 类型 | 含义 |
|---|---|---|
| `prompt_token_ids` | integer tensor | 已 tokenize 的 conditional 或 unconditional prompt |
| `video_latent` | floating tensor | VAE 编码后的单样本 video latent |
| `action_latent` | floating tensor | 当前扩散步的 action latent |
| `vision_timestep` | floating tensor | vision token timestep |
| `action_timestep` | floating tensor | action token timestep |
| `action_domain_id` | integer tensor | DROID embodiment domain ID |

输出名称：

| 名称 | 含义 |
|---|---|
| `vision_velocity` | 当前扩散步预测的 vision velocity |
| `action_velocity` | 当前扩散步预测的 action velocity |

精确 shape 以导出产生的 manifest 为准：

```text
cosmos3_policy_denoiser.onnx.json
```

## 当前状态和首次验证清单

当前版本是在无 PyTorch、无 CUDA、无 checkpoint 的外网代码机上完成的初版，只通过语法和静态检查，尚未声称端到端导出成功。在内网首次运行时必须完成：

1. 成功加载 Policy checkpoint；
2. 成功捕获 packed sequence；
3. PyTorch wrapper forward 成功；
4. `torch.onnx.export` 成功；
5. `onnx.checker.check_model` 通过；
6. 用 Netron 打开模型，确认六个输入和两个输出；
7. 安装 `onnxruntime-gpu` 后，对相同输入比较 PyTorch 与 ONNX 输出误差；
8. 根据 exporter 报告处理不支持的 attention、MoE 或自定义 CUDA 算子。

Cosmos3 使用的 fused attention/MoE 实现可能无法直接转换为标准 ONNX 算子。如果首次运行在这些算子处失败，应根据 exporter 报告为导出路径切换到可导出的 eager/SDPA 实现，不能通过删除网络计算来“让导出成功”。
