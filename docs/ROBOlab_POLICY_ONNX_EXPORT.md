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

导出器会关闭服务端默认启用的 `torch.compile`，确保 PyTorch ONNX
exporter 捕获 eager 模型，而不是已经由 Inductor 包装过的模型区域。
导出期间还会把两路 FlashAttention dispatch 替换为等价的 dense
`MatMul + Softmax` 实现，因为 FlashAttention 2/3 是没有标准 ONNX
lowering 的自定义 CUDA 算子。该替换只作用于导出模型，不改变正常服务。

## 环境要求

- 可正常运行 Cosmos3-Nano-Policy-DROID 推理的 CUDA 环境；
- PyTorch 及项目完整依赖；
- `onnx`；
- `onnxslim`（默认执行图简化，可用 `--no-simplify-onnx` 关闭）；
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

默认还会保留原始 ONNX，并额外生成：

```text
cosmos3_policy_denoiser.simplified.onnx
cosmos3_policy_denoiser.simplified.onnx.data
```

可指定简化模型路径或关闭简化：

```bash
python -m cosmos_framework.scripts.export_action_policy_onnx \
  --checkpoint-path /实际模型路径 \
  --output-path outputs/onnx/cosmos3_policy_denoiser.onnx \
  --simplified-output-path outputs/onnx/cosmos3_policy_denoiser.deploy.onnx

python -m cosmos_framework.scripts.export_action_policy_onnx \
  --checkpoint-path /实际模型路径 \
  --output-path outputs/onnx/cosmos3_policy_denoiser.onnx \
  --no-simplify-onnx
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

## 部署兼容性检查

检查指定不支持算子，以及任一输入或输出 rank 大于 4 的节点：

```bash
python -m cosmos_framework.scripts.inspect_action_policy_onnx \
  outputs/onnx/cosmos3_policy_denoiser.simplified.onnx
```

默认检查 `ConstantOfShape`、`Einsum`、`Gather`、
`ScatterElements` 和 `Where`，并生成：

```text
cosmos3_policy_denoiser.simplified.onnx.compatibility.json
```

可追加自定义算子、改变最大 rank，或在 CI 中发现问题时返回非零：

```bash
python -m cosmos_framework.scripts.inspect_action_policy_onnx \
  outputs/onnx/cosmos3_policy_denoiser.simplified.onnx \
  --target-op Trilu \
  --target-op GatherND \
  --target-op ScatterND \
  --max-rank 4 \
  --fail-on-findings
```

检查器使用 `load_external_data=False`，不会把多 GB 外部权重载入内存。
未知 rank 会单独记录；在缺少 `value_info` 时，不能把未知 rank 当作
四维以下。

## 受限后端图改写

对固定 DROID layout 执行部署改写，不修改 PyTorch forward：

```bash
python -m cosmos_framework.scripts.rewrite_action_policy_onnx \
  outputs/onnx/edge_policy.onnx \
  outputs/onnx/edge_policy.compatible.onnx
```

脚本会：

- 将 unary `Einsum` 改为 `Transpose`；
- 将 `ConstantOfShape` 改为标量 initializer + `Expand`；
- 折叠常量 `Gather`，将常量标量 `Gather` 改为 `Slice + Squeeze`；
- 将可证明为 axis-0 固定索引布局的 `ScatterElements` 改为 `ScatterND`；
- 将 causal `Trilu + Where` 改为固定 causal bias + `Add`；
- 将 `action_domain_id` 固定为 DROID domain ID 8；
- 将 `prompt_token_ids` 输入替换为 `prompt_embeddings`，并导出原始
  embedding 表 `.npy` sidecar。

改写输出必须与输入 ONNX 位于同一目录，确保原 external-data 相对引用仍然有效。
脚本只处理满足已知结构并能检查前置条件的 pattern，结束后运行
`onnx.checker` 和兼容性审计；目标算子仍有残留时返回错误。

等价性边界：

- `Einsum`、`ConstantOfShape`、常量/标量 `Gather` 和已验证索引布局的
  `ScatterElements` 为局部严格等价；
- domain 改写只对固定 `droid_lerobot`（ID 8）部署等价；
- 宿主端必须使用导出的 embedding 表，根据 token ID 生成
  `[108, 2048]` 的 `prompt_embeddings`；
- additive causal bias 要求 mask 前 attention score 为有限值。

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
