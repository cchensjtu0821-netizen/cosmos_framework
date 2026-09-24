# MOZ1 Policy 开环验证

这条路径只做模型生成和文件保存：输入同一时刻的三路相机关键帧、当前 20 维状态和任务文字，输出预测视频与未来 16 步动作。不连接机器人，也不执行动作。

当前 MOZ1 训练只把视频第 0 帧和 action 第 0 行状态作为条件。多个关键帧应分别建成多个独立样本；不能把多张不同时刻的图像当作一个样本的多帧条件。三路相机必须来自同一时刻。

## 输入格式

每个样本需要：

- `cam_high`、`cam_left_wrist`、`cam_right_wrist` 的 RGB 图片。准备脚本将 high 放在上方，两个 wrist 各缩成 half size 后放在下方，和 MOZ1 训练布局一致。
- 20 个数的 JSON 数组，表示当前**绝对状态**，顺序为左臂 Cartesian pose 6、左夹爪 1、右臂 Cartesian pose 6、右夹爪 1、躯干 Cartesian pose 6。输入状态不应手工做第二次归一化。
- 训练时使用的 `norm_stats.json`、任务文字，以及这次 MOZ1 训练所得 checkpoint。

先确认 `norm_stats.json` 与当前 checkpoint 配套。对已经训练好的 checkpoint，推理必须重现该 checkpoint 当时的状态/action 数值映射；`scripts/audit_moz1_action_values.py` 可检查数据集中的 action 是否在训练前已经归一化。

## 制作样本

在仓库根目录运行；`--out-dir` 必须是尚不存在的目录：

```bash
python -m cosmos_framework.scripts.prepare_moz1_open_loop_input \
  --high /path/to/cam_high.png \
  --left-wrist /path/to/cam_left_wrist.png \
  --right-wrist /path/to/cam_right_wrist.png \
  --state /path/to/state_20d.json \
  --stats "$MOZ1_ROOT/norm_stats.json" \
  --prompt "Pick up the object." \
  --name case_001 \
  --out-dir outputs/moz1_cases/case_001
```

生成 `conditioning.png` 和 `sample.json`。每个验证时刻运行一次准备命令，使用不同的 `--out-dir` 和 `--name`。

## 推理并保存结果

以下命令以训练产生的 DCP checkpoint 为例。沿用训练服务器的 `MOZ1_ROOT`、`BASE_CHECKPOINT_PATH`、`WAN_VAE_PATH`、`EDGE_HF_CHECKPOINT` 等环境设置。`--checkpoint-path` 指向目标 `iter_...` 目录；不要传入原始 Edge 基座 checkpoint。这里用训练权重而非 EMA 权重；如果你的训练另行保存并希望使用 EMA，请相应调整。

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  --experiment=action_policy_moz1_edge \
  --checkpoint-path=/path/to/checkpoints/iter_000001000 \
  --no-use-ema-weights \
  -i 'outputs/moz1_cases/*/sample.json' \
  -o outputs/moz1_open_loop
```

模型会按训练契约将 20 维状态放到条件行，生成视频和动作。每个样本输出目录包含：

- `vision.mp4`：预测的 17 帧视频，包含条件帧；
- `action.json`：未来 16 步、每步 20 维的预测动作，已去掉条件状态行；
- `sample_outputs.json`：同一动作数组和运行元数据。

`action.json` 使用**数据集存储的 action 数值空间**。当前训练代码又按 `norm_stats.actions.q01/q99` 对数据集字段做了线性映射；生成结果会反向做这一次映射。若数据集字段本来已按生产端规则归一化并裁剪，文件中的数仍是那层归一化数值，不是机器人控制器的物理单位。当前路径不做控制器的逆变换，也不发送动作。

## 检查重点

先看视频的第一帧是否对应输入合成图，再看后续帧是否保持物体、相机视角和任务语义。检查 `action.json` 的 shape 是否为 `[16, 20]`、数值是否有限，以及左右臂、夹爪和躯干维度是否有合理变化。如果样本来自留出的验证 episode，可用相同时刻的真实未来 16 步 action 和视频作配对比较；开环结果只能评估预测一致性，不能等同于真实机器人成功率。
