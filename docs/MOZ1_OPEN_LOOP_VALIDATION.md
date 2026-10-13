# MOZ1 Policy 开环验证

这条路径输入同一时刻的三路相机关键帧、当前 20 维状态和任务文字，输出预测视频与未来 16 步动作；一键评估还能将预测动作与留出数据配对并计算误差。不连接机器人，也不执行动作。

当前 MOZ1 训练只把视频第 0 帧和 action 第 0 行状态作为条件。多个关键帧应分别建成多个独立样本；不能把多张不同时刻的图像当作一个样本的多帧条件。三路相机必须来自同一时刻。

## 一键评估留出数据

`scripts/run_moz1_open_loop_eval.sh` 默认读取指定 MOZ1 训练目录的 `checkpoints/latest_checkpoint.txt`，使用物理 GPU 7，从训练时相同的 episode 级留出集（`split=val`、seed 42、比例 1%）固定抽取 32 个有效窗口。它加载模型一次，逐个生成视频和 16×20 动作，然后将预测动作与对应时刻的真实动作配对。

在服务器的仓库根目录运行：

```bash
bash cosmos_framework/scripts/run_moz1_open_loop_eval.sh
```

当前 GPU 7 若仍在训练，脚本会拒绝占用该卡；可等待训练结束，或指定空闲卡与样本数：

```bash
MOZ1_EVAL_GPU=6 MOZ1_EVAL_SAMPLES=32 bash cosmos_framework/scripts/run_moz1_open_loop_eval.sh
```

若要评估指定步数，可将 `MOZ1_EVAL_CKPT` 设为完整的 `checkpoints/iter_XXXXXXXXX` 目录；`MOZ1_EVAL_OUT` 可指定输出目录。默认每次生成独立的带时间戳输出目录。若推理失败后需要重试，设置 `MOZ1_EVAL_OUT` 为上次打印的结果目录，脚本会核对 checkpoint 与输入文件，复用已准备的样本。推理命令关闭 Guardrail，避免开环数值测评依赖额外的 Guardrail 模型文件。脚本会打印实际选择的 checkpoint 与结果目录，并验证 `model/.metadata` 是否存在。

结果目录包含 `manifest.json`（抽样索引、episode 编号、种子）、`inputs/*/`（条件图、状态、真实动作和最后一帧真值图）、`predictions/*/`（生成视频与动作）、`metrics.json` 和 `per_sample.csv`。指标包括整体 MAE/RMSE、零动作基线、每维及每步 MAE、左右臂位置/旋转/夹爪及躯干位置/旋转分组误差、每个样本第 16 步的 L2 误差。只有所有样本均成功输出有限的 `[16,20]` 动作时才生成汇总指标；缺失、失败或形状错误都会报错。

这些动作指标都在数据集**存储的数值空间**内，不是物理单位，也不代表机器人任务成功率。视频输出供人工查看；这里不计算视频 PSNR 等逐像素指标。抽样窗口可能来自相同的留出 episode，`metrics.json` 会报告实际 episode 数量。

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
