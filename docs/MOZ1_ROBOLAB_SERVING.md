# MOZ1 Edge 后训练策略服务

专用入口 `python -m cosmos_framework.scripts.action_policy_server_robolab_moz` 与旧 DROID 服务并存。它加载 MOZ1 Edge 训练产生的本地 DCP，通过 OpenPI MsgPack/NumPy WebSocket 协议提供在线推理。

从包含 `.venv` 和 `cosmos_framework/` 的仓库根目录运行：

```bash
bash cosmos_framework/scripts/run_moz1_robolab_server.sh
```

启动器沿用 `run_moz1_edge_gpu7_train.sh` 的目录布局，以仓库位置计算服务器工作区，激活 uv `.venv`，清空 `LD_LIBRARY_PATH`，默认物理 GPU 2、端口 8000。不会激活 conda 环境。

默认训练目录为：

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action/action_sft/action_policy_moz1_edge_gpu7_train
```

优先选择 `checkpoints/latest_checkpoint.txt` 发布的完整保存；指针缺失或保存未完成时，按 iteration 数值选择最新完整 DCP。显式指定未完成 iteration 会报错。文件预检要求非空 `model/.metadata` 和非空 `.distcp` 文件，真正的权重可加载性仍由模型加载确认。启动时固定 checkpoint，不会在服务中途追踪新权重。

```bash
# 指定另一个 run、GPU，或直接固定某次 iteration：
MOZ1_SERVER_CKPT=/path/to/run_or_iter \
MOZ1_ROOT=/path/to/training_dataset \
MOZ1_SERVER_GPU=2 MOZ1_SERVER_PORT=8000 \
bash cosmos_framework/scripts/run_moz1_robolab_server.sh

# 不加载 GPU，只做文件/统计量检查：
bash cosmos_framework/scripts/run_moz1_robolab_server.sh --check-only
```

`BASE_CHECKPOINT_PATH`、`EDGE_HF_CHECKPOINT`、`WAN_VAE_PATH`、`IMAGINAIRE_OUTPUT_ROOT`、`MOZ1_ROOT` 可覆盖启动器默认值。`MOZ1_STATS_PATH` 默认为 `$MOZ1_ROOT/norm_stats.json`，必须对应所选 checkpoint 的训练统计量。服务会校验两组 `q01/q99` 的 20 维有限数值，并在握手中提供统计量 SHA-256；文件内容与训练是否一致仍需保留原始训练资料确认。

也可以在已正确导出训练环境变量的环境里直接启动：

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab_moz \
  --checkpoint-path /path/to/run_or_checkpoints_or_iter \
  --stats-path /path/to/norm_stats.json --port 8000
```

实际权重来自自动选定/显式指定的 DCP，使用 `action_policy_moz1_edge` 实验、非 EMA 权重、throughput 并行配置、单卡加载。`BASE_CHECKPOINT_PATH` 只是满足原实验的路径引用。默认关闭 torch.compile 和 guardrails，采样参数与现有 MOZ1 开环样本相同：30 steps、guidance 1.0、shift 3.0、seed 0。可使用 `--experiment-override` 追加 Hydra 覆盖，使用 `--use-torch-compile`、`--decode-video`、`--deterministic-seed` 开启相应功能；修改实验时须保持下文 MOZ1 契约。

## 线上输入输出

连接后服务器首先发送 metadata，包括 `protocol=moz1_policy_v1`、`domain_name=moz1`、20 维动作、16 步 chunk、30 FPS、resolution 256、字段/相机顺序、stats hash、所选 checkpoint 及采样设置。

请求：

```text
protocol: moz1_policy_v1
state_space: dataset_stored_20d
observation/state: float[20]
observation/images:
  cam_high: RGB uint8[H,W,3]
  cam_left_wrist: RGB uint8[H,W,3]
  cam_right_wrist: RGB uint8[H,W,3]
prompt: 非空任务文字
session_id: 可选诊断标识（服务不维护动作历史）
```

也可用 `observation/image` 提供已经按 MOZ1 布局拼接的 RGB uint8 图像，替代三路图像；两种方式不能同时提供。布局为上方 high、下方左右 wrist，高位相机的 H/W 必须为偶数。客户端应保证三路图像与当前状态来自同一仿真时刻。

状态顺序与动作顺序均为：左臂 Cartesian pose 6、左夹爪 1、右臂 Cartesian pose 6、右夹爪 1、躯干 Cartesian pose 6。状态是数据集同样表示的当前绝对状态，尚未做 Cosmos 的 quantile 映射。

服务复用 `inference/action.py:build_action_batch`：按 `norm_stats.state` 处理状态，构建 1 行状态＋16 步动作、17 帧、JSON prompt、moz1 domain 和正确的反归一化记录。`generate_samples_from_batch` 已用记录做逆归一化；服务器只去掉第一行，返回有限的 `[16,20]` `action`，不再次反归一化，也不做 DROID 的最后一维夹爪翻转。

返回的 `action_space=dataset_stored_20d` 表示**数据集存储空间**，不是已验证的米/弧度或关节空间。数据生产端可能还有归一化、裁剪和相对位姿约定，仍由实际机器人适配器转换。当前数据适配器对动作字段的相对位姿含义尚有待生产端确认，不能直接将这些数下发为 QX 关节角。

## RoboLab 端

配套入口为 RoboLab 仓库的 `policies/cosmos3/run_moz.py`，一键包装为 `scripts/run_cosmos3_moz.sh`。它提供不启动 Isaac Sim 的 `--probe-sample /path/to/open_loop/sample.json` 检查，并支持实际机器人注册模块驱动的闭环评测与 WebRTC。完整用法见 RoboLab `policies/cosmos3/README_MOZ.md`。

QX USD 预览/WebRTC 已由用户确认正常，但当前 RoboLab 尚缺实际 QX Articulation/驱动/关节及末端映射、策略相机和动作控制注册。`run_moz.py` 不会用 DROID 兜底；必须提供真实注册模块才能进入仿真闭环。源码新增和 CPU 协议测试不能视为完成真实 checkpoint GPU 推理或 QX 闭环验证。
