# MOZ1 Edge Policy 的 RoboLab 闭环服务

此服务与 RoboLab 的 `scripts/run_cosmos3_moz_closed_loop.sh` 配合：每次请求传入当前仿真三路 RGB、20 维实测 state 和任务文字；服务使用指定后训练 DCP 生成 16×20 action。RoboLab 执行其中 N 步，再读取执行后的新观测重新请求。不得在这一路径固定使用 sample.json 或历史 action。

在 Cosmos 仓库根目录，使用原训练环境和原归一化文件：

```bash
export MOZ1_ROOT=/path/to/dataset
export EDGE_HF_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-Policy-DROID-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export MOZ1_SERVER_CKPT=/path/to/domain_projector_run/checkpoints/iter_XXXXXXXXX
export MOZ1_STATS_PATH="$MOZ1_ROOT/norm_stats.json"
export MOZ1_SERVER_GPU=2
bash cosmos_framework/scripts/run_moz1_closed_loop_server.sh --check-only
bash cosmos_framework/scripts/run_moz1_closed_loop_server.sh
```

GPU 编号按实际空闲卡调整。check-only 仅校验文件与统计量，不证明 CUDA 权重加载成功。新闭环启动器要求显式 checkpoint；通用 MOZ1 启动器的默认 run 改为 domain_projector_train，不再默认选择另一条 gpu7_train。

默认 `num_steps=30, guidance=1, shift=3, seed=0`，非 EMA；通过 `MOZ1_NUM_STEPS`、`MOZ1_GUIDANCE`、`MOZ1_SHIFT`、`MOZ1_GENERATION_SEED` 覆盖。跨请求默认使用从 seed 初始化的 RNG，`--deterministic-seed` 可用于重复输入对照。不要为追求请求频率未经对照减少去噪步数。

训练的 action 频率保持 30 Hz。控制频率与模型去噪步数不同：服务每次仍返回 16 步，RoboLab 的 execute-horizon=N 决定仿真时间中的重规划频率 30/N Hz。同步评测在模型请求期间暂停物理推进，墙钟吞吐取决于模型耗时，并不保证实时。

输出已经逆训练 quantile 归一化，并移除了 state 行；服务不额外加 state。MOZ1 adapter 当前直接读取 command 标签，没有执行 DeltaActions。原统计量与训练 checkpoint 必须配套，不得在部署时擅自更换。

响应增加可选 `request_id` 回显、输入 float32 state 的 `state_sha256`、准备/生成耗时 `timing`。不缓存前一请求观测或 action，不改变采样数学或 action 维度。旧客户端可以忽略这些附加字段。

RoboLab 的闭环客户端保存每轮实际 state、图像 hash、完整预测、执行 horizon、生成 seed、请求耗时以及 request/session ID；任务是否成功由 RoboLab 原任务判定，不能由服务返回 action 成功来代替。

QX frame/TCP、夹爪开度映射与训练相机仍需使用实际资产标定。开发机未加载真实 DCP 或 Isaac Sim，CPU 协议测试只验证调用/数据转换；模型质量和任务成功需要服务器运行确认。
