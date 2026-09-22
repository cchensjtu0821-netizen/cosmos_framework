# MOZ1 数据集与 Cosmos3 Edge Action Policy 训练说明

本文记录当前 MOZ1 数据集接入 `Cosmos3-Edge-Policy-DROID` 的实际实现，覆盖原始数据字段、样本窗口构造、LeRobot v3.0 兼容边界、`domain_id`、模型输入、训练模块、Rectified Flow loss、正式训练配置和已知限制。

本文描述的是仓库中当前实现，不代表 MOZ1 数据生产端的最终控制语义。尤其是三组 6 维 Cartesian command 目前被直接当作相对 action 使用；在找回生产端的相对位姿变换之前，这只是为了先跑通训练链路的临时假设。

## 1. 当前状态概览

已经由服务器实际验证的链路如下：

- 数据集元数据为 LeRobot v3.0，机器人类型为 `moz1`，原始频率为 30 FPS。
- 数据集共 1,147 个 episode、1,366,169 帧、2 个 task、3 路 H.264 视频。
- 按 episode 划分后，训练集为 1,136 个 episode。
- 16 步 future-action 窗口得到 1,336,083 个有效训练样本。
- 单样本检查输出为：
  - `video`: `[3, 17, 256, 256]`, `uint8`；
  - `action_raw`: `[17, 20]`, `float32`；
  - `action`: `[17, 64]`, `float32`；
  - `raw_action_dim=20`；
  - `domain_id=21`；
  - vision 第 0 帧和 action 第 0 行为 condition。
- 8 GPU、10 个 optimizer step 的 smoke test 已完成，loss 有限，成功保存 `iter_000000010`，最后打印 `Done with training`。
- 正式 TOML 当前配置为 8 GPU、梯度累积 16、10,440 个 optimizer step、每 1,000 步保存 checkpoint。

## 2. 数据集物理组织

数据根目录遵循 LeRobot v3 风格，核心内容可以抽象为：

```text
<MOZ1_ROOT>/
├── data/                       # 按 shard 保存的逐帧 parquet 数据
├── videos/                     # 各相机的 MP4/H.264 数据
│   ├── cam_high/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
├── meta/
│   ├── info.json               # 版本、FPS、字段 schema、视频信息
│   ├── episodes.parquet        # episode 起止索引、长度和统计信息
│   ├── tasks.parquet           # task_index 到任务文本的映射
│   └── stats.json              # LeRobot 原始字段统计
└── norm_stats.json             # 当前 20 维 state/action 的训练归一化统计
```

代码以 `meta/info.json` 中的 `features` 为 schema 真值，通过 LeRobot 的 metadata 和 dataset API 读取 parquet、任务文本和视频，不依赖目录名猜测字段。

### 2.1 数据集总体信息

| 项目 | 当前数据 |
| --- | --- |
| LeRobot codebase version | `v3.0` |
| robot type | `moz1` |
| FPS | 30 |
| episode 数 | 1,147 |
| 原始总帧数 | 1,366,169 |
| task 数 | 2 |
| 相机数 | 3 |
| 单路视频分辨率 | 320 × 240，即 `W=320, H=240` |
| 视频编码 | H.264 |
| 总时长估计 | 约 12.65 小时 |

### 2.2 原始 feature 字段与逻辑维度

下表列出当前 `info.json` 暴露的字段。标量字段在单帧语义上记为 `scalar`；控制字段均按最后一维列出。真正运行时，LeRobot 会根据 `delta_timestamps` 在这些字段前增加时间维。

| 类别 | 字段 | 单帧逻辑维度 | 当前 MOZ1 训练使用 |
| --- | --- | ---: | --- |
| base | `base_cmd_speed` | 3 | 否 |
| base | `base_state_speed` | 3 | 否 |
| video | `cam_high` | `[3, 240, 320]` | 是 |
| video | `cam_left_wrist` | `[3, 240, 320]` | 是 |
| video | `cam_right_wrist` | `[3, 240, 320]` | 是 |
| index | `episode_index` | scalar | 间接用于 episode 定位 |
| index | `frame_index` | scalar | 间接用于帧定位 |
| index | `index` | scalar | 间接用于全局行定位 |
| index | `task_index` | scalar | 是，解析为 task 文本 |
| time | `timestamp` | scalar | LeRobot 用于时间对齐 |
| left arm command | `leftarm_cmd_cart_pos` | 6 | 是，action `[0:6]` |
| left arm command | `leftarm_cmd_joint_pos` | 7 | 否 |
| left arm command | `leftarm_cmd_psi` | 1 | 否 |
| left gripper command | `leftarm_gripper_cmd_pos` | 1 | 是，action `[6]` |
| left arm state | `leftarm_state_cart_pos` | 6 | 是，state `[0:6]` |
| left arm state | `leftarm_state_joint_pos` | 7 | 否 |
| left arm state | `leftarm_state_psi` | 1 | 否 |
| left gripper state | `leftarm_gripper_state_pos` | 1 | 是，state `[6]` |
| right arm command | `rightarm_cmd_cart_pos` | 6 | 是，action `[7:13]` |
| right arm command | `rightarm_cmd_joint_pos` | 7 | 否 |
| right arm command | `rightarm_cmd_psi` | 1 | 否 |
| right gripper command | `rightarm_gripper_cmd_pos` | 1 | 是，action `[13]` |
| right arm state | `rightarm_state_cart_pos` | 6 | 是，state `[7:13]` |
| right arm state | `rightarm_state_joint_pos` | 7 | 否 |
| right arm state | `rightarm_state_psi` | 1 | 否 |
| right gripper state | `rightarm_gripper_state_pos` | 1 | 是，state `[13]` |
| torso command | `torso_cmd_cart_pos` | 6 | 是，action `[14:20]` |
| torso command | `torso_cmd_joint_pos` | 6 | 否 |
| torso state | `torso_state_cart_pos` | 6 | 是，state `[14:20]` |
| torso state | `torso_state_joint_pos` | 6 | 否 |

相机之外，当前真正进入 20 维 policy contract 的只有五个 state 字段和五个 command 字段。joint、psi、base 字段虽然存在于数据集中，但当前训练没有同时塞入模型，避免混合两个控制空间。

### 2.3 当前 20 维 state/action contract

20 维布局固定为：

```text
[0:6]    left Cartesian pose
[6]      left gripper
[7:13]   right Cartesian pose
[13]     right gripper
[14:20]  torso Cartesian pose
```

但是第 0 行和后续 16 行的语义不同：

| 时间行 | 内容 | 语义 |
| --- | --- | --- |
| `row 0` | 五个 state 字段拼接 | 当前时刻的绝对 state |
| `row 1..16` | 五个 command 字段拼接 | 暂时假定三组 pose 是相对 action；gripper 是绝对 command |

因此，一个训练样本的原始 action tensor 实际是：

```text
action_raw[17, 20]
  = concat_time(
      absolute_state_at_t0[1, 20],
      future_action_t0_to_t15[16, 20],
    )
```

这不是“17 个相同语义的 action”。第 0 行是模型 condition，后 16 行才计算 action 生成 loss。

### 2.4 相对 action 的当前假设

当前 adapter 对以下 pose command 字段不做任何变换：

- `leftarm_cmd_cart_pos`；
- `rightarm_cmd_cart_pos`；
- `torso_cmd_cart_pos`。

它们被原样解释为：

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

当前实现没有：

- 用 action 减 state；
- 做 `inv(T_current) @ T_target` 的 SE(3) 变换；
- 做 framewise 或 anchored pose conversion；
- 重编码旋转；
- 推断命令到实际执行之间的时间偏移。

`pose_convention=None` 的 warning 因此是预期行为，不影响程序运行，但明确表示真实控制语义尚未闭环。由于未知相对旋转和参考帧，idle-frame 语义检测也被禁用。

如果以后恢复出真正的 producer-side 转换，应只替换 action chunk 构造逻辑，并为转换后的精确 20 维向量重新计算 `norm_stats.actions`；不能继续沿用当前统计而不验证。

## 3. LeRobot v3.0 是否是 Cosmos Framework 原生适配格式

答案分两层：

1. **存储和读取层面兼容。** Cosmos 当前 action 数据基类直接使用 `lerobot.datasets.lerobot_dataset.LeRobotDataset` 和 `LeRobotDatasetMetadata`，支持从本地 LeRobot 数据根目录读取 metadata、parquet、task 和视频。项目依赖的是仓库固定 revision 的 LeRobot fork，而不是自行实现一套 parquet/video reader。
2. **机器人语义层面不会自动兼容。** 任意 LeRobot v3 数据集并不能仅靠字段存在就自动进入 Cosmos policy。每种机器人仍需明确：相机字段、state/action 字段、action 维度与顺序、位姿 convention、归一化统计、相机布局、FPS、chunk 长度和 `domain_id`。

因此，MOZ1 的现状是“LeRobot v3 容器被通用读取层支持，MOZ1 机器人 schema 由专用 adapter 适配”，不是“Cosmos 官方已有一个通用 MOZ1 adapter”。

当前接入还做了两项离线保护：

- 本地 source 使用 `repo_id="local"` 和 `revision="local"`，避免 metadata 初始化访问 Hugging Face；
- 正式运行应同时设置 Hugging Face/Transformers offline 环境变量，使遗漏的远程依赖直接失败，而不是在内网环境卡住下载。

## 4. 一个训练样本是如何抽取的

### 4.1 episode 级 train/val 划分

当前参数为：

```text
split="train"
split_seed=42
split_val_ratio=0.01
sample_stride=1
chunk_length=16
```

划分按 episode 完成，不会把同一个 episode 的帧分散到 train 和 val。实现先用固定 seed 打乱 episode ID，再取约 1% 为 validation：

```text
全部 episode: 1,147
train episode: 1,136
val episode: 11
```

当前正式训练配置只实例化 `split="train"`，并且 `run_validation=False`，所以这 11 个 validation episode 目前没有进入训练，也没有自动计算 validation loss。

### 4.2 有效窗口数量

对于长度为 `L_e` 的训练 episode，`chunk_length=16`、`stride=1` 时，有效窗口数为：

```text
N_e = max(L_e - 16, 0)
```

之所以减 16，是因为每个起点必须还能取得 17 个 observation/video 时刻，即起点加 16 个未来时刻。服务器实际索引结果为：

```text
训练 episode 原始帧数: 1,354,259
扣除每个 episode 尾部不足 16 帧的起点: 16 × 1,136 = 18,176
有效窗口: 1,354,259 - 18,176 = 1,336,083
保留率: 98.66%
```

这 1,336,083 是 dataset adapter 根据 episode span 实际建立的样本数，不是从 10,440 个训练 step 反推的。

### 4.3 时间窗口

对某个有效起点 `s`：

```text
observation/video timestamps: s + [0, 1, ..., 16] / 30 秒，共 17 帧
action timestamps:            s + [0, 1, ..., 15] / 30 秒，共 16 步
```

读取到的字段窗口是：

- 三路视频：各 `[17, 3, 240, 320]`；
- 五个 state 字段：各 17 个时刻，但最终只使用第 0 个时刻；
- 五个 action 字段：各 16 个时刻；
- `task`：由 `task_index` 经 metadata 解析出的文字任务。

若 task 文本以 `" | "` 包含多个同义描述，adapter 会随机选择一个非空描述作为 caption。

### 4.4 三相机合成

合成在 LeRobot 的 `[T,C,H,W]` 布局下进行：

1. `cam_high` 保持 `[17,3,240,320]`；
2. 左右 wrist 各双线性缩小为 `[17,3,120,160]`；
3. 左右 wrist 横向拼接为 bottom row `[17,3,120,320]`；
4. high 与 bottom row 纵向拼接为 `[17,3,360,320]`；
5. 转为 Cosmos action 视频布局 `[3,17,360,320]` 和 `uint8 [0,255]`；
6. transform 按 `resolution="256"` 做保持比例的 resize 和 reflection padding。

服务器上最终观测到的模型输入为：

```text
video: [3, 17, 256, 256], uint8
```

视图的语义为：上方是 high third-person，相机下方左/右分别是 left/right wrist。该说明也会进入 JSON prompt 的 `cinematography.framing`。

### 4.5 state/action 归一化

`norm_stats.json` 必须包含两个独立的 20 维块：

```text
norm_stats.state.{mean,std,q01,q99,...}
norm_stats.actions.{mean,std,q01,q99,...}
```

当前使用 `action_normalization="quantile"`。对每一维：

```text
offset = (q99 + q01) / 2
scale  = max((q99 - q01) / 2, 1e-8)
normalized = (raw - offset) / scale
```

当前配置没有 clamp，因此超出 `q01..q99` 的值可以落到 `[-1,1]` 外。

由于第 0 行是绝对 state、后 16 行是 action，adapter 会分别使用 `norm_stats.state` 和 `norm_stats.actions`，再沿时间维拼回 `[17,20]`。这一步不能用同一组 action 统计覆盖所有 17 行。

### 4.6 20 维到 64 维

模型的统一 action width 为 64。预处理会：

1. 保留未归一化、未 padding 的 `action_raw=[17,20]`；
2. 归一化真实 20 个通道；
3. 在末尾补 44 个零，得到 `action=[17,64]`；
4. 记录 `raw_action_dim=20`。

训练计算 action loss 时只切片前 20 维，因此 padding 的 44 维不会贡献 loss。模型加噪后也会把 padding 通道重新置零。

### 4.7 prompt 和 classifier-free dropout

当前 `format_prompt_as_json=True`，prompt 包含：

- `cinematography.framing`；
- `actions[].description`；
- duration 和 action time range；
- FPS；
- resize 后的 resolution；
- aspect ratio。

文本 tokenizer 使用本地 Edge Policy snapshot 中的 processor。`cfg_dropout_rate=0.1` 表示约 10% 样本会把 caption 置空，用于 classifier-free guidance 训练。

当前 17 帧、30 FPS 的窗口只有约 0.567 秒。JSON formatter 对 `duration` 使用整数截断、对 action end time 使用四舍五入，因此当前可能出现 `duration="0s"` 而 action range 结束于 `0:01` 的展示差异。这不改变视频/action tensor，但属于后续可清理的 prompt 元数据细节。

### 4.8 SequencePlan

当前训练模式为 `policy`，实际生成计划是：

```text
has_text = True
has_vision = True
has_action = True
has_sound = False
condition_frame_indexes_vision = [0]
condition_frame_indexes_action = [0]
```

因此模型观察 task text、第一帧图像和第 0 行绝对 state，并联合预测后续视频与 16 步 action。第 0 行 action 被当作 clean condition，不进入 action flow-matching loss。

## 5. `domain_id=21` 是怎么设置和使用的

### 5.1 注册

仓库维护一个跨机器人 embodiment ID 表。当前新增注册为：

```python
EMBODIMENT_TO_DOMAIN_ID["moz1"] = 21
EMBODIMENT_TO_RAW_ACTION_DIM["moz1"] = 20
```

adapter 构造基类时传入 `embodiment_type="moz1"`，基类调用 `get_domain_id("moz1")` 得到 21，并在每个样本结果中写入标量 long tensor：

```text
domain_id = tensor(21, dtype=torch.long)
```

### 5.2 在模型里的作用

Edge 模型配置支持 32 个 embodiment domain。`domain_id` 不是普通文字 token，也不是 loss label；它选择 action 输入/输出投影的 domain-specific 参数：

```text
normalized padded action[64]
  -> action2llm(domain_id=21)
  -> hidden_size=2048
  -> shared MoT transformer / generation experts
  -> llm2action(domain_id=21)
  -> predicted action velocity[64]
```

`action2llm` 和 `llm2action` 都是 `DomainAwareLinear`。其权重以 embedding 形式保存，每个 domain 有独立的一组线性权重和 bias。一个样本的 domain ID 会扩展到该样本的所有 action token，MOZ1 因而使用 index 21 对应的输入/输出投影。

`action_modality_embed` 是 action 模态共享 embedding，不按 domain 单独选择。

### 5.3 ID 稳定性要求

domain ID 是 checkpoint 接口的一部分，不能在已有 checkpoint 上随意重新编号。把 MOZ1 从 21 改成别的值，相当于切换到另一套 action projection 参数。新增机器人时还必须保证 ID 小于 `num_embodiment_domains=32`，或同步扩展模型配置和 checkpoint。

## 6. 从样本到一次训练更新

当前训练前向可以概括为：

```text
LeRobot episode/window
  -> 三相机解码与拼接
  -> task JSON prompt + text tokenizer
  -> state/action 分别 quantile normalize
  -> action 20 -> 64 维 padding
  -> SequencePlan(condition: vision[0], action[0])
  -> Wan2.2 VAE 编码 17 帧视频
  -> 对非 condition 的 vision/action 加 RF 噪声
  -> text + vision + action token 打包
  -> Cosmos3VFMNetwork / MoT transformer
  -> vision velocity head + action velocity head
  -> vision/action flow-matching loss
  -> 16 个 micro-step 梯度累积
  -> gradient clip + FusedAdam update + LR scheduler
```

视频 latent 的精确 `[C,T,H,W]` 形状由 Wan2.2 VAE、causal temporal encoding 和实际输入共同决定，不在 adapter 中硬编码。训练入口通过 `encode_exact_durations=[17]` 固定支持当前 17 帧窗口。

## 7. Rectified Flow 训练与 loss 设计

### 7.1 加噪目标

对 clean 数据 `x` 和高斯噪声 `epsilon`，代码使用：

```text
x_sigma = sigma * epsilon + (1 - sigma) * x
v_target = epsilon - x
```

模型输入 `x_sigma` 和 timestep，预测 velocity `v_pred`，基础 loss 是：

```text
L_FM = mean((v_pred - v_target)^2 * noisy_mask * time_weight)
```

其中：

- `noisy_mask = 1 - condition_mask`；
- condition 帧/行的 loss 被乘为 0；
- `train_time_weight="uniform"`，当前 time weight 为均匀权重；
- `normalize_loss_by_active=False`，所以 `.mean()` 的分母仍包含被 mask 的 condition 元素；
- action loss 在计算平方误差后先截取 `:raw_action_dim`，即只监督前 20 维。

### 7.2 vision 和 action 的 sigma

当前是 video batch：

- vision 的训练时间分布是 `waver`；
- 256 resolution 对应 shift 3；
- `independent_action_schedule=False`，action 与 vision 共用同一个 sigma/timestep schedule；
- 因此配置里的 `train_time_action_distribution="logitnormal"` 当前不会单独决定 action sigma；只有开启 independent action schedule 时它才成为独立采样来源。

### 7.3 总 loss

当前 Edge 配置中：

```text
loss_scale = 10.0
action_loss_weight = 10.0
sound_gen = False
lbl.coeff_und = None
lbl.coeff_gen = None
```

所以当前主要总 loss 为：

```text
L_total = 10 * L_vision + 10 * L_action
```

sound loss 不存在；MoE load-balancing auxiliary loss 的两个系数为 `None`，当前也不会加到 total loss。

日志中的 `Loss: 300` 左右是上述加权后的总 loss，不等于 action 的物理单位误差，也不能直接解释成末端位置误差。要判断 policy 是否学到动作，还需要分别记录 `flow_matching_loss_action`、反归一化动作指标和 validation/rollout 指标。

## 8. 模型结构和当前参与训练的模块

### 8.1 主模型结构

当前使用 Edge 而不是 Nano：

| 项目 | 当前值 |
| --- | --- |
| backbone | Nemotron3 Dense VL / Edge |
| hidden size | 2048 |
| intermediate size | 9216 |
| action width | 64 |
| embodiment domain 数 | 32 |
| joint attention | `two_way` |
| vision generation | 开启 |
| action generation | 开启 |
| sound generation | 关闭 |
| precision | bfloat16 |
| activation checkpointing | `full` |
| compile | 开启，compiled region 为 `language` |
| VAE | Wan2.2，spatial compression 16，temporal compression 4 |

`Cosmos3VFMNetwork` 将 text、VAE vision latent、action token 投到相同 hidden space，送入 MoT language model，然后分别用 `llm2vae` 和 domain-aware `llm2action` 解码 velocity。

### 8.2 optimizer allowlist

当前不是 full-model fine-tuning。optimizer 用参数名 substring allowlist，只训练名字包含以下任意字符串的参数：

| allowlist key | 作用 |
| --- | --- |
| `moe_gen` | transformer 中的 generation/MoT expert 参数 |
| `time_embedder` | diffusion/RF timestep embedding |
| `vae2llm` | vision latent 到 hidden 的输入投影 |
| `llm2vae` | hidden 到 vision velocity 的输出投影 |
| `k_norm_und_for_gen` | generation 到 understanding 路径相关的 K normalization |
| `action2llm` | domain-aware action 输入投影 |
| `llm2action` | domain-aware action velocity 输出投影 |
| `action_modality_embed` | action 模态 embedding |

不匹配 allowlist 的 `net` 参数会被设置为 `requires_grad=False`。因此主要冻结项包括：

- 未命中上述 key 的 understanding/reasoner 参数；
- 未命中 allowlist 的共享 transformer 参数；
- text/vision processor；
- Wan2.2 VAE tokenizer 本身。

Edge 配置开启 EMA；EMA 是训练权重的滑动副本，不属于 optimizer 直接更新的普通参数。

### 8.3 checkpoint 初始化

当前从 Edge Policy DCP warm start：

- 加载 `net` 权重；
- 只跳过 `net_ema.`；
- 不跳过 action heads，所以复用 Edge Policy checkpoint 里的 `action2llm`、`llm2action` 和 action embedding；
- `load_training_state=False`，基础 warm start 不继承原 checkpoint 的 optimizer/scheduler step；
- `strict_resume=False`，允许 warm-start key 集合存在差异。

这与早期从通用 Nano base 初始化并重置 action heads 的 DROID recipe 不同。

## 9. 当前正式训练配置：最终生效值

正式入口是：

```text
configs/toml_config/action_policy_moz1_edge_train.toml
```

TOML 只覆盖少量字段，其余值继承 `action_policy_moz1_edge`，后者又继承 DROID Nano action recipe 并替换为 Edge model config。因此不能只看 TOML 判断最终配置。

### 9.1 数据和并行

| 配置 | 最终值 |
| --- | --- |
| GPU 数 | 8 |
| distributed parallelism | FSDP |
| data parallel shard degree | 8 |
| data parallel replicate degree | 1 |
| per-rank dataloader batch | 1 |
| gradient accumulation | 16 |
| effective global batch | `8 × 1 × 16 = 128` samples/update |
| dataloader workers | 4/rank |
| persistent workers | True |
| prefetch factor | 2 |
| episode shuffle seed | 42 |
| sample stride | 1 |
| validation | 关闭 |

### 9.2 optimizer

| 配置 | 最终值 |
| --- | --- |
| optimizer | `FusedAdam` |
| base LR | `1e-5` |
| betas | `[0.9, 0.99]` |
| epsilon | `1e-8` |
| weight decay | `0.05` |
| gradient clip norm | `1.0` |
| non-finite gradient handling | `force_finite=True` |
| NaN step protection | 开启，最多连续 100 次 |

三个 action 接口模块继承了 `5×` LR multiplier：

```text
action2llm
llm2action
action_modality_embed
```

其他 allowlist 模块是 `1×`。

### 9.3 scheduler 的真实学习率

最终 scheduler 为单周期 `LambdaLinear`：

```text
cycle_lengths = [10440]
warm_up_steps = [500]
f_start = [0.0]       # 继承
f_max = [0.4]         # 继承
f_min = [0.0]         # 继承
```

这些 `f_*` 是 optimizer param-group LR 的倍率。因此：

| 参数组 | optimizer base LR | warmup 后峰值 | 训练末尾 |
| --- | ---: | ---: | ---: |
| 普通 allowlist 模块 | `1e-5` | `1e-5 × 0.4 = 4e-6` | 线性降到 0 |
| 三个 action 接口模块 | `5e-5` | `5e-5 × 0.4 = 2e-5` | 线性降到 0 |

所以 TOML 中的 `lr=1e-5` 不是训练期间实际达到的普通模块峰值；还必须乘 scheduler 的 0.4。

### 9.4 step、样本数和“约一轮”

训练集有效窗口数为 1,336,083，effective global batch 为 128：

```text
1,336,083 / 128 = 10,438.148...
ceil(...) = 10,439 optimizer steps
```

当前配置为 10,440 steps：

```text
10,440 × 128 = 1,336,320 sample slots
比有效窗口数多 237 个 slot
```

所以它是“约一轮”，不是严格 epoch。另一个原因是 iterable dataset 会无限循环、按 episode block 洗牌，并把 episode 分给不同 rank/worker；各 shard 长度不完全相等，worker 在自己的 shard 结束后会进入下一次洗牌，训练终止条件完全由 `trainer.max_iter` 控制。

这里的 `Iteration N` 表示第 N 个 optimizer update，不是第 N 个 micro-batch。每个 iteration 内部累计 16 次 forward/backward，每次每张 GPU 取 1 个样本。

### 9.5 checkpoint 与日志

| 配置 | 最终值 |
| --- | --- |
| max optimizer iterations | 10,440 |
| logging interval | 10 |
| checkpoint interval | 1,000 |
| 最后一步保存 | 是；若最后一步不落在保存周期，也会保存 final checkpoint |
| W&B | disabled |

训练初期日志中的：

```text
Hit counter: 12/50
```

只是 `IterSpeed` callback 的前 50 次测速/预热计数，不是总训练进度。总进度应看：

```text
Iteration 12 / trainer.max_iter 10440
```

## 10. 配置文件和启动方式

推荐从仓库根目录启动，所有机器路径均通过环境变量传入，不写死到仓库：

```bash
cd /path/to/cosmos-framework
source .venv/bin/activate

export MOZ1_ROOT=/path/to/moz1_lerobot_v3
export EDGE_HF_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-Policy-DROID-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output/moz1_edge_train

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHONPATH=. torchrun \
  --nproc-per-node=8 \
  --master-port=29517 \
  -m cosmos_framework.scripts.train \
  --sft-toml=cosmos_framework/configs/toml_config/action_policy_moz1_edge_train.toml
```

正式参数应修改 TOML，而不是每次把 `trainer.max_iter=...` 等覆盖项写在命令行。当前关键配置分布为：

| 内容 | 文件 |
| --- | --- |
| 正式步数、累积、scheduler、checkpoint | `configs/toml_config/action_policy_moz1_edge_train.toml` |
| MOZ1 字段映射和 dataloader 参数 | `configs/base/experiment/action/posttrain_config/action_policy_moz1_edge.py` |
| Edge 模型、RF loss 和并行默认值 | `configs/base/experiment/sft/models/edge_model_config.py` |
| 20 维样本构造 | `data/generator/action/datasets/moz1_lerobot_dataset.py` |
| LeRobot episode/window 索引 | `data/generator/action/datasets/cosmos3_action_lerobot.py` |
| resize、prompt、normalize、padding、SequencePlan | `data/generator/action/transforms.py` |
| flow-matching loss | `model/generator/algorithm/loss/flow_matching.py` |
| vision/action 总 loss 汇总 | `model/generator/omni_mot_model.py` |
| action/domain-aware 网络 | `model/generator/mot/cosmos3_vfm_network.py` |
| optimizer 参数筛选和 LR multiplier | `utils/generator/optimizer.py` |
| gradient accumulation 和 iteration 定义 | `trainer/__init__.py` |

## 11. 数据和训练检查命令

### 11.1 单样本检查

```bash
PYTHONPATH=. python \
  -m cosmos_framework.scripts.inspect_moz1_training_sample \
  --root "$MOZ1_ROOT"
```

当前期望至少看到：

```text
dataset_size=1336083
video: shape=(3, 17, 256, 256) dtype=torch.uint8 finite=True
action_raw: shape=(17, 20) dtype=torch.float32 finite=True
action: shape=(17, 64) dtype=torch.float32 finite=True
raw_action_dim=20
domain_id=21
condition_frame_indexes_vision=[0]
condition_frame_indexes_action=[0]
```

### 11.2 训练是否健康

训练链路“能跑通”的最低标准：

- 所有 rank 持续增加相同的 `Iteration`；
- loss 有限，不出现 NaN/Inf；
- 能经过 forward、backward、gradient clip、optimizer step；
- checkpoint 保存完成；
- 进程最后打印 `Done with training`，或正式训练持续推进到目标 step；
- 没有重新访问 Hugging Face；
- 没有 dataloader worker crash、NCCL hang 或 rank 不一致。

但这只证明工程链路成立，不证明 action 语义正确，也不证明 policy 可以安全部署。

## 12. 当前明确的限制与下一步

### 12.1 相对 action 公式仍是首要语义风险

当前训练能运行，是因为把现有 command pose 直接当相对 action。需要从以下至少一种证据恢复真实公式：

- 数据生成代码；
- 控制器/部署端的 action 解码代码；
- 对 raw state/command 做候选 framewise、anchored、SE(3) 变换，并与 `norm_stats.actions` 和逐样本 label 对齐验证。

在这之前，checkpoint 只能视为数据与训练链路实验产物。

### 12.2 当前没有 validation loop

虽然 episode split 保留了 11 个 validation episode，但正式 config 中 `run_validation=False`。后续应增加独立 validation dataset 和至少以下指标：

- `flow_matching_loss_action`；
- 反归一化后的每个 action block MAE/RMSE；
- translation、rotation、gripper 分项误差；
- action horizon 分步误差；
- 闭环 rollout 成功率和安全约束。

### 12.3 训练时长目前仅约一个数据 pass

10,440 optimizer steps 是按当前有效窗口和 batch 估算的一轮量级，不是经过收敛实验确定的最佳训练长度。是否增加 epoch、调整 LR 或解冻更多模块，应以 held-out validation 和 rollout 为依据。

### 12.4 prompt 的短时长显示

17/30 秒的 clip 在 JSON formatter 中可能显示为 `duration="0s"`。如果 Edge Policy 对 duration 文本敏感，应统一短 clip 的 duration/action-range 格式，并在改变格式后做 tokenizer 和训练回归检查。

### 12.5 不要用 total loss 代替动作质量

当前 total loss 同时包含 10 倍 vision loss 和 10 倍 action loss。只有 total loss 下降，无法区分是视频分支改善还是 action 分支改善；正式实验应单独导出 action loss 和反归一化指标。

## 13. 代码依据索引

- MOZ1 字段、20 维布局和临时相对 action 假设：`data/generator/action/datasets/moz1_lerobot_dataset.py`
- episode 划分、窗口数量和官方 LeRobot reader：`data/generator/action/datasets/cosmos3_action_lerobot.py`
- `moz1 -> domain_id 21` 和 raw width 20：`data/generator/action/domain_utils.py`
- Action SFT transform 包装与 iterable episode shuffle：`data/generator/action/datasets/action_sft_dataset.py`
- SequencePlan、JSON prompt、normalization 和 64 维 padding：`data/generator/action/transforms.py`
- quantile normalizer 和 `raw_action_dim`：`data/generator/action/action_processing.py`
- Edge 模型与 Rectified Flow 参数：`configs/base/experiment/sft/models/edge_model_config.py`
- MOZ1 experiment 和 trainable parameter allowlist：`configs/base/experiment/action/posttrain_config/action_policy_moz1_edge.py`
- 当前正式运行 TOML：`configs/toml_config/action_policy_moz1_edge_train.toml`
- Rectified Flow interpolation：`model/generator/diffusion/rectified_flow.py`
- flow-matching MSE：`model/generator/algorithm/loss/flow_matching.py`
- vision/action loss 汇总：`model/generator/omni_mot_model.py`
- domain-aware action projections：`model/generator/mot/domain_aware_linear.py` 与 `model/generator/mot/cosmos3_vfm_network.py`
- optimizer allowlist/LR multiplier：`utils/generator/optimizer.py`
- 梯度累积和 optimizer iteration：`trainer/__init__.py`
