# Cosmos3 Edge/Nano Policy 全链路输入输出维度

## 1. 范围和证据等级

本文沿 RoboLab Policy 的完整 forward 路径，对比 Cosmos3 Edge Policy 和 Nano
Policy 的 tensor shape。为避免把模型架构和请求布局混为一谈，使用三种标记：

- **已观测**：来自服务器 ONNX/OMG 数据或仓库 profiling 记录；
- **代码确定**：由配置常量或 shape 变换公式直接确定；
- **推导**：由已观测的上下游 shape 和代码公式反推，仍建议用对应导出的
  `.onnx.json` manifest 最终确认。

当前比较使用相同的非 canonical 32-step DROID 请求：1 行当前 state 加 32 行未来
action，共 33 个 pixel frames/action rows。Edge 部分以 2026-08-04 保存的完整
server profile 为准；Nano 采用仓库既有 profiling 记录，缺少 Nano 实机 profile 的
字段明确标为推导或待确认。

## 2. Edge 和 Nano 的模型架构差异

| 项目 | Cosmos3 Edge Policy | Cosmos3 Nano Policy | 来源 |
| --- | ---: | ---: | --- |
| text/MoT hidden size | 2,048 | 4,096 | 模型 JSON，代码确定 |
| MLP intermediate size | 9,216 | 12,288 | 模型 JSON，代码确定 |
| hidden layers | 28 | 36 | 模型 JSON，代码确定 |
| query attention heads | 16 | 32 | 模型 JSON，代码确定 |
| KV heads | 8 | 8 | 模型 JSON，代码确定 |
| head dimension | 128 | 128 | 模型 JSON，代码确定 |
| action model width | 64 | 64 | `max_action_dim`，代码确定 |
| VAE latent channels | 48 | 48 | Wan2.2 VAE，代码确定 |
| VAE temporal compression | 4 | 4 | Wan2.2 VAE，代码确定 |
| VAE spatial compression | 16 | 16 | Wan2.2 VAE，代码确定 |
| VFM spatial patch | 2×2 | 2×2 | `patch_spatial=2`，代码确定 |

因此 Edge 与 Nano 的 video/action 外部 latent 布局可以相同；核心区别主要体现在
text embedding、投影结果和 MoT hidden state 的最后一维。模型大小不决定 action
chunk：Edge checkpoint 也可以运行当前 32-step 非 canonical 请求。

## 3. 当前两个具体 32-step 布局总览

| 阶段 | Edge：实际 ONNX 布局 | Nano：profiling 布局 |
| --- | --- | --- |
| WebSocket observation image | `[540,640,3]` uint8，已观测 | `[540,640,3]` uint8（server 默认） |
| `build_sample.video` | `[3,33,544,736]` uint8，已观测 | `[3,33,544,736]`，既有 profiling |
| VAE pixel input | `[1,3,33,544,736]`，batch 维由代码增加 | `[1,3,33,544,736]` |
| VAE 原始输出（含 padding） | `[1,48,9,34,46]`，由压缩公式确定 | `[1,48,9,34,46]` |
| 去 padding 后 video latent | `[1,48,9,33,40]`，已观测 | Nano 实机 profile 待确认 |
| patch grid | `[9,17,20]` | `[9,17,23]` |
| vision patch tokens | `9*17*20=3,060` | `9*17*23=3,519` |
| runtime conditional text | raw `[105]` -> packed `[107]` -> `[107,2048]` | 待 Nano profile |
| action model input | `[33,64]` | `[33,64]` |
| runtime packed VFM sequence | `107+3060+33=3,200`，已观测 | 待 Nano profile |
| returned robot action | `[32,8]` | `[32,8]` |

Edge runtime profile 与当前 ONNX/OMG 记录不是同一个 text fixed layout：profile 的
conditional packed sequence 是 3,200，而 ONNX/OMG 使用 108 个 prompt embeddings，
所以是 3,201。两组数据必须分开记录，不能用其中一组覆盖另一组。

## 4. 完整 Policy forward 各阶段 shape

```text
WebSocket observation
  -> build_sample
  -> ActionTransformPipeline
  -> build_batch
  -> prepare_data_total
       get_data_and_condition -> vae_encode
       tokenize_text
       initial_pack
       initialize vision/action noise
  -> sampler_total
       conditional_forward
       unconditional_forward (guidance != 1)
       CFG velocity
  -> action_postprocess
  -> vae_decode (only --decode-video)
  -> WebSocket response
```

### 4.1 WebSocket observation

默认 `joint_pos` 请求：

| 输入 | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `observation/image` | uint8 | `[540,640,3]` | 单张 RGB 当前观测 |
| `observation/joint_position` | FP32 | `[7]` 或 `[T,7]` | 当前/历史关节位置 |
| `observation/gripper_position` | FP32 | scalar、`[T]` 或 `[T,1]` | 当前/历史夹爪位置 |
| `prompt` | string | scalar | 任务文本 |

Edge/Nano 在这一层没有模型 hidden-size 差异。

### 4.2 `build_sample`

32-step、`use_state=True` 时：

```text
T_pixel  = action_chunk_size + 1 = 33
T_action = current_state + future_actions = 1 + 32 = 33
```

| tensor | dtype | shape | 数据内容 |
| --- | --- | --- | --- |
| `video` | uint8 | `[3,33,544,736]` | resize/reflection padding 后；frame 0 当前图，后 32 帧为占位 |
| `action` | FP32 | `[33,64]` | row 0 当前 state，后 32 行未来 action；已 pad 到模型宽度 |

原始 DROID action 8 维为 7 维 joint position 加 1 维 gripper。进入模型前 padding
到 64，后 56 个 channel 为零。

### 4.3 `ActionTransformPipeline`

该阶段执行视频 resize/reflection padding、prompt 格式化/tokenize、sequence plan
构造，以及 action normalize/padding。

#### Edge 当前实测 transform 布局

完整 profile 已记录：

```text
video  [3,33,544,736] uint8
action [33,64] FP32
```

WebSocket 原图 `[540,640,3]` 经 transform 后变成 `[3,33,544,736]`。这里的
544×736 是送入 VAE 的 padded bucket；`build_batch`/VAE 调用时再增加 batch 维：

```text
[1,3,33,544,736]
```

### 4.4 `vae_encode`：pixel video 到 video latent

Wan2.2 VAE 的公式为：

```text
C_latent = 48
T_latent = 1 + floor((T_pixel-1)/4)
H_latent = H_pixel/16
W_latent = W_pixel/16
```

33 帧的时间压缩不是简单的 `33/4`，而是 causal VAE 公式：

```text
T_latent = 1 + floor((33-1)/4) = 9
```

| 模型/布局 | VAE 输入 | VAE 输出 |
| --- | --- | --- |
| Edge profile：VAE 原始输出 | `[1,3,33,544,736]` | `[1,48,9,34,46]` |
| Edge profile：去 padding | `[1,48,9,34,46]` | `[1,48,9,33,40]` |
| Nano 既有 profiling | `[1,3,33,544,736]` | `[1,48,9,34,46]`（未记录后续 crop） |

Edge 的第二步不是 VAE 再压缩，而是 `_remove_padding_from_latent` 裁剪。它读取
`image_size=[target_h,target_w,orig_h,orig_w]` 中的原始 540×640，并按空间压缩因子
16 计算 `floor(540/16)=33`、`floor(640/16)=40`，从左上角把 34×46 裁成 33×40。
因此不能由最终 33×40 反推 VAE pixel 输入是 528×640。第一个 latent frame 对应
conditioning 图像；其余 8 个 latent frames 是未来/noisy vision 区域。

### 4.5 vision patchify 和 `encode_vision`

VFM 使用 2×2 spatial patch。若 latent H/W 不是 2 的整数倍，会先 pad 到偶数。
时间维不 patchify。

#### Edge

```text
latent             [1,48,9,33,40]
spatial pad        H:33->34, W:40
patch grid         [T,Hpatch,Wpatch] = [9,17,20]
patch feature      48*2*2 = 192
packed vision      [3060,192]
vae2llm output     [3060,2048]
```

#### Nano

```text
latent             [1,48,9,34,46]
spatial pad        none
patch grid         [9,17,23]
patch feature      192
packed vision      [3519,192]
vae2llm output     [3519,4096]
```

### 4.6 text tokenize 和 `encode_text`

Edge runtime profile 同时记录了 conditional 和 unconditional：

| 分支 | tokenizer 原始 IDs | VFM `text_ids` | `encode_text` 输出 |
| --- | --- | --- | --- |
| conditional | `[105]` int64/CPU | `[107]` int64/CUDA | `[107,2048]` BF16 |
| unconditional | `[17]` int64/CPU | `[19]` int64/CUDA | `[19,2048]` BF16 |

VFM 比 tokenizer 结果多 2 个特殊 token。另一个 fixed-layout Edge ONNX 使用
`prompt_embeddings=[108,2048]`；它属于不同的导出请求布局，不等同于本次 profile
的 conditional 107。Nano 的 embedding 最后一维由架构确定为 4096，但实际 token
长度仍需 Nano profile/manifest。

兼容 ONNX 已把图内 embedding `Gather` 移到宿主侧，因此输入是
`prompt_embeddings`。Edge 与 Nano tokenizer/backbone 不同，任意新 prompt 或
unconditional prompt 的 token 数不保证仍为 108，不能让固定 conditional ONNX
直接接收不同长度。

### 4.7 action padding、noise 和 `encode_action`

| 阶段 | Edge | Nano |
| --- | --- | --- |
| raw DROID action | `[33,8]` | `[33,8]` |
| pad to model width | `[33,64]` | `[33,64]` |
| real/noisy rows | row 0 condition；rows 1..32 noisy | 相同 |
| `action2llm` output | `[33,2048]` | `[33,4096]` |

只有前 8 个 channel 是真实 DROID action；padding channel 在 noise/velocity 中保持
为零。

Edge profile 还记录了 sampler flatten 边界：

| tensor | dtype | shape | 数量来源 |
| --- | --- | --- | --- |
| `vision_x0` | FP32 | `[1,48,9,33,40]` | `570240` elements |
| `action_x0` | BF16 | `[33,64]` | `2112` elements |
| `initial_noise` | FP32 | `[572352]` | `570240+2112` |
| `condition_reference` | FP32 | `[572352]` | 同一 joint layout |
| `condition_mask` | FP32 | `[572352]` | 同一 joint layout |
| per-call `joint_noise` | FP32 | `[572352]` | sampler 当前 state |
| per-call `timestep` | int64 | `[1,1]` | 当前 diffusion step |

### 4.8 packed MoT sequence

#### Edge runtime conditional

```text
text    107
vision  9*17*20 = 3060
action  33
total   107+3060+33 = 3200
hidden  [3200,2048]
```

Edge runtime unconditional 为 `19+3060+33=3112`，hidden 为 `[3112,2048]`。
两路的 full-only 部分都为 `3060+33=3093`；变化的只是 causal text 部分。

#### Nano（既有数据推导示例，非实机定稿）

```text
text    108
vision  9*17*23 = 3519
action  33
total   108+3519+33 = 3660
hidden  [3660,4096]
```

Edge fixed ONNX/OMG 的 3,201 则来自 `108+3093`。这也解释了 OMG v4 报错中的
`scaleShape[0]=3093`：它是 vision 3,060 加 action 33 的 full-only token 数。

### 4.9 单次 denoise/VFM ONNX 边界

#### Edge runtime profile：实际 PyTorch VFM 边界

| 分支/字段 | dtype | shape |
| --- | --- | --- |
| conditional `text_ids` | int64 | `[107]` |
| unconditional `text_ids` | int64 | `[19]` |
| `vision_tokens` | BF16 | `[1,48,9,33,40]` |
| `action_tokens` | BF16 | `[33,64]` |
| `vision_timesteps` | FP32 | `[2720]` |
| `action_timesteps` | FP32 | `[32]` |
| `preds_vision` / `vision_velocity` | BF16 | `[1,48,9,33,40]` |
| `preds_action` / `action_velocity` | BF16 | `[33,64]` |

`vision_timesteps=2720` 来自 8 个 noisy latent frames × 17×20 spatial patches；
conditioning latent frame没有 timestep。动作输入/输出保留 33 行，只有 timestep
对应 32 个未来/noisy rows。

#### Edge fixed ONNX：用户保存的导出接口

| 输入 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `prompt_embeddings` | FP32 | `[108,2048]` | 宿主完成 embedding lookup |
| `video_latent` | FP32 | `[1,48,9,33,40]` | 原始 ONNX rank-5 输入 |
| `action_latent` | FP32 | `[33,64]` | 1 condition row + 32 noisy rows |
| `vision_timestep` | FP32 | `[2720]` | 推导：`8*17*20` noisy vision patches |
| `action_timestep` | FP32 | `[32]` | 32 noisy action rows |

| 输出 | dtype | shape |
| --- | --- | --- |
| `vision_velocity` | FP32 | `[1,48,9,33,40]` |
| `action_velocity` | FP32 | `[32,64]`（以该 ONNX manifest 为准） |

这里必须保留一个接口差异：runtime profile 的网络输出为 `[33,64]`，而用户保存的
fixed ONNX 记录为 `[32,64]`。这表示 export wrapper 在边界上截掉了 condition row，
或者两次记录来自不同导出版本；不能仅凭 runtime profile 把 ONNX 表改成 33。最终
应由对应文件的 `.onnx.json` manifest 和 graph output 再确认。

兼容 rewrite 会固定 `action_domain_id=8`，因此它不再是外部输入；还可能把 batch=1
的 vision I/O 从 `[1,48,9,33,40]` 降为 `[48,9,33,40]`。应区分原始 denoiser
ONNX 和 rank-lowered deployment ONNX 的接口。

#### Nano：相同 32-step、profiling spatial layout

| 输入 | 导出 dtype | shape | 证据 |
| --- | --- | --- | --- |
| `prompt_embeddings` | FP32 | `[108,4096]` | token length 已记录；hidden width 由模型确定 |
| `video_latent` | FP32 | `[1,48,9,34,46]` | profiling 已记录 |
| `action_latent` | FP32 | `[33,64]` | profiling 已记录 |
| `vision_timestep` | FP32 | `[3128]` | 推导：`8*17*23` |
| `action_timestep` | FP32 | `[32]` | 32 noisy action rows |

| 输出 | 导出 dtype | shape | 证据 |
| --- | --- | --- | --- |
| `vision_velocity` | FP32 | `[1,48,9,34,46]` | 与 vision latent layout 相同，代码确定 |
| `action_velocity` | FP32 | `[33,64]` 或 wrapper 截取后的 `[32,64]` | 需 Nano manifest 确认 |

Nano 的 `prompt_embeddings`、timestep 和 output 表中含有代码推导值；在服务器导出
Nano ONNX 后，应直接用 `.onnx.json` manifest 替换“推导”状态。

运行时 Policy 通常使用 BF16 latent/hidden；上表的 FP32 是当前 ONNX export 选择
`--export-dtype float32` 后的边界类型。

### 4.10 sampler/CFG

UniPC/EDM 把 vision/action noisy state flatten 后共同更新。每个采样步：

1. conditional forward 使用上述 fixed conditional text layout；
2. `guidance != 1` 时再运行 unconditional forward；
3. CFG 合成两路 velocity；
4. 更新 vision latent 和 action latent。

默认 `num_steps=4`、`guidance=3` 时共有 4 次 conditional 和 4 次 unconditional
network forward。ONNX 只包含其中一次 VFM forward，不包含 sampler loop 或 CFG。

### 4.11 action postprocess 和 WebSocket 输出

模型输出只保留前 8 个真实 action channels，移除当前 state/history row，并恢复
gripper 方向：

```text
VFM action velocity [33,64] BF16
model action        [33,8]  FP32/CUDA
drop history row    [32,8]
WebSocket action    [32,8]  FP32/NumPy CPU
```

Edge 和 Nano 的这部分完全相同。

### 4.12 可选 `vae_decode`

只有 `--decode-video` 开启时执行。Wan2.2 causal VAE 的逆时间公式为：

```text
T_pixel = (T_latent-1)*4+1
```

| 模型/布局 | decoder 输入 | decoder 输出 | WebSocket video |
| --- | --- | --- | --- |
| Edge cropped latent | `[1,48,9,33,40]` | 具体 padding/restore 路径需 decode profile | 未观测（本次 `decode_video=false`） |
| Nano profiling latent | `[1,48,9,34,46]` | `[1,3,33,544,736]`（公式值） | `[33,544,736,3]`（推导） |

## 5. canonical Edge 16-step 与当前 32-step 的区别

仓库工作记录指出 canonical Edge Policy 使用 16 个未来 actions、17 个 video
frames、conditioning FPS 5 和 JSON prompt。若 transform/padding/crop 规则不变，
仅改变时间/action chunk，则代码公式给出：

```text
VAE pixel input   [1,3,17,544,736]
raw VAE latent    [1,48,5,34,46]   # 1+(17-1)/4 = 5
cropped latent    [1,48,5,33,40]
patch grid        [5,17,20]
action latent     [17,64]
action timestep   [16]
action velocity   [16,64]
returned action   [16,8]
```

这里没有给出 canonical prompt token 数和 `vision_timestep` 的最终固定值，因为 JSON
prompt 经 Edge tokenizer 后的实际长度必须从对应请求 profiling/manifest 获取。

## 6. 仍需服务器确认的项目

1. Nano ONNX 的完整 `.onnx.json` manifest，特别是 prompt length、两类 timestep
   和 `action_velocity` shape。
2. Edge fixed ONNX 的 `action_velocity` 是否确为 `[32,64]`，以及 export wrapper 在
   哪一步从 runtime `[33,64]` 截去 condition row。
3. fixed conditional ONNX 不能默认兼容本次 profile 的 unconditional `[19]` layout。
4. rank-lowered ONNX 的最终 graph I/O，确认 vision batch 维是否已删除。

## 7. 代码和文档依据

- Server sample/response：`scripts/action_policy_server_robolab.py`
- Action transform：`data/generator/action/transforms.py`
- 分辨率 bucket：`data/generator/utils.py`
- Wan2.2 VAE：`model/generator/tokenizers/wan2pt2_vae_4x16x16.py`
- VFM patch/pack/head：`model/generator/mot/cosmos3_vfm_network.py`
- Nano config：`configs/base/experiment/sft/models/nano_model_config.py`
- Edge config：`configs/base/experiment/sft/models/edge_model_config.py`
- Nano profiling shape：`docs/ROBOlab_POLICY_INFERENCE_PROFILING.md`
- ONNX boundary：`scripts/export_action_policy_onnx.py`
