# RoboLab Policy 源码导读：配置、推理、MoT Block 与 Reasoner

本文面向第一次阅读 `cosmos-framework` 模型代码的人，目标不是罗列所有类，而是回答下面几个问题：

1. RoboLab Policy 收到一次请求后，究竟按什么顺序执行？
2. 模型的 `layers`、Transformer `block` 到底藏在哪里？
3. Policy diffusion 的一次 forward 如何经过文本、视觉和 action？
4. `generate_reasoner_text()` 在哪里，如何进行自回归 forward？
5. 配置文件如何找到、读取、实例化并最终决定网络结构？

本文以默认的 `nvidia/Cosmos3-Nano-Policy-DROID` 和
`scripts/action_policy_server_robolab.py` 为主线。代码持续演进时，应以文中列出的函数名为准，
不要只依赖行号。

---

## 1. 先建立正确的目录地图

这个仓库看起来“不像一个模型仓库”，主要因为它把四类职责拆开了：

```text
scripts/
  action_policy_server_robolab.py    对外服务、请求预处理、调用模型

inference/
  args.py                            推理参数以及模型模式
  inference.py                       创建模型、读取 checkpoint、加载权重
  common/config.py                   Python/Hydra、JSON、YAML 配置读取

configs/
  base/config.py                     Hydra 配置总入口
  base/defaults/model_config.py      OmniMoTModelConfig 数据结构
  base/defaults/reasoner.py          VLM/LLM 配置及 LazyCall
  base/experiment/sft/models/
    nano_model_config.py             Nano Policy 的实际模型结构参数
  base/experiment/action/posttrain_config/
    action_policy_droid_nano.py      DROID Policy 训练 recipe

model/generator/
  omni_mot_model.py                  顶层模型：采样、CFG、VAE、Reasoner API
  mot/cosmos3_vfm_network.py         文本/视觉/action 编码、联合网络、输出 head
  mot/unified_mot.py                 真正的 Transformer layers 和 block
  mot/attention.py                   MoT attention 的外围实现
  mot/dot_product_attention.py       packed attention 计算
  mot/domain_aware_linear.py         不同机器人 domain 的 action 投影
  reasoner/qwen3_vl/                 原始 Qwen3-VL 配置和基础模型实现
  reasoner/qwen3_vl_moe/             Qwen3-VL MoE 基础实现
  reasoner/nemotron_3_dense_vl/      Edge/Nemotron 基础实现
  diffusion/samplers/                UniPC、EDM 等 diffusion sampler
  tokenizers/                        video VAE 等 tokenizer
```

最容易产生误解的一点是：

> Policy 的 Transformer block 不在 `model/layers.py` 或 `model/blocks.py`，而在
> `model/generator/mot/unified_mot.py` 的 `MoTDecoderLayer`。

推荐阅读顺序：

```text
action_policy_server_robolab.py
  → omni_mot_model.py
  → cosmos3_vfm_network.py
  → unified_mot.py
```

不要一开始就进入 `reasoner/qwen3_vl/qwen3_vl.py`。那里面包含大量 Hugging Face 风格的基础
VLM 组件；Policy 真正使用的“统一 MoT 层”是在 `unified_mot.py` 中重新组装的。

---

## 2. 一次 Policy 请求的完整调用链

下面先给出总图，后续逐段展开。

```text
客户端 observation
  │
  ▼
RobolabPolicyService.infer()
  │
  ├─ _build_sample()
  │    ├─ prompt → ai_caption
  │    ├─ 当前 RGB → 33 帧 video
  │    ├─ 当前 state → 33×8 action
  │    └─ ActionTransformPipeline
  │
  ├─ _build_data_batch_from_sample()
  │
  ▼
OmniMoTModel.generate_samples_from_batch()
  │
  ├─ get_data_and_condition()
  │    └─ video VAE encode
  ├─ tokenize_text()
  ├─ 构建 SequencePlan / PackedSequence
  ├─ 初始化待去噪的 video/action noise
  │
  ▼
UniPC / EDM sampler 循环
  │
  ├─ conditional denoise
  ├─ unconditional denoise（guidance != 1 时）
  └─ CFG 合成 velocity
       │
       ▼
OmniMoTModel.denoise()
  │
  ▼
Cosmos3VFMNetwork.forward()
  ├─ _encode_text()
  ├─ _encode_vision()
  ├─ _encode_action()
  ├─ language_model(...)
  │    └─ unified_mot._impl_forward()
  │         └─ for decoder_layer in self.layers
  │              └─ MoTDecoderLayer.forward()
  │                   ├─ PackedAttentionMoT
  │                   ├─ generation/understanding 双路径
  │                   └─ MLP 或 MoE
  ├─ _decode_vision() → vision velocity
  └─ _decode_action() → action velocity
       │
       ▼
sampler 更新 noisy action，重复若干步
  │
  ▼
Server 截取前 8 维、删除 state 行、恢复 gripper 约定
  │
  ▼
返回 action[32, 8]
```

### 2.1 服务入口

文件：`scripts/action_policy_server_robolab.py`

程序入口是：

```python
main()
  → tyro_cli(RobolabServerArgs)
  → serve(args)
  → RobolabPolicyService(args)
```

`RobolabPolicyService.__init__()` 做四件关键事情：

1. `_resolve_checkpoint_path()`：本地路径不存在时，将已知模型名映射到 Hugging Face repository；
2. `_build_setup_args()`：把 checkpoint、sampler、compile 等参数变成 `OmniSetupArgs`；
3. `OmniInference.create(setup_args)`：创建模型并加载权重；
4. `_build_transform()`：优先恢复训练时的数据 transform，否则使用默认
   `ActionTransformPipeline`。

WebSocket 收到 observation 后，最终进入：

```python
RobolabPolicyService.infer(obs)
```

它加锁是因为模型采样和随机种子状态不能由多个请求同时修改。启用 profiling 时会走
`_infer_profiled()`，但模型计算路径相同。

### 2.2 请求如何变成模型输入

文件：`scripts/action_policy_server_robolab.py`

核心函数：

```python
RobolabPolicyService._build_sample()
```

默认输入：

```text
prompt                               str
observation/image                    [H,W,3] RGB
observation/joint_position           [7] 或 [T,7]
observation/gripper_position         scalar、[T] 或 [T,1]
```

也支持 wrist + 两路 exterior camera，服务会先把三路图拼成一张组合图。

默认 `action_chunk_size=32`、`use_state=True`，所以服务构造：

```text
video  [3,33,540,640]
  frame 0      当前真实图像
  frame 1..32  全零，占未来生成位置

action [33,8]
  row 0        当前 7D joint + 1D gripper state
  row 1..32    全零，占未来 action 生成位置
```

然后补充：

```text
ai_caption       任务文本
mode             "policy"
domain_id        droid_lerobot 对应的 domain ID
conditioning_fps 15
```

`ActionTransformPipeline` 会做图像 resize/padding、prompt 格式化、action padding 等。
外部 action 是 8 维，但 Nano 配置的 `max_action_dim=64`，因此模型内部 action token 通常是
64 维，后 56 维只是 padding。

`_build_data_batch_from_sample()` 再增加 batch 维并形成 dataloader 风格的字典。

### 2.3 模型生成入口

服务直接调用：

```python
self.model.generate_samples_from_batch(
    data_batch,
    guidance=...,
    seed=...,
    num_steps=...,
    shift=...,
)
```

文件：`model/generator/omni_mot_model.py`

函数：`OmniMoTModel.generate_samples_from_batch()`

这个函数不是“一次 forward”，而是一次完整 diffusion 生成任务。它主要负责：

1. `_prepare_inference_data()`：
   - 根据 batch 创建多模态 `SequencePlan`；
   - 调用 `get_data_and_condition()`；
   - VAE 编码真实条件图像；
   - tokenize prompt；
   - 构建 packed sequence；
   - 创建 vision/action 初始噪声。
2. 创建 sampler；
3. 把 denoise 函数交给 UniPC 或 EDM；
4. 根据 guidance 执行 CFG；
5. 返回采样结束的 vision/action。

### 2.4 为什么 `num_steps=4` 可能调用 8 次网络

默认 `guidance=3.0`，不是 1，因此启用 classifier-free guidance：

```text
每个 sampler step
  ├─ conditional forward：带真实 prompt
  └─ unconditional forward：空/负条件 prompt
       ↓
  velocity = uncond + guidance * (cond - uncond)
```

所以 4 个采样步通常是：

```text
4 conditional + 4 unconditional = 8 次 denoise/network forward
```

对应代码位于：

```text
OmniMoTModel.generate_samples_from_batch()
OmniMoTModel._run_classifier_free_guidance()
OmniMoTModel.denoise()
```

每次得到的是当前噪声状态的 velocity 预测，sampler 再根据 velocity 更新 action latent；
模型不是一次性直接回归最终 32 步 action。

---

## 3. 一次 Policy denoise forward 内部发生什么

### 3.1 顶层网络是怎样创建的

文件：`model/generator/omni_mot_model.py`

函数：

```python
OmniMoTModel.build_net(dtype)
```

核心逻辑可以简化为：

```python
language_model = lazy_instantiate(self.vlm_config.model_instance)

network_config = Cosmos3VFMNetworkConfig(
    vlm_config=language_model.config,
    action_gen=self.config.action_gen,
    action_dim=self.config.max_action_dim,
    ...
)

net = Cosmos3VFMNetwork(language_model, network_config)
```

因此层级关系是：

```text
OmniMoTModel
  └─ net: Cosmos3VFMNetwork
       ├─ language_model: Qwen3VLTextForCausalLM
       │    ├─ model: Qwen3VLTextModel
       │    │    └─ layers: ModuleList[MoTDecoderLayer]
       │    └─ lm_head
       ├─ vae2llm / llm2vae
       ├─ action2llm / llm2action
       ├─ time_embedder
       └─ modality embeddings
```

`torch.device("meta")` 表示先只创建参数形状、不立即分配完整显存，之后 checkpoint loader
再把真实权重加载进来。这也是为什么单看构造函数可能看不到普通的
`model.cuda(); load_state_dict(...)`。

### 3.2 `Cosmos3VFMNetwork` 是多模态适配壳

文件：`model/generator/mot/cosmos3_vfm_network.py`

类：

```python
Cosmos3VFMNetwork
```

它并不是主要 Transformer block，而是负责把不同模态送进同一个 hidden space：

```text
文本 token ID
  → language_model.model.embed_tokens

video VAE latent
  → patchify
  → vae2llm
  → 加 vision modality/timestep embedding

action latent
  → action2llm（DomainAwareLinear）
  → 加 action modality/timestep embedding
```

`Cosmos3VFMNetwork.forward()` 的主流程：

```python
packed_sequence, dtype = self._encode_text(packed_seq)
self._encode_vision(packed_seq, packed_sequence, dtype)
self._encode_action(packed_seq, packed_sequence, dtype)

outputs = self.language_model(
    inputs_embeds=packed_sequence,
    ...
)

self._decode_vision(...)
self._decode_action(...)
```

实际参数名会随 packed metadata 略有不同，但逻辑就是：

```text
不同模态编码 → scatter 到统一 token 序列 → MoT → 按索引取回各模态 → 输出 velocity
```

Policy 关心的两个输出：

```text
preds_vision  视觉 latent velocity
preds_action  action latent velocity
```

其中：

```python
action_hidden_states = last_hidden_state[action.mse_loss_indexes]
preds_action = self.llm2action(action_hidden_states, per_token_domain_id)
```

`llm2action` 也是 `DomainAwareLinear`，不同 embodiment/domain 可以选择不同的投影参数。

### 3.3 PackedSequence 是理解代码的关键

这里不是传统 `[batch, sequence, hidden]` 的单一文本序列。文本、视觉和 action token
会被打包进一个一维 token buffer，同时保留索引：

```text
PackedSequence
  ├─ text.tokens / sequence_indexes
  ├─ vision.tokens / sequence_indexes / noisy_frame_indexes
  ├─ action.tokens / sequence_indexes / noisy_frame_indexes / domain_id
  ├─ position_ids
  ├─ attention mode
  └─ split lengths
```

所以源码中经常出现 scatter、index、pack/unpack。这不是多余的数据搬运，而是在回答：

> 联合序列中哪些位置属于文本，哪些属于条件图像，哪些属于待去噪 action？

`mse_loss_indexes` 指向需要输出 diffusion velocity 的 noisy token；条件 token 可以参与
attention，但不一定需要预测或计算 loss。

---

## 4. 真正的 layers 和 block 在哪里

文件：

```text
model/generator/mot/unified_mot.py
```

### 4.1 layers 的创建

共享初始化函数：

```python
_impl_init()
```

其中明确创建：

```python
self.layers = nn.ModuleList()
for layer_idx in range(config.num_hidden_layers):
    self.layers.append(
        MoTDecoderLayer(...)
    )
```

Qwen3-VL 8B 使用：

```text
Qwen3VLTextForCausalLM
  → Qwen3VLTextModel
  → _impl_init()
  → ModuleList[MoTDecoderLayer]
```

### 4.2 整个 Transformer 的 forward

`Qwen3VLTextModel.forward()` 自己很薄：

```python
def forward(self, *args, **kwargs):
    return _impl_forward(self, *args, **kwargs)
```

`_impl_forward()` 才是真正执行所有层的地方：

```python
for i, decoder_layer in enumerate(self.layers):
    hidden_states, lbl_metadata, kv = decoder_layer(
        hidden_states,
        attention_mask,
        position_embeddings,
        ...
    )
```

最后分别做双路径 final norm：

```text
understanding/reasoner token → self.norm
generation token             → self.norm_moe_gen
```

### 4.3 单个 block

类：

```python
MoTDecoderLayer
```

一个 block 包含：

```text
self.self_attn
self.mlp
self.mlp_moe_gen
self.input_layernorm
self.input_layernorm_moe_gen
self.post_attention_layernorm
self.post_attention_layernorm_moe_gen
```

它不是一个普通的单路径 DecoderLayer，而是两条并行参数路径：

```text
understanding / reasoner 路径
  input_layernorm
  attention 中无 _moe_gen 后缀的 Q/K/V/O
  post_attention_layernorm
  mlp

generation / diffusion 路径
  input_layernorm_moe_gen
  attention 中带 _moe_gen 后缀的 Q/K/V/O
  post_attention_layernorm_moe_gen
  mlp_moe_gen
```

两类 token 在 attention 中联合交互，但根据 token 类型使用不同参数。这就是
Mixture of Transformers（MoT），不是“先跑一个 Reasoner，再跑一个 Policy”。

### 4.4 Attention 在哪里

`MoTDecoderLayer.self_attn` 的类型是：

```python
PackedAttentionMoT
```

也定义在 `unified_mot.py`。它将 understanding token 和 generation token 分开投影
Q/K/V，再按照 attention layout 联合计算。

更底层的 packed/dot-product attention 分布在：

```text
model/generator/mot/attention.py
model/generator/mot/dot_product_attention.py
model/attention/
```

`model/attention/` 是后端层，负责 FlashAttention 2/3、cuDNN、NATTEN 等不同 kernel；
它不是模型结构定义处。阅读模型结构时先不要深入这些后端。

### 4.5 Dense MLP 和 MoE 从哪里来

`LayerTypes` 根据模型 variant 选择基础组件：

```text
qwen3_vl_dense   → Qwen3-VL dense MLP/RMSNorm/Rotary
qwen3_vl_moe     → Qwen3-VL MoE
nemotron_dense   → Nemotron dense 组件
```

Nano 的 `Qwen3-VL-8B-Instruct` 是 dense 路径。Super/更大配置可能使用其他 dense 或
MoE variant。基础类来自：

```text
model/generator/reasoner/qwen3_vl/qwen3_vl.py
model/generator/reasoner/qwen3_vl_moe/qwen3_vl_moe.py
model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py
```

所以这些 `reasoner/` 文件夹不等于“Policy 完全不经过这里”。它们提供 Qwen/Nemotron
基础组件和配置；统一 Policy/Reasoner block 则由 `unified_mot.py` 组装。

---

## 5. Policy 模式的双路径究竟如何工作

理解以下三个概念即可：

### 5.1 understanding token

主要是 prompt 等因果文本 token。它们走不带 `_moe_gen` 后缀的 Reasoner/understanding
参数。

### 5.2 generation token

video/action diffusion token 走带 `_moe_gen` 后缀的 generation 参数，并接收 timestep
embedding。它们的输出最终送到 `llm2vae`、`llm2action`。

### 5.3 一次联合 forward

Policy forward 不是：

```text
Reasoner 生成文字 → 把文字传给 Policy
```

而是：

```text
prompt token ───────────────┐
video latent + timestep ────┼→ 同一个 packed attention / MoT layer stack
action latent + timestep ───┘
                                     │
                                     ├→ vision velocity
                                     └→ action velocity
```

Prompt token 提供语义条件，但默认 RoboLab Policy server 不生成中间思维文本。

---

## 6. Reasoner 代码在哪里，如何生成文字

Reasoner 有两个层次的入口。

### 6.1 高层字符串入口

文件：

```text
model/generator/omni_mot_model.py
```

函数：

```python
OmniMoTModel.generate_reasoner_text(
    inputs: list[str],
    max_new_tokens: int,
    images=None,
    videos=None,
    ...
) -> list[str]
```

它负责：

1. 把字符串组织成 chat messages；
2. 纯文本时调用 `tokenize_text()`；
3. 带图片/视频时调用 `vlm_processor.apply_chat_template()`，得到：
   - `input_ids`
   - `attention_mask`
   - `pixel_values` / `pixel_values_videos`
   - `image_grid_thw` / `video_grid_thw`
4. 调用低层：

```python
self.net.generate_reasoner_text(...)
```

5. 调用 `detokenize_text()` 把 token ID 变回字符串。

### 6.2 低层自回归入口

`self.net` 是 `Cosmos3VFMNetwork`，其 Reasoner 方法最终委托给 language model 的
自回归实现。核心代码集中在：

```text
model/generator/mot/unified_mot.py
```

相关函数/类：

```text
_impl_generate_reasoner_text()
_impl_reasoner_forward()
ReasonerKVCache
PackedAttentionMoT.reasoner_forward()
Qwen3VLTextModel.reasoner_forward()
```

自回归过程是标准的 prefill + decode：

```text
input_ids
  → embed_tokens
  → 可选视觉 tower，将 image placeholder 替换为 image embeddings
  → reasoner prefill forward（整个 prompt）
  → lm_head 得到最后位置 logits
  → greedy / sample 选下一个 token
  → 把新 token 再 forward，复用每层 KV cache
  → 直到 EOS 或 max_new_tokens
```

Reasoner 只使用：

```text
embed_tokens
每层无 _moe_gen 后缀的 attention/norm/MLP
reasoner final norm
lm_head
可选 visual tower
```

它绕过：

```text
vae2llm / llm2vae
action2llm / llm2action
diffusion generation pathway
UniPC / EDM sampler
```

### 6.3 Reasoner 的单层 forward

`_impl_reasoner_forward()` 会遍历同一个 `self.layers`，但调用各层的 reasoner-only
逻辑。每层执行：

```text
reasoner input norm
  → reasoner Q/K/V
  → RoPE
  → 更新/读取 ReasonerKVCache
  → causal attention
  → residual
  → reasoner post-attention norm
  → reasoner MLP
  → residual
```

prefill 时 prompt 内使用 causal mask；单 token decode 时 query 位于最右侧，可以读取
cache 中全部历史 K/V。

### 6.4 RoboLab Policy server 有没有调用它

默认没有。

`action_policy_server_robolab.py` 只调用：

```python
generate_samples_from_batch(...)
```

并且没有传非空 `upsample_task`，所以不会在 Policy 前自动执行 prompt upsampling。
也没有直接调用：

```python
generate_reasoner_text(...)
```

但是 Policy 联合 forward 仍会使用同一个 language model 中的 understanding 路径处理
prompt，因此不能把所有“reasoner/understanding”权重简单删除。

---

## 7. 配置文件如何被读取

这里同时存在两套配置来源：

1. 训练/DCP 使用 Python + Hydra ConfigStore；
2. 导出的 Hugging Face/safetensors checkpoint 使用 `config.json`。

### 7.1 Server 参数先变成 `OmniSetupArgs`

`RobolabPolicyService._build_setup_args()`：

```text
RobolabServerArgs
  → OmniSetupOverrides.model_validate(...)
  → overrides.build_setup()
  → OmniSetupArgs
```

默认配置入口定义在：

```text
inference/common/args.py
DEFAULT_CONFIG_FILE = "cosmos_framework/configs/base/config.py"
```

### 7.2 `OmniInference.create()` 如何分流

文件：

```text
inference/inference.py
```

核心类：

```python
OmniInference
```

其 `_create()` 根据 checkpoint 类型分两条路。

#### DCP + Python module config

```text
setup_args.experiment
setup_args.config_file
setup_args.experiment_overrides
  → load_model_from_checkpoint(...)
  → import configs/base/config.py
  → Hydra 选择 experiment
  → instantiate model
  → 加载 DCP 权重
```

#### 导出的 HF/safetensors checkpoint

```text
checkpoint/config.json
  → setup_args.load_model_config_dict()
  → Cosmos3OmniConfig(model=model_dict)
  → Cosmos3OmniModel.from_pretrained_dcp(...)
  → 构造 OmniMoTModel
  → 加载 safetensors
```

RoboLab OSS server 默认期待第二种：包含 `config.json` 和 safetensors 的 consolidated
checkpoint。

### 7.3 Python/Hydra 配置如何加载

文件：

```text
inference/common/config.py
```

函数：

```python
load_config(config_file, experiment, overrides)
```

执行顺序：

```text
importlib.import_module(config module)
  → config_module.make_config()
  → Hydra override:
       experiment=<experiment>
       其他 experiment_overrides
```

`configs/base/config.py` 会注册各 experiment，其中包括：

```text
action_policy_droid_nano
```

这个 recipe 位于：

```text
configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py
```

### 7.4 Nano Policy 的结构参数从哪里来

DROID recipe 先：

```python
model.config = copy.deepcopy(NANO_MODEL_CONFIG)
```

`NANO_MODEL_CONFIG` 位于：

```text
configs/base/experiment/sft/models/nano_model_config.py
```

关键值：

```text
action_gen=True
vision_gen=True
max_action_dim=64
state_ch=48
latent_downsample_factor=16
joint_attn_implementation="two_way"
precision="bfloat16"
vlm_config.model_name="Qwen/Qwen3-VL-8B-Instruct"
vlm_config.layer_module="Qwen2MoTDecoderLayer"
```

最关键的是 `vlm_config.model_instance`：

```python
LazyCall(Qwen3VLTextForCausalLM)(
    config=LazyCall(create_vlm_config)(
        base_config=LazyCall(Qwen3VLMoTConfig.from_json_file)(
            json_file=".../Qwen3-VL-8B-Instruct.json"
        ),
        layer_module="MoTDecoderLayer",
        ...
    )
)
```

这条 LazyCall 链解释了为什么搜索普通的：

```python
Qwen3VLTextForCausalLM(...)
```

可能找不到直接调用。对象是在：

```python
lazy_instantiate(self.vlm_config.model_instance)
```

时递归实例化的。

### 7.5 基础 Qwen JSON 配置

Nano 的层数、hidden size、attention heads 等基础 Transformer 参数来自：

```text
model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json
```

读取函数：

```python
Qwen3VLMoTConfig.from_json_file(...)
```

然后 `create_vlm_config()` 将 `layer_module`、QK norm、是否冻结 understanding 路径等
override 覆盖到基础配置。

因此配置合成可以概括为：

```text
Qwen JSON 基础结构
  + NANO_MODEL_CONFIG 的 MoT/diffusion/action 参数
  + action_policy_droid_nano recipe 的训练参数
  + CLI experiment_overrides
  + 推理期 parallelism/compile/sampler override
  = 最终运行配置
```

### 7.6 为什么还要读取 checkpoint metadata

模型结构由 checkpoint 的 `config.json` 恢复后，Server 还会尝试读取：

```text
checkpoint.json
```

`_load_training_config()` 使用其中的：

```text
config_file
experiment
experiment_overrides
```

恢复训练时的 dataset transform，主要是为了找回：

```text
resolution
chunk_length
fps
prompt JSON 格式
```

这一步主要决定输入预处理，不是第二次创建模型。如果 metadata 不可用，Server 使用默认
`ActionTransformPipeline`。

---

## 8. 权重如何对应到双路径

从 state dict 名称可以快速判断路径：

```text
无 _moe_gen 后缀
  → understanding / reasoner 路径

有 _moe_gen 后缀
  → diffusion generation 路径
```

典型参数：

```text
language_model.model.layers.N.self_attn.q_proj
language_model.model.layers.N.mlp
language_model.model.layers.N.input_layernorm
  → Reasoner/understanding

language_model.model.layers.N.self_attn.q_proj_moe_gen
language_model.model.layers.N.mlp_moe_gen
language_model.model.layers.N.input_layernorm_moe_gen
  → Policy/video/action diffusion generation
```

此外：

```text
language_model.model.embed_tokens  文本 embedding
language_model.lm_head             Reasoner 文本输出
vae2llm / llm2vae                  视觉输入输出
action2llm / llm2action            action 输入输出
```

某些 checkpoint 可以通过 `exclude_reasoner_weights_from_checkpoint` 不重复保存基础
Reasoner 权重，但这通常意味着它们从预训练 backbone 另行加载，并不表示 Policy forward
完全不需要 understanding 路径。

---

## 9. 最小阅读路线

如果只想搞懂 Policy，不要通读整个 `model/`。按下面顺序阅读即可：

1. `scripts/action_policy_server_robolab.py`
   - `RobolabPolicyService.__init__`
   - `_build_sample`
   - `infer`
2. `model/generator/omni_mot_model.py`
   - `__init__`
   - `set_up_tokenizers`
   - `build_net`
   - `_prepare_inference_data`
   - `generate_samples_from_batch`
   - `_run_classifier_free_guidance`
   - `denoise`
3. `model/generator/mot/cosmos3_vfm_network.py`
   - `__init__`
   - `_encode_text`
   - `_encode_vision`
   - `_encode_action`
   - `_decode_action`
   - `forward`
4. `model/generator/mot/unified_mot.py`
   - `LayerTypes`
   - `PackedAttentionMoT`
   - `MoTDecoderLayer`
   - `_impl_init`
   - `_impl_forward`
   - `Qwen3VLTextModel`
5. Reasoner 额外阅读：
   - `OmniMoTModel.generate_reasoner_text`
   - `_impl_generate_reasoner_text`
   - `_impl_reasoner_forward`
   - `ReasonerKVCache`
6. 最后才按需看：
   - `reasoner/qwen3_vl/qwen3_vl.py`
   - `mot/dot_product_attention.py`
   - `model/attention/` 各 CUDA backend

---

## 10. 调试时建议打断点的位置

一次 Policy 请求：

```text
RobolabPolicyService.infer
RobolabPolicyService._build_sample
OmniMoTModel.generate_samples_from_batch
OmniMoTModel._prepare_inference_data
OmniMoTModel.denoise
Cosmos3VFMNetwork.forward
unified_mot._impl_forward
MoTDecoderLayer.forward
Cosmos3VFMNetwork._decode_action
```

Reasoner：

```text
OmniMoTModel.generate_reasoner_text
Cosmos3VFMNetwork.generate_reasoner_text
unified_mot._impl_generate_reasoner_text
unified_mot._impl_reasoner_forward
PackedAttentionMoT.reasoner_forward
```

建议观察：

```text
type(self.model)
type(self.model.net)
type(self.model.net.language_model)
len(self.model.net.language_model.model.layers)
packed_seq.text.sequence_indexes
packed_seq.vision.sequence_indexes
packed_seq.action.sequence_indexes
packed_seq.action.mse_loss_indexes
out["preds_action"]
```

---

## 11. 最终心智模型

可以把整个模型记成三层：

```text
第一层：OmniMoTModel
  管 sampler、CFG、VAE、tokenizer、Reasoner 高层 API

第二层：Cosmos3VFMNetwork
  把文本、视觉、action 映射到统一 hidden space，
  调 language_model，再映射回各模态 velocity

第三层：unified_mot
  真正的 Transformer layers、MoTDecoderLayer、attention、MLP/MoE，
  同时容纳 Reasoner/understanding 与 diffusion generation 两套路径
```

默认 RoboLab Policy 的一句话总结：

> Server 将当前图像、机器人 state 和任务文本打包成联合多模态序列；UniPC/EDM 多次调用
> MoT 网络预测 action velocity；MoT 的 understanding 路径处理 prompt，generation 路径
> 处理带 timestep 的视觉/action token；最后 sampler 从噪声逐步得到 32×8 action。

默认 Reasoner 的一句话总结：

> `generate_reasoner_text()` 使用相同 MoT layer 中无 `_moe_gen` 后缀的 understanding
> 权重，通过 prompt prefill、逐 token decode 和每层 KV cache 生成文本；它不经过 diffusion
> sampler，也不使用 action/vision velocity head。
