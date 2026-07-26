# RoboLab Cosmos Policy 推理耗时与 FLOPs 统计

## 统计口径

- 启用参数：`--enable-module-profile`。
- 输出文件：`<profile-output-dir>/module_profile.jsonl`。
- `cpu`、`cuda` 保留原有耗时统计；新增 `flops`，每项包含整数 `total` 和 `tflops`。
- 理论 FLOPs 采用一次乘加等于 2 FLOPs；按请求的实际 token 数、packed sequence 和模型配置计算。
- 子模块 FLOPs 自动向其活动中的父区间汇总，因此总量与分项是包含关系，不能把所有层级直接相加。
- FLOPs 是理论运算量，不是硬件计数器值；CUDA kernel 融合、稀疏执行、通信、访存和 CPU 操作不会等价反映在 FLOPs 中。

## Cosmos Policy 推理流程

```text
WebSocket observation
  → build_sample
      图像校验/resize，构造视频条件、动作状态和 prompt
  → ActionTransformPipeline
  → build_batch
  → prepare_data_total
      sequence_plan
      get_data_and_condition → vae_encode
      tokenize_text
      initial_pack
      初始化 vision/action noise
  → sampler_total（UniPC/EDM）
      每个采样步：
        conditional_forward
          pack_per_step
          denoise_total
            network_forward
              encode_text
              encode_vision
              encode_action
              build_attention
              mot_joint_forward
              vision_head
              action_head
        unconditional_forward（guidance != 1 时）
          同一条网络 forward 链
        CFG 合成 conditional/unconditional velocity
  → action_postprocess
      截取动作维度与历史帧、夹爪方向恢复、可选位姿反归一化
  → vae_decode（仅 --decode-video）
  → WebSocket response
```

`num_steps=4` 不必然等于 4 次网络 forward。启用 CFG 且 `guidance != 1` 时，通常每步包含 conditional 和 unconditional 两次 forward，实际次数应以 `diffusion_network_calls`、`conditional_forward_calls` 和 `unconditional_forward_calls` 为准。

## 统计模块表

| JSONL 名称 | 含义 | FLOPs 统计 | 为什么统计 |
|---|---|---:|---|
| `request_total` | 单次服务请求总耗时 | 自动汇总全部已估算子模块 | 用户感知端到端指标 |
| `build_sample` | resize、视频/动作样本构造、ActionTransformPipeline | 0 | 主要是 CPU、索引、内存和数据变换，浮点公式不能代表瓶颈 |
| `build_batch` | dataloader 风格 batch 包装和 seed | 0 | 数据结构与内存操作 |
| `generator_total` | 模型生成全阶段 | 自动汇总 | 模型侧请求总量 |
| `prepare_data_total` | 推理数据准备总阶段 | 汇总 VAE encode 等已估算子项 | 区分一次性准备与逐步采样 |
| `sequence_plan` | 构建多模态序列计划 | 0 | CPU 控制流 |
| `get_data_and_condition` | 生成 clean condition | 汇总 `vae_encode` | 条件视觉编码入口 |
| `vae_encode_total` / `vae_encode` | RGB 图像/视频编码为 latent | Wan 2.2 VAE 卷积、残差块和 attention 理论公式 | 一次性但计算量较大的视觉预处理 |
| `tokenize_text` | prompt 生成 token IDs | 0 | tokenizer 是 CPU 字符串/查表处理 |
| `initial_pack` | 初始 condition mask 和 packed sequence | 0 | 主要为索引、拼接与数据移动 |
| `sampler_total` | UniPC/EDM 完整采样 | 自动汇总每次网络 forward | 扩散推理主体 |
| `conditional_forward` | 条件分支 forward | 自动汇总 | CFG 条件计算 |
| `unconditional_forward` | 无条件分支 forward | 自动汇总 | CFG 额外计算成本 |
| `pack_per_step` | 每次 forward 重新打包 token | 0 | 主要为拼接、索引和拷贝 |
| `denoise_total` | 一次 denoise 调用 | 自动汇总 | 单次扩散网络调用 |
| `network_forward` | Cosmos3VFMNetwork forward | 自动汇总 | 完整 MoT 网络调用 |
| `encode_text` | token ID 到 embedding 并 scatter | 0 | embedding lookup 按常见 FLOPs 口径为 0，瓶颈主要是访存 |
| `encode_vision` | VAE latent patch 投影到 LLM hidden，并加 timestep embedding | 线性层和 timestep MLP | 模态输入映射 |
| `encode_action` | action token 投影到 LLM hidden，并加 timestep embedding | DomainAwareLinear 和 timestep MLP | 动作输入映射 |
| `build_attention` | attention metadata、位置与并行布局构造 | 0 | 控制流、索引和元数据构造 |
| `mot_joint_forward` | MoT Transformer 主干 | Q/K/V/O、attention、softmax、RMSNorm、dense/MoE MLP 和 final norm | 推理计算量最大核心 |
| `vision_head` | hidden 映射为 vision velocity | `llm2vae` 线性投影 | 输出视觉扩散速度 |
| `action_head` | hidden 映射为 action velocity | `llm2action` 线性投影 | Policy 最终动作预测头 |
| `action_postprocess` | 截取、CPU 转换、夹爪/位姿恢复 | 0 | NumPy/CPU 后处理，不纳入模型理论 FLOPs |
| `vae_decode` | 可选 latent 解码为 RGB | 暂记 0 | 当前仓库只有可靠 encoder 公式；避免给出错误 decoder 数值 |
| `video_to_numpy` | GPU 视频复制到 CPU NumPy | 0 | 数据传输，不是浮点计算 |
| `memory` | 峰值 allocated/reserved 显存 | 不适用 | 容量指标，不是计算量 |
| `counters` | conditional、unconditional、network、VAE 调用次数 | 不适用 | 解释总 FLOPs 为何随 sampler/CFG 变化 |

## 如何读取 JSONL

示例结构：

```json
{
  "cuda": {"mot_joint_forward": {"total_ms": 12.3}},
  "flops": {
    "mot_joint_forward": {"total": 123456789, "tflops": 0.000123456789},
    "sampler_total": {"total": 987654312, "tflops": 0.000987654312}
  },
  "counters": {
    "diffusion_network_calls": 8,
    "conditional_forward_calls": 4,
    "unconditional_forward_calls": 4
  },
  "memory": {"peak_allocated_mb": 12345.0}
}
```

比较模块时应使用同一请求的实际 token 数、采样步数、guidance 和 decode-video 配置。耗时高但 FLOPs 低通常说明访存、CPU、同步或通信占主导；FLOPs 高但耗时没有同比增长通常来自更高的 GPU 利用率或 kernel 融合。
