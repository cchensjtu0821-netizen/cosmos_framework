# RoboLab Policy ONNX 有效修改汇总（2026-08-11）

本文记录 2026-08-11 已进入 `main` 的有效修改、服务器验证证据、仍待验证项，
以及不能再复用的中间方案。目标模型是 `Cosmos3-Edge-Policy-DROID` 的固定布局
INT8 fake-quant ONNX 流程。

## 1. 当前有效提交

| Commit | 修改 | 当前状态 |
| --- | --- | --- |
| `946f8c0` | Step 5 ORT CUDA 验证前预加载 cuDNN | 已在服务器验证 |
| `5af88f8` | `COSMOS3_START_STEP5=1` 从 Step 5 继续执行到流程结束 | 已验证控制流 |
| `25fa336` | 默认恢复隐式 Mul 广播，显式 Expand 改为可选 | 已通过严格 rank audit |
| `982bc97` | 28 个 causal `Where` 改为有限值 additive bias | 结构生效；属于明确的端侧近似 |
| `06710c2` | Nemotron MRoPE 去掉 stride-3 原地赋值和 ScatterND | 源码改写有效，第一版 Stack 形式已被后续提交替代 |
| `0f821b5` | overwrite ScatterND 按连续 run 压缩 | 已在服务器验证，节点大幅下降 |
| `844ab1a` | MRoPE 改为 rank-3 interleave，避开 OMG rank-4 Concat | 已上库；必须重新执行 Step 4 后复验 |

## 2. Step 5 CUDA/cuDNN 验证恢复

### 问题

ONNX Runtime 能看到 `CUDAExecutionProvider`，但执行原始图的 Einsum 时无法
`dlopen libcudnn.so`。服务器实际已经安装兼容的 PyTorch、CUDA 13、cuDNN 9 和
`onnxruntime-gpu`；问题是 cuDNN 位于 Python 环境的 NVIDIA site-package 目录，
而新的 Step 5 Python 进程只从 CUDA toolkit 路径查找动态库。

### 当前实现

[rewrite_action_policy_onnx.py](../scripts/rewrite_action_policy_onnx.py) 在创建 CUDA
ORT session 前调用：

```python
onnxruntime.preload_dlls(directory="")
```

旧版 ORT 没有该接口时，回退为先导入 PyTorch。rewrite report 记录实际使用的
预加载方式。

### 验证证据

修复后服务器 Step 5 原始/改写图比较成功：

```text
vision_velocity: allclose=True finite=True nonfinite=(0,0) max_abs_error=0
action_velocity: allclose=True finite=True nonfinite=(0,0) max_abs_error=0
```

因此不需要依赖 `use_cuda_130` 把 toolkit 路径写入 `LD_LIBRARY_PATH` 来解决
Python wheel 中 cuDNN 的发现问题。

## 3. 从 Step 5 继续执行完整下游流程

[run_cosmos3_quant_onnx_full.sh](../onnxsrc/run_cosmos3_quant_onnx_full.sh) 当前支持：

```bash
COSMOS3_START_STEP5=1 bash onnxsrc/run_cosmos3_quant_onnx_full.sh
```

该模式：

1. 保留现有量化目录和 raw fake-quant ONNX；
2. 跳过 Step 1–4；
3. 执行 Step 5、6、7、8；
4. `COSMOS3_RUN_OMG=1` 时继续执行 Step 9；
5. 最后执行 Step 10 清理。

旧变量 `COSMOS3_ONLY_STEP5=1` 只作为兼容别名保留，语义也是“从 Step 5
继续到底”，不再表示只执行 Step 5。

重要限制：模型源码的 MRoPE 修改只会进入 Step 4 新导出的 raw ONNX。验证
MRoPE 时不能使用 `COSMOS3_START_STEP5=1`。

## 4. Mul 广播默认回退

显式 Mul broadcast pass 曾插入 480 个 Expand，导致：

```text
nodes=242531
unknown_rank=480
```

严格 Step 8 audit 因此在 OMG 前终止。当前默认保留 ONNX 隐式广播：

```bash
COSMOS3_MATERIALIZE_MUL_BROADCASTS=0
```

只有诊断 OMG broadcast 问题时才显式启用：

```bash
COSMOS3_MATERIALIZE_MUL_BROADCASTS=1
```

默认路径随后达到 `unknown_rank=0`。不能为了通过 audit 使用
`--allow-unknown-rank` 掩盖新增张量缺失 shape metadata 的问题。

## 5. 有限值 additive causal mask

### 当前改写

固定布局文本 causal attention 原来使用：

```text
Where(mask, -Inf, attention_scores)
```

当前 Step 5 将28个固定 causal pattern 改为共享静态 bias：

```text
Add(attention_scores, causal_bias)
visible bias = 0
masked bias  = -10000
```

`-10000` 在 FP16 中仍是有限值。`-100000000` 会溢出为 `-Inf`，因此不能作为
“FP16可表示的较小值”。

### 影响和边界

- 移除了28套 causal GatherND/ScatterND 兼容图。
- 最终节点从历史 `242051` 降为 `193219`，精确减少 `48832`。
- ORT 输出比较仍完整记录 `finite`、误差和 `allclose`。
- `allclose=False` 只打印 WARNING，不阻断 Step 6–9；checker、运行失败、shape、
  rank 和结构审计仍然阻断。
- 该方案不再保留 `Where` 对 masked NaN/Inf 的分支隔离，是为端侧精度限制接受的
  显式近似，不应描述为对任意输入严格等价。

## 6. overwrite ScatterND 连续 run 压缩

### 原问题

旧 Step 6 对每个静态 `[N,1]` overwrite ScatterND 都把 updates 切成 `N` 个
单行 Slice。即使 indices 是完整连续区间，也会制造海量 Slice/Concat。

### 当前实现

[finalize_onnx.py](../onnxsrc/finalize_onnx.py) 把严格递增、无重复的 row indices
划分为最大连续 run：

- 完全连续：直接使用完整 updates，只保留必要的0–2个 data prefix/suffix Slice；
- 覆盖 data 全部行：使用 `Identity(updates)`；
- 多个连续段：每个 run 一个 updates Slice，不再每行一个；
- 不满足静态 `[N,1]`、shape匹配、边界合法等条件的节点不强行套用该改写。

`reduction=add` 的连续区间仍使用已经验证的：

```text
Slice(data) -> Add(updates) -> Concat(prefix, updated, suffix)
```

新增 report 字段包括：

```text
lowered_contiguous_scatter_nd_overwrite_count
lowered_segmented_scatter_nd_overwrite_count
scatter_nd_overwrite_update_row_count
scatter_nd_overwrite_update_run_count
scatter_nd_overwrite_generated_update_slice_count
scatter_nd_overwrite_avoided_update_slice_count
```

### 服务器结果

连续 run 压缩前后的最终节点数：

```text
before: 193120
after:    4432
delta: -188688（约 -97.7%）
```

Step 7/8 结果：

```text
matched=340 missing_from_onnx=0 onnx_without_params=4092
nodes=4432 empty_names=0 duplicate_names=0 duplicate_producers=0
target_nodes=0 high_rank_nodes=0 high_rank_io=0 unknown_rank=0
```

其中 `4092 = 4432 - 340`，说明 Step 7 计数闭合；`onnx_without_params` 是所有
非量化节点数量，不是缺失量化参数的错误。

## 7. MRoPE 去 ScatterND 与 OMG Concat 规避

### 原始语义

活动 `mrope_section=[24,20,20]` 的64个 RoPE frequency-pair 槽排列为：

```text
[T,H,W] * 20 + T * 4
```

原始源码先使用全部 temporal freqs，再执行：

```text
height indices = [1,4,7,...,58]
width indices  = [2,5,8,...,59]
```

两个 stride-3 原地切片赋值被 ONNX functionalization 导出为 overwrite
ScatterND。

### 当前最终实现

[nemotron_3_dense_vl.py](../model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py)
不再执行索引写入。当前有效构图是：

```text
Slice temporal/height/width
-> rank-3 Concat([T-block,H-block,W-block])
-> Reshape([-1,3,20])
-> Transpose([0,2,1])
-> Reshape(...,60)
-> Concat(temporal tail 60:64)
```

`Concat -> Reshape -> Transpose -> Reshape` 严格恢复 `[T,H,W] * 20`，不会改变
预训练权重使用的位置编码通道映射。

### 已替代的中间方案

`06710c2` 第一版用：

```text
Stack(T,H,W) -> Flatten
```

它成功消除了 MRoPE ScatterND，但 ONNX 将 Stack 导为：

```text
Unsqueeze * 3 -> rank-4 Concat(axis=3)
```

OMG 在该 Concat 常量折叠附近报：

```text
malloc(): invalid size (unsorted)
Aborted (core dumped)
```

该错误表示 OMG 原生进程检测到堆结构已被破坏，不表示三个输入必须位于相邻内存，
也不等同于系统内存或显存不足。仅凭日志能确定 rank-4 Concat 是最接近的触发路径，
不能证明闭源 OMG 内部具体哪条写操作越界。

`844ab1a` 已用 rank-3 方案替代 Stack。最近一次复现仍使用旧 raw ONNX 从 Step 5
开始，因此不能用于判断 `844ab1a` 是否通过 OMG。该提交必须重新执行 Step 4。

## 8. 正确复跑方式

### 8.1 验证最新 MRoPE 源码和完整流程

```bash
git pull --ff-only origin main
git log -5 --oneline

COSMOS3_START_STEP5=0 \
COSMOS3_ONLY_STEP5=0 \
bash onnxsrc/run_cosmos3_quant_onnx_full.sh
```

这会重新执行 Step 1–4，确保 Step 4 raw ONNX 包含 rank-3 MRoPE。

### 8.2 仅复用已确认正确的 raw ONNX

只有当修改完全位于 Step 5/6 且 raw ONNX 无需更新时，才使用：

```bash
COSMOS3_START_STEP5=1 bash onnxsrc/run_cosmos3_quant_onnx_full.sh
```

### 8.3 本地连续 run 单测

```bash
PYTHONPATH=.. python3 -m unittest cosmos_framework.onnxsrc.finalize_onnx_test
```

## 9. 当前验证状态

| 项目 | 状态 |
| --- | --- |
| cuDNN preload 与 ORT CUDA session | 已验证 |
| Step 5 resume-through-completion 控制流 | 已验证 |
| 默认隐式 Mul broadcast、严格 rank audit | 已验证 |
| finite causal Add 图结构与节点缩减 | 已验证；数值近似为非阻断诊断 |
| overwrite ScatterND 连续 run 压缩 | 已验证，最终图降至4432节点 |
| MRoPE 去 ScatterND 的通道排列 | 本地映射证明通过 |
| rank-3 MRoPE 新 raw ONNX | 待服务器重新执行 Step 4 |
| rank-3 MRoPE 的 OMG/OMC | 待服务器验证 |
| Step 5 compatible 与 Step 6 final 数值比较 | 仍建议执行，用于隔离 finalizer 等价性 |

最终部署判据仍然是：ONNX checker 成功、Step 7 无缺失量化节点、Step 8
`target/high-rank/unknown-rank` 全部为0、兼容图与最终图数值满足预期，并且 OMG/OMC
完整完成。
