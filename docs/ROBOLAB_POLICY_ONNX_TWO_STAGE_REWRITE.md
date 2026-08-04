# RoboLab Policy ONNX 两阶段算子改写与 OMG 版本记录

## 1. 文档目的

本文把 Edge Policy ONNX 的适配过程按触发原因分成两个阶段：

1. **目标后端已知不支持算子阶段**：按最初给出的不支持列表处理
   `ConstantOfShape`、`Einsum`、`ScatterElements`、`Gather`、`Where`。
2. **OMG/OMC 实际转换问题阶段**：第一阶段输出进入量化和 OMG/OMC
   流程后，根据真实转换报错继续处理 `GatherND`、两类 `ScatterND` 和
   `Mul` 广播等问题，因此产生 `compatible.omg.onnx`、`omg_v2`、
   `omg_v3`、`omg_v4` 等版本。

这两个阶段不能混为一谈。第一阶段解决“已知算子集合不被部署后端接受”；
第二阶段解决“替代算子或合法 ONNX 图仍触发 OMG 的实现限制/shape inference
问题”。本文中的节点数来自 2026-08-04 在服务器上对相应文件执行
`onnx.load_model(..., load_external_data=False)` 后的统计。没有保存到日志的
早期原图节点数不作推测。

## 2. 流程边界

```text
原始固定 layout denoiser ONNX
  -> 第一阶段：rewrite_action_policy_onnx.py
  -> compatibility ONNX（已知五类目标算子为 0，rank <= 4）
  -> 量化/图最终化
  -> 第二阶段：finalize_onnx.py
  -> compatible.omg[_vN].onnx
  -> OMG 生成 IR
  -> OMC 生成部署模型
```

导出的 ONNX 只包含一次固定 packed-layout 的 VFM denoiser forward，不包含
tokenizer、VAE encoder、UniPC sampler、CFG、latent 更新和 action 后处理。

## 3. 第一阶段：最初声明不支持的五类算子

第一阶段的审计入口是 `inspect_action_policy_onnx.py`，默认目标集合就是以下
五类算子。实现入口是 `rewrite_action_policy_onnx.py`。

### 3.1 `ConstantOfShape`

1. **为什么处理**：最初明确给出的不支持算子之一，兼容图中不能残留。
2. **原始作用**：输入是 shape tensor，输出用固定标量填满该 shape。shape 或
   `value` attribute 引用的数据可能存放在 ONNX external data 中。
3. **怎么处理**：解析静态 shape 和填充值；把填充值保存成 scalar initializer；
   使用该 scalar 和原 shape 输入构造 `Expand`。
4. **算子变化**：`ConstantOfShape -> scalar initializer + Expand`。
5. **处理条件**：shape 和填充值必须能静态解析；无法解析就不能改写，也不能把
   节点直接删除。
6. **语义**：在固定 shape 下为局部严格等价。

### 3.2 `Einsum`

1. **为什么处理**：最初明确给出的不支持算子之一。
2. **原始作用**：exporter 在 patchify/unpatchify 周边生成单输入 unary `Einsum`，
   实际只做维度置换，没有乘加或 reduction。
3. **怎么处理**：解析 equation；仅当输入 label 和输出 label 一一对应、无重复、
   无消失 label 时，根据 label 顺序计算 `perm`。
4. **算子变化**：`Einsum(equation) -> Transpose(perm)`。
5. **处理条件**：只改写纯 permutation equation；一般多输入或含求和的 `Einsum`
   不套用该规则。
6. **语义**：满足上述条件时局部严格等价。

### 3.3 `Gather`

`Gather` 不是用同一种方式全部替换，而是按数据来源和索引布局分四类处理。

#### 3.3.1 常量数据、常量索引

1. **原 pattern**：`Gather(initializer, constant_indices)`。
2. **怎么处理**：离线计算 Gather 结果，把结果直接保存为 initializer，删除节点。
3. **算子变化**：`Gather -> initializer`。

#### 3.3.2 单个标量索引

1. **原 pattern**：indices 只有一个元素，axis 和 index 可静态解析。
2. **怎么处理**：先用 `[index:index+1]` 做 `Slice`，再在相同 axis 上 `Squeeze`。
3. **算子变化**：`Gather -> Slice + Squeeze`。

#### 3.3.3 axis-0 固定向量索引

1. **原 pattern**：`Gather(axis=0)`，indices 是固定一维向量，共 `N` 个索引。
2. **怎么处理**：把 indices reshape 成紧凑的 `[N,1]` int64 坐标。
3. **算子变化**：`Gather(axis=0) -> GatherND(indices=[N,1])`。
4. **后续影响**：这个 `GatherND` 是第一阶段的中间替代算子；OMG 仍有兼容问题，
   所以在第二阶段 base→v2 又被继续改成 `Slice/Concat/Reshape`。

#### 3.3.4 prompt embedding lookup

1. **原 pattern**：用 `prompt_token_ids` 从模型 embedding table 做 `Gather`。
2. **怎么处理**：导出原 embedding table 为 sidecar；图输入从 token IDs 改成宿主
   已完成 lookup 的 `prompt_embeddings`。
3. **算子/接口变化**：删除图内 embedding `Gather`；宿主提供固定布局
   `[108,2048]` 的 embedding。
4. **处理条件**：宿主必须使用导出的同一张 embedding table，否则不等价。

上述四类处理的共同原因都是 `Gather` 属于最初不支持列表。前三类在固定索引下
严格等价；第四类是接口等价。

### 3.4 `ScatterElements`

1. **为什么处理**：最初明确给出的不支持算子之一。
2. **原始 pattern**：`Unsqueeze -> Expand -> ScatterElements`，其中
   `axis=0`、`reduction=add`；expanded indices 描述对 `N` 行的累加更新。
3. **怎么处理**：不再保留与 updates 同形状的 expanded indices；解析每个更新
   对应的 axis-0 行号，生成紧凑 `[N,1]` int64 row indices。
4. **算子变化**：

   ```text
   ScatterElements(axis=0, reduction=add, expanded_indices)
     -> ScatterND(reduction=add, row_indices=[N,1])
   ```

5. **处理条件**：必须能证明原节点是 axis-0 add、每个 expanded index 与紧凑
   row index 表达同一批更新，并且 data/indices/updates shape 匹配。
6. **后续影响**：`ScatterND(reduction=add)` 仍不被 OMG 接受，所以第二阶段
   v2→v3 再将其中 2 个改成 `Slice + Add + Concat`。

### 3.5 `Where`

`Where` 分为固定常量选择和 causal mask 两类。

#### 3.5.1 固定 layout 的常量 `Where`

1. **原 pattern**：condition、true branch、false branch 都可在固定配置下求值。
2. **怎么处理**：离线计算完整输出并保存为 initializer。
3. **算子变化**：`Where -> initializer`。

#### 3.5.2 causal attention `Where`

1. **原 pattern**：`Where(mask, -Inf, attention_scores)`。
2. **不能使用的方案**：早期 `Clip(scores) + causal_bias` 会读取并计算被 mask 的
   NaN/Inf，破坏 `Where` 的分支隔离语义，已拒绝。
3. **最终处理**：把 causal mask 广播到静态 score shape；仅生成未遮挡位置的完整
   坐标；用 `GatherND` 只读取未遮挡 scores；以全 `-Inf` initializer 为底板，用
   overwrite `ScatterND` 写回未遮挡值。
4. **算子变化**：

   ```text
   Where(mask, -Inf, scores)
     -> GatherND(unmasked scores)
     -> ScatterND(overwrite into all--Inf initializer)
   ```

5. **语义**：masked source value 从未被读取或用于算术，因此 NaN/Inf 不会污染
   输出，保持原 `Where` 的分支隔离语义。
6. **后续影响**：产生的固定 `GatherND` 在第二阶段 base→v2 被继续 lower；部分
   overwrite `ScatterND` 在 v3→v4 被继续 lower。

### 3.6 第一阶段辅助处理：不属于五类缺失算子

这些处理是固定部署接口、rank 限制或验证要求引入的，不能误归因为最初的
五类算子列表：

- 固定 DROID `action_domain_id=8`。
- 把 `prompt_token_ids` 改为宿主生成的 `prompt_embeddings`，避免在图内保留
  embedding `Gather`。
- 将 batch=1 的 rank-5 vision I/O 降为 rank 4。
- rank-6 patchify 改成 rank-4 `Transpose -> SpaceToDepth -> Transpose -> Reshape`；
  unpatchify 使用 `Reshape -> Transpose -> DepthToSpace(mode=DCR) -> Transpose`
  逆变换。通道次序必须为 `[p,q,C]`。
- 所有 `GatherND`/`ScatterND` indices 规范为 `int64`，并去重节点名。

### 3.7 第一阶段数值验证结论

服务器历史验证表明，在 causal `Where` 精确改写和 vision rank lowering 修正后，
输出满足 `finite=True`、`nonfinite=(0,0)`、`allclose=True`，并达到五类目标算子
为 0、节点和 graph I/O rank 均不超过 4。每次重新导出仍必须重新执行同样的
数值对比，不能把历史结果当成新模型的验证结果。

## 4. 第二阶段：OMG/OMC 实测问题触发的改写

第二阶段并不是继续处理最初五类算子，而是处理第一阶段产生的替代算子以及
OMG 自身的 shape inference/算子实现限制。本节只按文件版本变化逐段说明。

### 4.1 base：第二阶段的起点

文件：`compatible.omg.onnx`。

1. **它是什么**：第一阶段 compatibility graph 经过量化/最终化后的 OMG 输入基线。
2. **仍包含的关键中间算子**：`GatherND=36`、`ScatterND=154`。
3. **为什么还需要继续改**：这些算子虽然不属于最初五类不支持算子，但在实际
   OMG 转换中继续暴露兼容性和 shape inference 问题。
4. **规模**：4,402 个节点、26 种 op type。

### 4.2 base -> v2：改写 36 个固定 `GatherND`

文件变化：`compatible.omg.onnx -> compatible.omg_v2.onnx`。

#### 4.2.1 为什么改变

base 仍有 36 个固定索引 `GatherND`。为避免 OMG 处理这些 full-coordinate
GatherND 路径时的兼容问题，第二阶段将它们全部 lower 为基础 shape 算子。

#### 4.2.2 怎么改变

对每个 `GatherND(data, indices)`：

1. 静态解析 `data_shape` 和 `indices`。
2. 若 indices 是连续的 `[N,1]` axis-0 row indices，直接生成一个 `Slice`。
3. 若 indices 是完整坐标 `[N,rank(data)]`：
   - 用 `ravel_multi_index` 转成 row-major flat indices；
   - 先 `Reshape(data, [-1])`；
   - 把相邻 flat index 合并成连续区间；
   - 每个区间生成一个 `Slice`；
   - 用 fan-in 不超过 64 的分层 `Concat` 拼回 GatherND 输出；
   - 这里新增的 `Reshape` 用于先把原 data 展平成一维，随后按 flat index 切片。
4. 无法静态解析、越界或顺序不满足安全条件的节点必须报告 unresolved，不得
   猜测改写。

#### 4.2.3 改了哪些算子和数量

```text
GatherND  -36     36 -> 0
Slice  +47,972   290 -> 48,262
Concat    +784   143 -> 927
Reshape    +28   340 -> 368
总节点 +48,748  4,402 -> 53,150
```

数量严格守恒：`47,972 + 784 + 28 - 36 = 48,748`。

#### 4.2.4 结果

v2 已无 `GatherND`，但仍有 `ScatterND=154`。节点名可能继续保留
`node_GatherND_*`，只是其 `op_type` 已变为 `Slice` 或替代子图；OMG 日志打印
旧节点名不代表模型仍含 GatherND op。

### 4.3 v2 -> v3：改写 2 个 `ScatterND(reduction=add)`

文件变化：`compatible.omg_v2.onnx -> compatible.omg_v3.onnx`。

#### 4.3.1 为什么改变

第一阶段为了消除 `ScatterElements(axis=0,reduction=add)`，生成了中间形式
`ScatterND(reduction=add)`。OMG 仍不接受带 add reduction 的 ScatterND，因此
必须继续 lower。

#### 4.3.2 适用数据条件

只处理同时满足以下条件的节点：

1. indices 是静态 `[N,1]`；
2. indices 表示连续 axis-0 行；
3. updates 的第 0 维是 `N`，其余维与 data 对应区域完全相同；
4. 更新范围在 data axis-0 边界内。

#### 4.3.3 怎么改变

```text
ScatterND(data, rows=[start:start+N], updates, reduction=add)
  -> prefix  = Slice(data, [0:start])
  -> current = Slice(data, [start:start+N])
  -> updated = Add(current, updates)
  -> suffix  = Slice(data, [start+N:end])
  -> Concat(prefix, updated, suffix, axis=0)
```

空 prefix/suffix 按边界情况省略。不能证明上述条件的 reduction ScatterND 会让
finalization 失败，不允许静默改变语义。

#### 4.3.4 改了哪些算子和数量

```text
ScatterND  -2   154 -> 152
Slice      +4   48,262 -> 48,266
Add        +2   453 -> 455
Concat     +2   927 -> 929
总节点     +6   53,150 -> 53,156
```

数量严格守恒：`4 + 2 + 2 - 2 = 6`。这表明 2 个 add 型 ScatterND 各自生成
2 个 Slice、1 个 Add、1 个 Concat。

#### 4.3.5 结果

v3 消除了全部 2 个 `ScatterND(reduction=add)`，但仍有 152 个 ScatterND；其中
overwrite ScatterND 随后触发 OMG shape inference 问题。

### 4.4 v3 -> v4：改写 124 个 overwrite `ScatterND`

文件变化：`compatible.omg_v3.onnx -> compatible.omg_v4.onnx`。

#### 4.4.1 为什么改变

OMG 在 `node_ScatterND_239` 终止 IR 生成，报：

```text
indices.dim[0] = 20
updates.dim[0] = 19
Failed to generator IR graph
```

但精确 ONNX 数据是：

```text
data    [64,3201,1]
indices [20,1] = [1,4,7,...,55,58]
updates [20,3201,1]
```

updates 来自 `Slice(start=1,end=60,step=3,dim=64)`；按 ONNX exclusive-end/ceil
语义结果应为 20 行。OMG 却在下游 ScatterND verifier 中推成 19 行。因此这里
不是 ONNX ScatterND 自身 shape 非法，而是 OMG 对 strided Slice 的 shape inference
与 ONNX 语义不一致，错误最终暴露在 ScatterND。

#### 4.4.2 适用数据条件

只处理同时满足以下条件的 default-overwrite ScatterND：

1. reduction 缺省或为 `none`；显式 `reduction=none` attribute 先移除；
2. indices 是静态 `[N,1]` axis-0 row indices；
3. row indices 已排序、互不重复、全部在 data 边界内；
4. updates shape 与 `N` 行 data slice 完全匹配。

#### 4.4.3 怎么改变

1. 按 row indices 把原 data 切成未更新区间。
2. 从 updates 中为每个目标行生成对应的单行 `Slice`。
3. 按原 axis-0 行顺序交替排列“原 data 区间”和“update 行”。
4. 使用 fan-in 不超过 64 的分层 `Concat(axis=0)` 重建完整 tensor。
5. 不能满足条件的 ScatterND 保留并写入 report，不强行改写。

#### 4.4.4 改了哪些算子和数量

```text
ScatterND    -124   152 -> 28
Slice    +185,938   48,266 -> 234,204
Concat     +3,081   929 -> 4,010
总节点   +188,895   53,156 -> 242,051
```

数量严格守恒：`185,938 + 3,081 - 124 = 188,895`。巨量 Slice/Concat 是逐行精确
重建 overwrite 语义的直接代价，不是重复统计。

#### 4.4.5 结果和 v4 新报错

- v4 消除了 124 个满足条件的 overwrite ScatterND，还剩 28 个不满足该 row-wise
  rewrite 条件的 ScatterND。
- `node_ScatterND_239` 不再是终止点。
- v4 继续执行后，在 `node_mul_23` 出现新的 124/128 feature 维广播错误，OMG
  仍未成功生成 IR。

### 4.5 v4 当前终止错误：`node_mul_23`

v4 的 OMG 日志按执行顺序说明：

1. 多个由 overwrite ScatterND 生成的 `ConcatD` 已完成 constant folding。日志中的
   `inputConcatAxis ...` 和一次 `0 dim concatted` 是 kernel 过程信息；对应节点随后
   出现 `[const_folding_success]`，因此它们不是本次 OMG 退出点。
2. 名为 `node_GatherND_288`、`295`、`306`、`313` 的节点也出现
   `[const_folding_success]`，实际 IR op 已是 `StridedSliceV2`。v4 的 ONNX op count
   为 `GatherND=0` 并不矛盾：finalizer 改了 `op_type`，但保留原节点名，OMG 日志
   打印的是这个历史名称。
3. 四条 Slice 分别覆盖 sequence 的 `0:108` 或 `108:3201`，同时都读取 feature
   维 `0:124`，OMG 看到的该维实际大小也是 124。随后的 `ExpandDims`、`Cos/Sin`
   和 `Squeeze` constant folding 均成功。
4. 真正的 fatal chain 是
   `DynamicQuantMaxInfer(scaleShape[0]=3093)` ->
   `node_mul_23: dim[2] 124 should be 128` ->
   `MathBroadCastInfer failed` -> `Failed to generator IR graph`。

因此，**v4 失败的直接原因**已经确定：OMG IR shape inference 时，
`node_mul_23` 的两个输入在 feature 维分别为 124 和 128，无法按 ONNX 广播规则
相乘。`3093` 对应 `3201-108`，解释的是 sequence 维，不是 124/128 feature
差异的成因。

**更上游的根因尚未由这段日志确定。** 当前证据只证明 124 宽张量来自 RoPE
相关的 Slice/Cos/Sin 路径；还不能判断：

- v4 ONNX 中该 Slice 的源张量本来就是 124 宽；
- 量化参数或前序 shape metadata 把应为 128 的维度记录成 124；
- 或 OMG constant folding/IR shape inference 把 ONNX 中的 128 错推成 124。

要区分这三种情况，必须在服务器直接读取 v4 ONNX，打印 `node_mul_23` 两个输入
及其 producer chain，并打印四个历史 `node_GatherND_*` 节点当前的 `op_type`、
Slice inputs、源 tensor shape、starts/ends/axes/steps。若 ONNX 本身已是 124 对
128，图在进入 OMG 前就存在真实 shape 不一致；若 ONNX 是 128 而 OMG 日志变成
124，才可以归因于 OMG 推形或 constant folding。

## 5. base/v2/v3/v4 完整算子数量对比

所有已统计文件位于同一 `edge_policy_16actions_int8_dyn_s8_v2` 输出目录。
`compatible.omg_v1.onnx` 当时不存在并报 `FileNotFoundError`，所以本文不推断 v1
数据。下表保留全部已出现的 op type，而不是只列发生变化的算子。

| Op | base | v2 | v3 | v4 |
| --- | ---: | ---: | ---: | ---: |
| `Slice` | 290 | 48,262 | 48,266 | 234,204 |
| `Concat` | 143 | 927 | 929 | 4,010 |
| `Mul` | 738 | 738 | 738 | 738 |
| `Add` | 453 | 453 | 455 | 455 |
| `Gemm` | 342 | 342 | 342 | 342 |
| `Reshape` | 340 | 368 | 368 | 368 |
| `Pow` | 254 | 254 | 254 | 254 |
| `Transpose` | 241 | 241 | 241 | 241 |
| `ReduceMean` | 198 | 198 | 198 | 198 |
| `Sqrt` | 198 | 198 | 198 | 198 |
| `Reciprocal` | 198 | 198 | 198 | 198 |
| `Cast` | 190 | 190 | 190 | 190 |
| `ScatterND` | 154 | 154 | 152 | 28 |
| `Neg` | 140 | 140 | 140 | 140 |
| `Unsqueeze` | 123 | 123 | 123 | 123 |
| `Expand` | 118 | 118 | 118 | 118 |
| `MatMul` | 115 | 115 | 115 | 115 |
| `Softmax` | 56 | 56 | 56 | 56 |
| `Relu` | 56 | 56 | 56 | 56 |
| `GatherND` | 36 | 0 | 0 | 0 |
| `Squeeze` | 9 | 9 | 9 | 9 |
| `Cos` | 3 | 3 | 3 | 3 |
| `Sin` | 3 | 3 | 3 | 3 |
| `Sigmoid` | 2 | 2 | 2 | 2 |
| `SpaceToDepth` | 1 | 1 | 1 | 1 |
| `DepthToSpace` | 1 | 1 | 1 | 1 |

v4 的 242,051 个节点主要是精确展开 overwrite ScatterND 的代价。节点更多、
ScatterND 更少不等于更适合部署；必须以 OMG 是否成功、转换时间/内存，以及
原图与改写图的数值一致性共同判断。

## 6. 两阶段归属速查

| 项目 | 阶段 | 原因 |
| --- | --- | --- |
| `ConstantOfShape -> Expand` | 第一阶段 | 最初声明算子不支持 |
| unary `Einsum -> Transpose` | 第一阶段 | 最初声明算子不支持 |
| `Gather` 折叠/改写/embedding 外置 | 第一阶段 | 最初声明算子不支持 |
| `ScatterElements -> ScatterND(add)` | 第一阶段 | 最初声明算子不支持 |
| constant/causal `Where` 改写 | 第一阶段 | 最初声明算子不支持 |
| vision rank-5/rank-6 lowering | 第一阶段辅助 | rank<=4 部署约束，不属于五类算子 |
| `GatherND -> Slice/Concat` | 第二阶段 | OMG 兼容化后续处理 |
| `ScatterND(add) -> Slice/Add/Concat` | 第二阶段 | OMG 不接受 reduction=add |
| overwrite `ScatterND -> Slice/Concat` | 第二阶段 | OMG strided-Slice/ScatterND shape inference 报错 |
| static `Mul` broadcast 显式 `Expand` | 第二阶段 | OMG 在 RoPE 区域报 124/128 broadcast 问题 |
| external data `--weight` | 第二阶段辅助 | OMG 调用参数缺失，不是算子问题 |

## 7. 当前结论和未完成验证

- 第一阶段五类目标算子改写已经有历史数值等价证据，但任何新导出都需要重新验证。
- 第二阶段的 v2/v3/v4 节点变化均能由具体 pass 精确解释；v1 没有文件，不能补数据。
- v4 仍有 28 个 `ScatterND`，它们不是静态 `[N,1]` 行 overwrite pattern，不能
  为追求计数为 0 而强行套用现有 rewrite。
- v4 已确认因 `node_mul_23` 的 124/128 feature 维无法广播而失败；124 是来自
  RoPE Slice 源数据、量化/shape metadata，还是 OMG 推形错误，仍需检查精确 v4
  ONNX 才能定论。只有 singleton-axis 广播可以安全显式化；真实宽度不一致不能
  通过 padding 或 blanket `Expand` 修补。
- 本地工作区没有用户服务器上的 ONNX/CUDA/OMG/OMC 运行环境，因此本文整理的是
  已记录的服务器证据和代码语义，不声称本地完成了 OMG/OMC 或数值验证。

每个候选模型仍应在服务器执行：

```bash
python -c "import onnx; onnx.checker.check_model('<candidate.onnx>')"

python -m cosmos_framework.scripts.inspect_action_policy_onnx \
  <candidate.onnx> \
  --target-op GatherND \
  --target-op ScatterND \
  --max-rank 4
```

随后使用相同输入比较原始和候选图的 `vision_velocity`、`action_velocity`：结果
必须 finite，非有限值计数必须为 0，并在约定 `atol/rtol` 下 `allclose=True`；
最后再以 OMG/OMC 实际退出码和输出日志确认转换成功。

## 8. 代码和记录入口

- 第一阶段审计：`scripts/inspect_action_policy_onnx.py`
- 第一阶段改写：`scripts/rewrite_action_policy_onnx.py`
- 第二阶段最终化：`onnxsrc/finalize_onnx.py`
- 量化/OMG 流程：`onnxsrc/run_cosmos3_quant_onnx_full.sh`
- 历史诊断：`docs/agent_logs/2026-08-03.md`、
  `docs/agent_logs/2026-08-04.md`
