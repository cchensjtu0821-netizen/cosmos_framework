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

| 原算子 | 原图中的用途/数据 | 第一阶段处理 | 等价性和约束 | 改写后主要算子 |
| --- | --- | --- | --- | --- |
| `ConstantOfShape` | 按固定 shape 生成常量张量；shape 或 value 可能位于 external data | 解析固定 shape 和填充值，增加标量 initializer，以 `Expand` 生成目标张量 | 固定 shape 下严格等价；无法解析的节点不能假装已处理 | `Expand` |
| `Einsum` | exporter 生成的单输入维度置换，例如 patchify/unpatchify 周边 permutation | 仅对“输入/输出 label 一一对应、无求和”的 unary equation 计算 `perm`，替换为 `Transpose` | 只接受纯 permutation pattern，局部严格等价 | `Transpose` |
| `Gather` | 常量索引、标量索引、axis-0 vector 索引，以及 prompt embedding lookup | 常量结果折叠为 initializer；标量索引改为 `Slice + Squeeze`；axis-0 固定 vector 索引改成 `[N,1]` 索引的 `GatherND`；prompt token embedding 移到宿主侧 | 前三类为固定索引严格等价；宿主必须用导出的原 embedding table 生成 `prompt_embeddings` | initializer、`Slice`、`Squeeze`、`GatherND`；prompt 输入为 `[108,2048]` embedding |
| `ScatterElements` | `Unsqueeze -> Expand -> scatter_add` 形成的 axis-0 行累加；更新行数为 `N` | 去掉 expanded 全坐标索引，构造紧凑 `[N,1]` 行索引，改成 `ScatterND(reduction=add)` | 只处理 `axis=0`、`reduction=add` 且索引/updates shape 可证明匹配的 pattern | 中间形式 `ScatterND(reduction=add)` |
| `Where` | 一类是固定 layout 的 shape/选择逻辑；另一类是 causal attention mask：`Where(mask, -Inf, scores)` | 常量 `Where` 直接折叠；causal `Where` 将 mask 广播到静态 score shape，只 `GatherND` 未遮挡 score，再写入全 `-Inf` initializer | 保留分支隔离：被 mask 的 NaN/Inf 不参与读取或算术；早期 `Clip + bias` 方案已拒绝 | causal 路径为 `GatherND + ScatterND` overwrite |

### 3.1 第一阶段同时发生但不属于“五类缺失算子”的处理

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

### 3.2 第一阶段数值验证结论

服务器历史验证表明，在 causal `Where` 精确改写和 vision rank lowering 修正后，
输出满足 `finite=True`、`nonfinite=(0,0)`、`allclose=True`，并达到五类目标算子
为 0、节点和 graph I/O rank 均不超过 4。每次重新导出仍必须重新执行同样的
数值对比，不能把历史结果当成新模型的验证结果。

## 4. 第二阶段：OMG/OMC 实测问题触发的改写

第二阶段并不是继续处理最初五类算子，而是处理第一阶段产生的替代算子以及
OMG 自身的 shape inference/算子实现限制。

| 触发点 | OMG/OMC 现象 | 已确认数据 | 第二阶段处理 | 状态 |
| --- | --- | --- | --- | --- |
| 固定 `GatherND` | 为避免 OMG 对这些固定索引路径的兼容问题而继续 lower；base 中有 36 个 | base: `GatherND=36`；v2: `GatherND=0` | 连续索引直接改 `Slice`；一般固定 full-coordinate 索引先 flatten，再按连续段生成 `Slice`，用 bounded-fan-in `Concat` 合并，必要时 `Reshape` | 已完成结构改写；仍需每版 OMG/数值验证 |
| `ScatterND(reduction=add)` | OMG 不接受由第一阶段 `ScatterElements` lowering 产生的 add reduction | 共 2 个；静态 `[N,1]` 连续 axis-0 行索引，updates shape 匹配 | `Slice(data) -> Add(updates) -> Concat(prefix, updated, suffix)`；未满足条件的 reduction 必须令 finalization 失败 | v3 已消除这 2 个 |
| overwrite `ScatterND` | `node_ScatterND_239` 报 `indices.dim[0]=20`、`updates.dim[0]=19` 并终止 IR 生成 | ONNX 实际为 data `[64,3201,1]`、indices `[20,1]`、updates `[20,3201,1]`；indices=`[1,4,7,...,55,58]`。上游 Slice 是 `start=1,end=60,step=3,dim=64`，按 ONNX 语义应有 20 行 | 对静态、排序、唯一、边界内的 `[N,1]` 行索引，保留 data 未更新区间并插入 updates 的单行 `Slice`，最后用 fan-in<=64 的分层 `Concat`；显式 `reduction=none` 属性先移除 | 诊断为 OMG strided-Slice shape inference 偏差；v4 消除 124 个，仍留 28 个非适配 layout |
| `Mul` 广播 | overwrite ScatterND 问题绕过后，v4 的 OMG 在 `node_mul_23` 报 124 与 128 无法广播 | 四条已 lower 的 GatherND 路径均被 OMG constant-fold 为 feature width `124` 的 Slice；`node_mul_23` 另一输入要求 `128`；`scaleShape[0]=3093=3201-108` 是 sequence 后半段长度 | 仅当两输入 fully-static、同 rank、且恰有一个 axis 从 1 广播到目标值时，显式插入 `Expand -> Mul`。若实际是 124 对 128，不补零、不伪造数值，保持未处理并继续诊断 | `INVESTIGATING`；这是 v4 的终止错误。现有 v1-v4 均为 `Expand=118`，不能声称已包含或解决此改写 |
| external weight 参数 | OMG 命令未传分离的 ONNX external data | 权重默认 `${COSMOS3_ONNX_RAW}.data` | 脚本检查文件存在并传 `--weight` | `RESOLVED`；这是调用参数，不是算子改写 |

此外，最终化阶段还会处理 `Reshape.allowzero`、graph-output `Identity`、节点命名、
Gemm 名称映射和 rank 审计。这些是 OMG/DOPT 工具链规范化，不是上述算子
报错的数值语义替代。

## 5. 多版 OMG ONNX 文件和完整算子数据

所有已统计文件位于同一 `edge_policy_16actions_int8_dyn_s8_v2` 输出目录。
`compatible.omg_v1.onnx` 当时不存在，日志记录为 `FileNotFoundError`，因此不能
补写或推断其算子数。

| 文件 | 总节点 | 算子种类 | 相对上一可用版本的主要变化 | 对应处理 |
| --- | ---: | ---: | --- | --- |
| `compatible.omg.onnx` | 4,402 | 26 | 基线 | 第一阶段之后、第二阶段各类 ND 算子仍存在 |
| `compatible.omg_v1.onnx` | 不可用 | 不可用 | 文件缺失 | 无可验证数据 |
| `compatible.omg_v2.onnx` | 53,150 | 25 | 总计 `+48,748`；`GatherND -36`、`Slice +47,972`、`Concat +784`、`Reshape +28` | 36 个固定 full-coordinate `GatherND` 展开；增减严格守恒 |
| `compatible.omg_v3.onnx` | 53,156 | 25 | 总计 `+6`；`ScatterND -2`、`Slice +4`、`Add +2`、`Concat +2` | 2 个连续 axis-0 `ScatterND(add)` 改为 Slice/Add/Concat |
| `compatible.omg_v4.onnx` | 242,051 | 25 | 总计 `+188,895`；`ScatterND -124`、`Slice +185,938`、`Concat +3,081` | 124 个静态行 overwrite ScatterND 展开；仍剩 28 个；OMG 最终在 `node_mul_23` 的 124/128 广播推形失败 |

### 5.1 v4 的实际终止错误和因果边界

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

### 5.2 每版完整 op count

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
