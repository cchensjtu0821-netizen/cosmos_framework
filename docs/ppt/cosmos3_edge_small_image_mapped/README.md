# Edge Policy 192×320 映射版

本目录是原 `../cosmos3_edge_model_insights/cosmos3_edge_model_insights.pptx` 的两页更新版，保留原版。

## 数据口径

- 时延：原 PPT 各模块耗时 × 2026-10-27 选定小图请求 / 当日大图第 4 次请求的对应模块耗时。不是实测时延，也未平均多个请求。
- Generator 映射：312.75 × 336.0410461425781 / 759.0964965820312 = 138.4499041351248 ms。
- 理论计算量：采用小图配置的 20.582022123008 TFLOPs/请求，乘加计 2 FLOPs。
- TFLOP/s：理论计算量 / 映射耗时（秒），不是硬件峰值或测得的硬件利用率。
- 7.223 inf/s 是映射 Generator 耗时倒数，不是服务端实测吞吐。
- 显存为小图观测值：allocated 7909.24267578125 MiB，reserved 8048 MiB。
- 各区间嵌套，模块独立映射不保证严格加和；跨次测试运行状态差异未解释，不能视为同环境验证结果。
- PPT 只展示映射结果，不展示选定小图请求的实测对照。

## 更新范围

第一页更新 video、latent、vision patches、packed hidden 和 joint noise 形状；第二页更新时延、占比、理论计算量、折算速率、显存、日期、口径及讲稿。保留 2048 hidden、32 个未来动作、4 steps、CFG=3 和不解码视频的配置。

## 复现

使用 ppt-master 的 Edit Native PPTX 工作流，将原版导入一个新的 roundtrip workspace：

```bash
python3 "$PPT_MASTER_DIR/scripts/pptx_to_svg.py" docs/ppt/cosmos3_edge_model_insights/cosmos3_edge_model_insights.pptx -o /tmp/edge-small-mapped --inheritance-mode both --roundtrip
python3 docs/ppt/cosmos3_edge_small_image_mapped/apply_mapping.py /tmp/edge-small-mapped
python3 "$PPT_MASTER_DIR/scripts/svg_authoring_view.py" /tmp/edge-small-mapped/authoring-svg-flat --refresh-summary
python3 "$PPT_MASTER_DIR/scripts/svg_quality_checker.py" /tmp/edge-small-mapped --roundtrip --json
python3 "$PPT_MASTER_DIR/scripts/svg_to_pptx.py" /tmp/edge-small-mapped --roundtrip
```

验证：两页 SVG 容量检查通过，无错误或警告；导出后的 PPTX 再执行结构检查及文本回读。未在本地运行 CUDA 性能验证。
