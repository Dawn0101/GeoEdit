# P2 数据包(“推多远”检验)

给 Claude 做分析用的**纯数据**材料,无图片。全部 108 KB。
数据来自 700 张 PIE-Bench × t∈{0.9(2 seeds), 0.7, 0.5} × 38 个变体 = **106,400 次评测**。

## 先读哪个

| # | 文件 | 内容 | 大小 |
|---|---|---|---|
| 1 | **`p2_main_table_t0.9_seed0.csv`** | 主表:38 个配置在 t=0.9/seed=0 下的 7 个指标均值 + 标准误,含 NFE 与 n | 4.5 KB |
| 2 | **`hypotheses.json`** | H2a–H2d(+H1c)的判定:点估计、95% CI、p 值、n、`out_of_range` 标记 | 3.4 KB |
| 3 | `p2_sweep_all_t.csv` | 同上但按 t 分开(0.9/0.7/0.5,raw 与 prox),给 H2d 用 | 7.2 KB |
| 4 | `matched_clip.csv` | Pareto 匹配:ΔPSNR / ΔLPIPS / ΔStructDist + CI + `out_of_range` | 4.2 KB |
| 5 | `summary.csv` | 全部 152 个配置 × 7 指标的均值与 CI(完整版,1~4 都是它的摘录) | 64 KB |
| 6 | `REPORT_p1_p2.md` | 最终报告(§9.3 的 8 节),判定表在第 3 节、P2 主表在第 4 节、异常在第 6 节 | 7.4 KB |
| 7 | `DEVIATIONS.md` | 9 条偏离设计(D1 fp32、D3 null 用 disk、D8 prox 标记修复、D9 指标精简等) | 4.3 KB |

列名约定:`psnr` / `lpips` / `mse` / `ssim` 都是**非编辑区(1−mask)**上的值,与 PIE-Bench 官方口径一致;
`clip_whole` 是输出对源 prompt,`clip_edited` 是对目标 prompt;`struct_dist` 是 DINO ViT-B/8 结构距离;
`nfe` 是 UNet 前向次数(不含 VAE 解码)。

## 一句话结论

**P2 不支持“一步编辑系统性编辑不足、Picard 迭代能修好”。**
CLIP-Edited 在 λ≈1–1.5 饱和(K=1→K=3 内部位移翻倍但 CLIP-E −0.07,PSNR 20.3→16.1);
Picard 在每个 K 上都被 λ 曲线支配。唯一正向发现:`avg`(多噪声平均)在匹配 CLIP 下比 λ 曲线高 ~0.8 dB
——收益来自降方差,不是不动点收敛。

## 三条口径提醒

1. **全量 fp32**(不是 fp16)。fp16 未通过 S3:与 fp32 的残差相对差中位数 1.30% > 1%。
2. **`recon` 是 VAE 往返上限**:背景 PSNR = **32.40**(raw)。任何编辑方法的 PSNR 都要跟它比才算"保住了背景"。
3. **prox 是独立因子**:`prox=0` 直接解码,`prox=1` 先做一次 t_c=0.3 的目标分支 Tweedie 投影(多加 1 NFE)。
   主分析用 raw(prox=0),prox 作副分析。

## 故意没放的东西

| 缺的 | 大小 | 何时需要 |
|---|---|---|
| `per_image.csv`(逐图 106,400 行) | 22 MB | 想自己重算任何统计量时;可以按要求切 50 张图的子集 |
| `diag.csv`(Picard 每步 ‖δ_n‖、c_n) | 2.7 MB | 想看单张图的收敛轨迹;汇总已在 `hypotheses.json` 的 H2c |
| Pareto / 收敛 / 定性网格图 | — | 按要求未放图片 |
