# P3 数据包（v3 主实验 P3a 中间报告）

给分析用的**纯数据**，无图片、无逐图大文件。全部 37 KB。
数据来源：700 张 PIE-Bench × 11 个变体 × 6 个 λ × 3 个 t + recon = **114,100 次评测**
（SD-Turbo，fp32，raw 编辑，官方 PIE-Bench 指标）。主分析排除类别 8/9 → **540 张**。

## 先读哪个

| # | 文件 | 内容 | 大小 |
|---|---|---|---|
| 1 | **`REPORT_interim_p3a.md`** | 中间报告：判定表、mask 质量、前沿、根因、偏离 | 9 KB |
| 2 | `p3a_hypotheses_t0.9.json` | H3a–H3d 的判定 + 事后分析（匹配 PSNR 前沿、公共 CLIP 锚点、whole-CLIP 参照） | 22 KB |
| 3 | `p3a_main_table.csv` | t=0.9 每个（变体 × λ）的均值与标准误（+ recon） | 5 KB |
| 4 | `p3a_mask_quality_medians.csv` | 每种 mask 的连续 AUROC / IoU / 面积（按 t） | 0.5 KB |
| 5 | `p3a_variance_medians.csv` | latent 空间方差分解 V / S / frac_random（按 t 与 mask 内外） | 0.2 KB |

## 一句话结论

**预先注册的 SNR 权重 `w = max(0, 1 − c/T)`（`T = N·M̄/4S̄`）在真实残差上不是定位器**：
AUROC 0.49，因为编辑区内"平均能量"和"方差"同比放大（t=0.5: 8.19 vs 8.27），比值是常数。
同一个残差，DiffEdit 式的 N 次平均能量 AUROC **0.86**。所以 H3b/H3c 在设计上无法判定
（锚点在 V4 的 CLIP 范围之外），H3a 无差异，H3d 显著但效应量 0.008（无信息量）。

## 三条口径提醒

1. 主语义指标是 **`clip_target_edit`**（官方 `clip_similarity_target_image_edit_part`）。P2 的
   `clip_edited` 对应 `clip_target_whole`，P2 的 `clip_whole` 是"输出 vs 源 prompt"，**不是**官方
   `clip_similarity_source_image`（那个是"源图 vs 源 prompt"，对每张图是常数）。数值没错，名字有歧义。
2. **同 λ 比较 ≠ 前沿比较**。同 λ 下 GT mask（V2）比 V0 高 10 dB 背景 PSNR、只掉 0.17 CLIP；
   但 V0 用小 λ 就能换到同样的 PSNR，匹配 PSNR 后两者曲线重合（25 dB 处 19.70 vs 19.82）。
3. **`psnr` / `lpips` / `mse` / `ssim` 都是非编辑区（1−mask）**；`recon` 是 VAE 往返上界
   （背景 PSNR 32.5）。fp32 全量（v2 D1）。

## 故意没放的东西

| 缺的 | 大小 | 何时需要 |
|---|---|---|
| `per_image.csv`（114,100 行） | 25 MB | 想自己重算任何统计量；可按要求切子集 |
| `mask_quality.csv` / `variance_decomp.csv` 逐图版 | 0.6 / 0.25 MB | 想看单张图；汇总已在上面 |
| 对齐图 `checks/align_*.png`、Pareto 图 | — | 按要求不放图片 |
| P3b / P3c / 事后实验 | — | 还在跑，出来另发 |
