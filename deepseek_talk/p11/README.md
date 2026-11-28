# P11 数据包（六方同底座对比，5 NFE）

官方 ChordEdit 底座、同噪声 seed=42、严格 5 次前向、700 图全量。约 0.05 MB（纯表格）。

## 一句话

**SCAD-A 在 700 图上 CLIP-E 22.98 排第一，PreciseEdit(γ=1) 22.96 紧随其后**；SCAD 显著优于
ChordEdit（+0.25 偏好，胜率 61%）和 BudEdit（+0.44，胜率 65%），与 PreciseEdit 打平（+0.16，CI 含 0）。
**三家基线都是我们按论文公式的重实现**（作者均未放出代码）。

| 方法 | PSNR↑ | LPIPS↓ | DINO↓ | CLIP-E↑ |
|---|---|---|---|---|
| **Ours: SCAD-A (α=1)** | 22.60 | 0.092 | 0.028 | **22.98** |
| PreciseEdit (γ=1) | 22.99 | 0.117 | 0.032 | 22.96 |
| Ours: SCAD-Adap | 22.97 | 0.087 | 0.026 | 22.88 |
| Ours: SCAD-CleanBG | 22.99 | **0.083** | **0.025** | 22.85 |
| Official ChordEdit | 22.45 | 0.121 | 0.030 | 22.58 |
| ER-Chord (μ=1) | 22.47 | 0.119 | 0.029 | 22.46 |
| BudEdit | **25.99** | 0.062 | 0.020 | 22.42 |
| ER-Chord (μ=2) | 22.51 | 0.118 | 0.029 | 22.35 |
| PreciseEdit (带 γ) | 25.51 | 0.086 | 0.021 | 21.86 |

## 核心发现

1. **SCAD 的增益集中在语义重写类**：材质 +1.53、物体风格 +1.03、删除物体 +0.79、换背景 +0.70。
2. **SCAD 与 PreciseEdit 互补**：PreciseEdit 在"添加景物"上 +1.33 而 SCAD 为 0.00；SCAD 在"材质/风格"
   上大幅领先。二者融合是一个自然的 P12 方向。
3. **保真高 ≠ 被偏好**：BudEdit PSNR 25.99（最高）但在偏好上与 ChordEdit 打平（0.00）。
4. **PreciseEdit 的 γ 门控自伤**：γ=1 时 22.96，开启后 21.86（我们的图上 γ≈0.5）。

## 文件

| # | 文件 | 内容 |
|---|---|---|
| 1 | **`REPORT_p11.md`** | 六方主表、10 类别细分、A/B、实现要点、限制 |
| 2 | `p11_700_summary.csv` | 700 图 × 9 变体四项指标 |
| 3 | `p11_700_by_category.csv` | 10 类别 × 7 方法的 CLIP-E 细分 |
| 4 | `p11_ab_table.csv` | 5 对 VLM 配对（n=200） |
| 5 | `p11_ab_scad_vs_official.csv` | SCAD vs 官方逐图配对分 |
| 6 | `DEVIATIONS_p11.md` | P11-D1…D4（重实现声明、NFE 修正、显存泄漏、Adap 偏移） |

定性图：`http://127.0.0.1:3081/index.html`（9 列：source / ChordEdit / ER-Chord / BudEdit /
PreciseEdit / SCAD-A / SCAD-CleanBG / …）。
