# P6 数据包（Ours-4 的 Mask 内 SDS ＋ 同源 A/B 闸门 ＋ ChordEdit 同台）

给 Claude 的纯数据 + 报告（约 0.5 MB）。逐图 PNG 在 `cache/images/p6/`、`results/p6/vis_k/`，不进包。

## 三句话

1. **SDS 项是对的**：`A = R + e_src` 不是 `R` 的重标定（‖e_src‖/‖R‖ = 1.77，cos = 0.28），
   而且它的掩码内**可重复性高 20 倍**（随机占比 0.964 → 0.542）。
2. **k=1 有效**：匹配背景 PSNR 下 `k=1 − k=0` = **+0.50 nat [+0.38,+0.75]**，胜率 **71%**；
   k=2 与 k=1 打平（只更破坏）。
3. **仍然打不过 ChordEdit**：同底座/同 200 图/同评测器，`Ours-4(k=1) − ChordEdit` =
   **−0.91 [−1.31,−0.40]**，胜率 22%。

## 先读哪个

| # | 文件 | 内容 |
|---|---|---|
| 1 | **`REPORT_p6.md`** | 主报告：Stage 0 诊断、A/B 闸门（含失败形式）、50 组视觉、§4 主表、NFE 双轨记账、与 PreciseEdit 的关系、下一步 |
| 2 | `p6_ab_table.csv` | **主表**：4 个对比 × 全部/`\|ΔPSNR\|<1` 子集，配对 bootstrap |
| 3 | `p6_ab_validity.json` | C1–C5 效度（λ=1 全过、λ=0.5 边缘） |
| 4 | `p6_sds_diagnose.{json,csv}` | e_src/A vs R、每图方差分解、AUROC |
| 5 | `p6_ab_pairs_matched.csv` | 200 图逐图的 A/B 分与两侧背景 PSNR |
| 6 | `p6_vis_k.csv` | 50 组的 k/γ 四项指标 |
| 7 | `p6_chord_200.csv` | ChordEdit 在同一 200 图上的官方指标（PSNR 22.02 / LPIPS 0.125 / DINO 0.016 / CLIP-E 22.59） |
| 8 | `p6_matched_lam.csv` | λ→背景 PSNR 的反解（含每个 run 在 22/24/26 dB 的 λ） |
| 9 | `fig_p6_ab.png` | 森林图 + k 权衡散点 |
| 10 | `DEVIATIONS_p6.md` | P6-D1…D6 |

## 两个必须一起看的方法学点

- **闸门形式**：强制二选一 `logP(A)−logP(B)` **失败**（换序相关 −0.38，按位置作答）；改成
  "同一源图的 Yes/No 配对差"后 λ=1 五项全过。判据改动已记 `DEVIATIONS` P6-D3。
- **NFE 双轨**：ChordEdit 实测**每图 5 次 UNet 前向**（4 个 batch 元素 + 1 次 prox），但论文口径
  写 2。我们 4。写"4 vs 2"会误导，写"4 vs 5"才对得上实测。
