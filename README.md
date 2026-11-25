# GeoEdit — 文本引导图像编辑

一个版本一条分支：`v1`、`v2`、`v3`…… 每条分支**自带全部内容**，新版本需要什么就从上一个版本复制
（不要的就删）。本文件只做一件事：**查每个版本是什么、有什么文件、结论如何**。

## 版本目录

| 版本 | 分支 | 想法（一句话） | 结论 | 日期 |
|---|---|---|---|---|
| v1 | `v1` | 用部分（TV-unbalanced）最优传输估计"哪里该改"，拿它的边际当软 mask 混合源图与 proposal | ✗ **负结果**：背景偏差是高频纹理重绘（空间容忍无余量），且传输计划根本没被用到 | 2026-11-24 |
| v2 | `v2` | 一步编辑到底"改得够不够"、迭代（Picard）能不能把它推得更远 | ✗ **负结果**：λ 在 ≈1–1.5 饱和；Picard 被 λ 曲线支配；唯一正向发现是**多噪声平均**（+0.8 dB @ 匹配 CLIP），收益来自降方差而非不动点收敛 | 2026-11-24 |

## v2 有什么

想法一页纸：[`docs/ideas/v2-one-step.md`](docs/ideas/v2-one-step.md)｜交接文档：[`docs/handoff/v2_p1_p2.md`](docs/handoff/v2_p1_p2.md)

| 类别 | 文件 | 说明 |
|---|---|---|
| **模型** | `geoedit/model.py` | SD-Turbo 封装：`a/b` 表、`x0_pred`（ε/v 两种预测）、VAE 编解码、prompt embedding 缓存、NFE 计数 |
| **编辑算子** | `geoedit/edit_ops.py` | 残差 `R`、`λ`、**Picard** 迭代、`avg`（多噪声平均）、`prox`（Tweedie 投影） |
| **数据** | `geoedit/data.py`、`geoedit/pie.py` | PIE-Bench 700 张；`mask512` 与官方 `mask_decode` 逐像素一致；`mask64`/`dist64` 给光晕统计用 |
| **统计** | `geoedit/halo.py`、`geoedit/stats.py` | 光晕统计 `ρ`、mask 位移零假设；图级 bootstrap CI、Wilcoxon、**不外推**的 Pareto 插值 |
| **评测** | `geoedit/metrics.py` | 包官方 PnPInversion 的 `MetricsCalculator`（不改其源码），本地 CLIP 软链绕过 HF |
| **实验** | `scripts/10_p1_compute.py`、`11_p1_analyze.py` | P1：700×3t×4seed 的残差 `R`，光晕/各向异性/收缩率检验（`results/p1/latents/` 315 MB，不入库） |
| | `scripts/20_p2_generate.py`、`21_p2_evaluate.py`、`22_p2_analyze.py` | P2：38 个变体 × 2800 个 (t,seed) 束 = 106,400 次评测（3.3 GB latents，不入库） |
| **数据包** | `deepseek_talk/p2/` | 给分析用的纯数据（112 KB）：主表、判定 json、Pareto 匹配、汇总 CSV、报告、偏离记录 |
| **报告** | `report/REPORT_p1_p2.md`、`report/REPORT_interim_p1.md`、`report/DEVIATIONS.md` | 最终报告（判定表 §3、P2 主表 §4）/ P1 中间报告 / 9 条偏离 |
| **环境** | `env/requirements.txt`、`scripts/prepare_sd_turbo.py` | 精确版本；sd-turbo 目录整形（发布版是 Euler，loader 要 DDPM） |
| **配置** | `configs/base.yaml`、`p1_halo.yaml`、`p2_howfar.yaml` | 全部可调参数都在这里，含 `num_shards` / `gpus` |

**关键数字**（t=0.9, seed=0, raw）：CLIP-Edited 在 λ0.5/1/1.5/3 = 23.63 / 24.37 / 24.44 / 24.14，
背景 PSNR = 25.16 / 20.29 / 17.70 / 14.44；`recon` 上限 **PSNR 32.40**。
`avg`（6 次噪声平均，λ=1）PSNR **25.44** / CLIP 23.71 —— 同 PSNR 档位下 λ 需要 ~4 才到，但 CLIP 只有 ~22.5。
**CLIP 不可跨论文比较**（输入标度 0–255 vs 0–1 能差 9.9 分）。

## v1 有什么（现在只在本分支的历史里）

`git switch v1` 可看完整实现：`src/geoedit/uot.py`（部分 OT 求解器 + 20 项单测）、
`src/geoedit/chord.py`（ChordEdit 基线）、`scripts/run_e1.py`（背景偏差诊断）、`docs/DECISION.md`。
本分支只保留回顾用的 [`docs/ideas/v1-partial-ot.md`](docs/ideas/v1-partial-ot.md)。

## 新增 v3

```bash
git switch -c v3 v2        # 需要的文件留着，不要的删掉
# 写方法模块 + 新建 docs/ideas/v3-<短名>.md
git add -A && git commit -m "v3: <一句话>"
git push origin v3
```

然后在上面「版本目录」加一行。

## 跑起来

```bash
conda activate geoedit                       # torch 2.6.0+cu124 / diffusers 0.31.0 / fp32
cd /data/lx/Code/GeoEdit                     # 脚本用相对路径读 configs/ 与 results/
python scripts/00_sanity.py                  # S1–S5 冒烟（含 fp16/fp32 比对）
python scripts/10_p1_compute.py  --shard 0 --device cuda:0    # P1 残差
python scripts/20_p2_generate.py --shard 0 --device cuda:0    # P2 latent
python scripts/21_p2_evaluate.py --shard 0 --device cuda:0    # P2 评测
python pytest -q                             # 13 项单测（不需要权重）
```

## 四条踩过的坑（别重踩）

1. **PIE-Bench 的 `*_unedit` 对 mask 方法有约 `−10·log10(背景占比)` dB 的固定加成**（先把两张图乘 mask 再按全图尺寸算），拿它跟无 mask 方法比 PSNR 不公平。
2. **CLIP 分数不可跨论文比较**：同一张图同一权重，`0–255` 输入 27.39、`0–1` 输入 17.52。
3. **先问自己：方法能不能被一个 mask 消融复现**。能的话就等于一个区域估计器，要跟块匹配 / 逐点阈值 / GT-mask 上界比，不是跟无 mask 的原方法比。
4. **全量 fp32**。fp16 未通过 S3（残差相对差中位数 1.30% > 1%，见 `DEVIATIONS.md` D1）；VAE/UNet/文本编码器统一 fp32，别混。
