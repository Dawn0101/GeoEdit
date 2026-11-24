# GeoEdit — 文本引导图像编辑

一个版本一条分支：`v1`、`v2`、`v3`…… 每条分支**自带全部内容**，新版本需要什么就从上一个版本复制
（不要的就删）。本文件只做一件事：**查每个版本是什么、有什么文件、结论如何**。

## 版本目录

| 版本 | 分支 | 想法（一句话） | 结论 | 日期 |
|---|---|---|---|---|
| v1 | `v1` | 用部分（TV-unbalanced）最优传输估计"哪里该改"，拿它的边际当软 mask 混合源图与 proposal | ✗ **负结果**：背景偏差是高频纹理重绘（空间容忍无余量），且传输计划根本没被用到 | 2026-11-24 |

## v1 有什么

想法一页纸：[`docs/ideas/v1-partial-ot.md`](docs/ideas/v1-partial-ot.md)｜完整推导：[`2026_11_24.md`](2026_11_24.md)

| 类别 | 文件 | 说明 |
|---|---|---|
| **方法** | `src/geoedit/uot.py` | 部分 OT 求解器（窗口 + 稠密参考），20 项单测 `tests/test_uot.py` |
| **基线** | `src/geoedit/chord.py` | 官方 ChordEdit 封装：transport / **批量** prox / VAE 编解码 |
| **评测** | `src/geoedit/metrics.py`、`src/geoedit/pie.py`、`scripts/eval_run.py` | PIE-Bench 官方口径（8 项指标），输出逐图 CSV |
| **缓存** | `scripts/build_cache.py`、`src/geoedit/cache.py`、`cache/` | 700 张 proposal（`x_src/y/y_prox/noise`，282 MB，**不入库**，重建 4.4 分钟/2 卡） |
| **诊断** | `scripts/run_e1.py` | 背景偏差：`r0` / 块匹配 `r*` / `gain` / 低频能量占比 / DINO 余弦 |
| **特征** | `src/geoedit/features.py` | latent 3×3 unfold + DINOv2 patch，统一 64×64 网格 |
| **结果** | `results/B0_chord_noprox/`、`results/B1_chord/`、`results/E1/` | 逐图指标 CSV + 23,704 行 patch 数据 |
| **结论文档** | `docs/DECISION.md`、`docs/baseline.md`、`docs/e1_report.md` | 决策记录 / 复现报告 / 诊断报告 |
| **环境** | `scripts/prepare_sd_turbo.py` | sd-turbo 目录整形（发布版是 Euler，loader 要 DDPM） |
| **杂项** | `scripts/{collect,smoke_test,bench_uot}.py`、`third_party/`、`archive/` | 汇总表 / 冒烟+官方比对 / 求解器性能 / ChordEdit(MIT) / 归档的旧计划 |

**关键数字**：PSNR(unedit) B0 **23.58** / B1 **22.45**（论文 23.89 / 22.20）；LPIPS 0.0878 / 0.121（论文 0.0884 / 0.1283）。
CLIP-Edited 不可跨论文比较（输入标度 0–255 vs 0–1 能差 9.9 分）。

## 新增 v2

```bash
git switch -c v2 v1        # 需要的文件留着，不要的删掉
# 写方法模块 + 新建 docs/ideas/v2-<短名>.md
git add -A && git commit -m "v2: <一句话>"
git push origin v2
```

然后在上面「版本目录」加一行。

## 三条踩过的坑（别重踩）

1. **PIE-Bench 的 `*_unedit` 对 mask 方法有约 `−10·log10(背景占比)` dB 的固定加成**（先把两张图乘 mask 再按全图尺寸算），拿它跟无 mask 方法比 PSNR 不公平。
2. **CLIP 分数不可跨论文比较**：同一张图同一权重，`0–255` 输入 27.39、`0–1` 输入 17.52。
3. **先问自己：方法能不能被一个 mask 消融复现**。能的话就等于一个区域估计器，要跟块匹配 / 逐点阈值 / GT-mask 上界比，不是跟无 mask 的原方法比。
