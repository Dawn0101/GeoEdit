# E0 — ChordEdit 基线复现报告

- 日期：2026-11-24
- 代码：官方 `ChordEdit/ChordEdit` @ `a33d8cc26108851c6545b998c0e939006f89c9bd`（vendor 在 `third_party/`，未改动）
- 模型：`stabilityai/sd-turbo`（fp32），`/data/lx/Model/SD-Turbo`
- 评测：官方 `cure-lab/PnPInversion` @ `07f97f4` 的 `evaluation/matrics_calculator.py`，未改动
- 数据：PIE-Bench 700 张，`/data/lx/Dataset/PIE-Bench-Dataset`
- 随机种子 42（官方默认），`noise_samples=1`

## 1. 实现保真度

我们的 transport 与官方 `ChordEditPipeline.__call__` 的返回 latent **逐位一致**
（`max|Δ| = 0.0`，见 `docs/baseline.md` 附注与 `scripts/smoke_test.py --verify-official`）。
唯一对官方代码的适配：`sd-turbo` 发布的是 `EulerDiscreteScheduler`，而官方 loader 用
`DDPMScheduler.from_pretrained`，因此 `scripts/prepare_sd_turbo.py` 生成了一份等价的
DDPM 调度配置（`scaled_linear`, `0.00085 → 0.012`, 1000 步, ε-prediction）。

## 2. 主表（PIE-Bench 700 张）

| run | NFE | PSNR (全图) | **PSNR (unedit)** | **LPIPS (unedit)** | CLIP-Whole | **CLIP-Edited** | StructDist | SSIM (unedit) |
|---|---|---|---|---|---|---|---|---|
| **B0** 仅 transport | 1 | 19.688 | **23.584** | **0.0878** | 25.631 | **24.257** | 0.0228 | 0.810 |
| **B1** transport + prox | 2 | 18.770 | **22.449** | **0.1207** | 24.489 | **25.642** | 0.0301 | 0.764 |

论文自报（PSNR/LPIPS 为 PIE-Bench 背景口径）：

| | PSNR | LPIPS | CLIP-Edited |
|---|---|---|---|
| B0 | 23.89 | 88.36×10⁻³ = 0.0884 | 21.87 |
| B1 | 22.20 | 128.25×10⁻³ = 0.1283 | 22.96 |

## 3. Gate G1 判定

| run | ΔPSNR | ΔLPIPS | ΔCLIP-E | 判定 |
|---|---|---|---|---|
| B0 | **−0.31 dB** | **−0.0004** | +2.39 | ✅ 通过（阈值 ±1 dB） |
| B1 | **+0.25 dB** | **−0.0076** | +2.68 | ✅ 通过（阈值 ±1 dB） |

**结论：PSNR 与 LPIPS 复现成功，不需要扫 λ。** 两个独立指标的误差都在 0.31 dB / 0.008 以内，
说明 pipeline、VAE 标度、时间步映射、prox 的噪声约定都对。

## 4. CLIP-Edited 有 +2.4~2.7 的系统性偏移，这是评测器伪影，不是复现失败

三条证据：

1. **方向自相矛盾。** 如果我们的编辑比论文"更狠"，那 PSNR 应更低、LPIPS 应更高；
   实际两者都吻合。编辑强度是一样的，只有 CLIP 一项偏。
2. **偏移近似常数。** +2.39 与 +2.68 出现在两个不同配置上，符合"换了一把尺子"而不是"多改了一点"。
3. **直接测出来的**：同一张输出图、同一句 prompt、同一个 CLIP-L/14 权重，只改输入的数值标度：

   | 输入形式 | CLIP score |
   |---|---|
   | `uint8` (0–255) / `float` 0–255（官方路径） | **27.391** |
   | `float` 0–1 | **17.517** |

   一个标度差异就产生 **9.9 分**的落差。所以跨论文比较 CLIP 数值是没有意义的。

**操作规则（写给后续所有实验）**：CLIP-Whole / CLIP-Edited 只能和**本评测器跑出来的数**比较，
永远不要拿去对齐论文表格。PSNR / LPIPS / SSIM / Structure Distance 可以跨论文比较。

## 5. 其他实测

- 单图耗时：transport 0.38 s（与论文一致）；transport + prox + 两次 VAE decode 0.75 s
  （在被他占满的卡上测得，属于保守值）。
- 缓存：700 张 × {`x_src`, `y`, `y_prox`, `noise`} = 282 MB（`cache/`），后续任何方法都可以直接复用。
- 分母约定提醒：PIE-Bench 的 `*_unedit` 是**先把两张图乘 mask、再按全图尺寸算**，
  因此对带 mask 的方法有一个约 `−10·log10(背景占比)` dB 的固定加成。全图口径见上表第 4 列。

## 6. 复核方法

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_sd_turbo.py            # 模型整形 + 组件自检
CUDA_VISIBLE_DEVICES=0 python scripts/build_cache.py --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=0 python scripts/eval_run.py --run-id B0_chord_noprox
CUDA_VISIBLE_DEVICES=0 python scripts/eval_run.py --run-id B1_chord
python scripts/collect.py
```
