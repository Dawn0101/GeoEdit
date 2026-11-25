# 偏离设计之处

每条都附原因;§8.1 要求"设定照原样报告",所以凡改动的都在这里。

## D1. 全部改用 fp32(§6 S3 未通过)

- 原设定:`UNet 和 VAE 默认 fp16`(§2)。
- 实测:20 张图、t=0.9,`||R_fp16 − R_fp32|| / ||R_fp32||` 的中位数 = **1.30%**(最大 2.31%),
  阈值为 1%。→ 按文档"S3 不满足就全部改用 fp32"。
- 处理:`configs/base.yaml` 的 `unet_precision` / `vae_precision` 全部设为 `fp32`,
  所有阶段(P1、P2、评测)统一 fp32。S3 的判定逻辑改为"基线为 fp32 时该项视为通过",数值照报。
- 影响:显存 ~6.5GB/进程、速度约为 fp16 的 0.6 倍;所有耗时统计都基于 fp32。

## D2. 背景为空时 `*_unedit` 指标返回 NaN

- 官方 `evaluate.py` 在 `(1-mask).sum()==0` 时返回字符串 `"nan"`;
  `matrics_calculator.py` 本身没有这个保护,直接调用会得到 `+inf`(两张图被 mask 乘成全零,
  完全一致),把均值污染成 `inf`。
- 处理:`geoedit/metrics.py` 在调用官方函数之前判断背景是否为空,为空则返回 `float("nan")`,
  与官方 `calculate_metric` 的行为一致。类别 8/9 里 mask 覆盖全图的样本会命中这条。

## D3. null 平移的"外扩 6px"用欧氏 disk 实现

- 文档只说"不与原 mask 外扩 6px 后的区域重叠",没说结构元。
- 处理:用**欧氏距离 ≤ 6** 的 disk(而不是方形或菱形),这是"6px 外扩"最直接的读法,
  也让大物体的角落位置可用。合法平移**精确枚举**(不是拒绝采样),
  每张图最多取 20 个(random state 固定为 0,可复现)。
- 后果:mask 较大的图可能**没有任何合法平移**,此时 `rho_null = NaN`(文档允许)。
  实测 700 张里有 96 张 null 全为 NaN(interior 太大)。

## D4. P1 的"主检验带" (2,6] 需要单独计算

- `configs/p1_halo.yaml` 的 `bands` 是画图用的 5 条带 `(0,2] (2,4] (4,6] (6,10] (10,∞)`,
  不含主检验带 `(2,6]`。
- 处理:`11_p1_analyze.py` 在逐带计算之外,**额外**算一次主带(σ=3,以及 σ=1.5 的副设置),
  否则 H1 会拿到空集。输出里主带的 `exclude_reason` 与其它带同规则。

## D5. H1b 的判定只有"SUPPORTED / 未达成"

- 文档只定义了 H1b 的 SUPPORTED 条件(相邻两档配对差的 CI 不含 0 且方向为"t 越小越大")。
- 处理:方向相反时我的代码记 `INCONCLUSIVE`,但在报告里**明确指出方向相反、CI 不含 0**,
  不把它当成"证据不足"来含糊过去。

## D6. 本机到 GitHub 不通,第三方评测代码用镜像抓取

- `cure-lab/PnPInversion` 的两个文件(`evaluation/{evaluate.py,matrics_calculator.py}`)
  经 `ghproxy.net` 取回,内容与 pin 的 commit `07f97f4` 一致(17583 / 16358 字节)。
- 代码未做任何修改;我们只是 import 它的 `MetricsCalculator`。

## D7. S4 健全性门槛用 50 张**等间隔抽样**的图

- 文档写"PIE 全集",但为了在阶段 A 快速判定,用 700 张里等间隔抽 50 张(覆盖全部 10 类),
  而不是前 50 张(那是同一类,会把门槛测偏)。全量结果见 P2 的 `per_image.csv`。

## D8. prox 标记的解析 bug 与修复

- 生成脚本把变体命名成 `lam|1|prox`(把 `|raw|prox` 改写成 `|prox`),而评测脚本最初按
  "第 4 段才是 prox"解析,于是 `lam|1|prox` 被记成 `prox=0`,与 `lam|1|raw` 撞键,
  合并去重时把 53,200 行 prox 结果覆盖掉了。
- 修复:`parse_variant` 现在两种形式都认(第 3 段是 `prox` 或第 4 段是 `prox`)。
- 已有 CSV 的修复方式:每个 (image, t, seed, method, param) 组恰好两行,文件顺序为先 `prox` 后 `raw`
  (生成脚本按 `sorted(bundle.keys())` 写入,`...|prox` 排在 `...|raw` 之前)。
  三项独立验证:(a) `recon` 第一行 PSNR 25.47 < 第二行 32.40(后者是 VAE 重建上限);
  (b) 修复后 `prox=1` 的 `lam|1` PSNR 22.69 < `prox=0` 的 24.31,方向与 v1 的 B0/B1 一致;
  (c) Picard K=1 与 λ=1 逐元素相同(最大差 0.0)。
- 以后再跑不需要这个修复(解析函数已改)。

## D9. 评测只计算 §4.2 要求的指标(lean 模式)

- 原实现沿用 v1 的 `PieMetrics.evaluate`,会同时算全图 PSNR/LPIPS/MSE/SSIM 和 edit-part CLIP。
- §4.2 只要求:Structure Distance、非编辑区 PSNR/LPIPS/MSE/SSIM、CLIP-Whole、CLIP-Edited。
- 处理:新增 `lean=True`,跳过全图与 edit-part 指标。单图 59.4s → 45.4s。
  被跳过的量不是交付物,也没有进任何判定。
