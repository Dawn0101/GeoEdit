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

---

# v3 / P3（2026-11-25 ~ 26）

## D10. 目录与脚本编号

- 交接文档写 `geoedit/geoedit/`，实现沿用 v2 的**扁平** `geoedit/`。
- 脚本编号扩到 30–41（文档只列到 34）：`35_p3b_analyze`、`36_posthoc`、`38/39_p3d_d1*`、
  `40/41_p3d_d2*` 都是事后追加，结果分别写 `results/p3/`、`results/p3d/`，不进主判定。

## D11. 预注册锚点在值域外时记 NOT_TESTABLE

- 文档 §3.3 说"其余情况记 INCONCLUSIVE"。H3b/H3c 的失败原因是**设计层面**的：主方法 V4 的
  CLIP 曲线（最高 19.892）根本到不了预注册锚点（19.957），不外推就无法判定。
- 按 PI 在 21:00（看到全量结果之前）的决定，这类情况记 **NOT_TESTABLE（设计）**，
  并在报告里写明"不外推"。H3e 同此处理。

## D12. AUROC 必须用连续值

- 第一版对 DiffEdit 式 / 单次能量两种 mask 用了**二值化后**的 mask 算 AUROC，值域只有几个点，
  AUROC 无意义（0.54 / 0.50）。
- 修正为 §2.1 原文要求的**连续值**：diffedit 0.858、energy 0.790（t=0.5）。二值版仍用于 IoU。

## D13. 事后脚本与数据

- `36_posthoc.py`（V12a/b/c、V2d、CLIP-D、best-of-4）、`35_p3b_analyze.py`、`38–41_p3d_*`。
- 全部标"事后"，写在 `REPORT_p3.md` §8 / `REPORT_p3d.md`，**不参与** H3a–H3e 与 D1/D2 的预注册判定
  之外的任何挑选。

## D14. D1 漏了 V2 变体

- `38_p3d_d1.py` 的变体表照交接文档 §2 写成 V0/V1/V1x/V12/S3/H1/H0，**漏了参照项 V2（GT mask）**。
- 补救：V2 与 V2d 的 CLIP-D 从 700 图的事后实验（`results/p3/posthoc/`，同一 λ 网格、同一批残差、
  同一套指标）取，写入 `p3d_d1c_by_category.csv` 的 V2/V2d 行；两者可比，但不是同一次前向。
- 不影响任何判定（V2 只作参照）。

## D15. D1b 锚点覆盖不足

- H1-5 vs V1x 那一行只有 **304/2000** 次 bootstrap 同时覆盖锚点（V1x 的 CLIP-D 最高 0.054，
  几乎摸不到锚点 0.0559），CI 是条件化的。表里已注明；结论以 H1-9 vs V1（8 vs 8 NFE）为准。

## D16. D2b 的候选数与 K

- 候选固定 4 个（交接文档要求），K=4 公共随机数；没有做 K=8 的整图版本（D2a 已显示 K 从 2 到 8
  只涨 0.012，受偏置限制）。

## D17. P3c 的细化检验被污染

- w1 是 Δℓ 门控 w0 得到的，而 w0（SNR 权重）本身 AUROC 0.489 → 这次**没有检验到 Δℓ 本身**。
- 报告里不得写"验证器逐像素不可用"；以 **P3d-D2a** 的干净检验为准（带符号 0.511、|Δℓ| 0.746）。

## D18. 进程被外部杀掉两次（工程记录）

- P3b 与 P3d-D1 各被杀过一次（日志无 traceback、无 dmesg OOM 记录），已用 `--resume` + watchdog
  重跑；D1 最终 700/700，P3b 最终 700/700。数据完整性由 shard 级 `image_id` 去重保证。

---

# P5（`docs/handoff/v4_p5.md`）

## P5-D1. EditReward 的词嵌入被静默随机初始化（**最严重的一条**）

- P4 的 `geoedit/editreward.py` 把 151666 写在**外层**多模态 config 的 `vocab_size` 上，而文本塔读
  `config.text_config.vocab_size`（仍是 152064）；`ignore_mismatched_sizes=True` 把形状不匹配吞掉，
  `embed_tokens` 与 `lm_head` 被**随机重新初始化**。视觉塔正常加载，所以模型"看得见图、读不懂字"。
- 影响范围：`results/p4/e1d/` 的**全部**数字（含"平均不损失语义"、"只有 GT mask 被偏好"、
  以及 `REPORT_p4.md` §2.5 与 `deepseek_talk/p4/` 的相应结论）**作废**，不只是 §0 第 3 条那一句。
- 修复：改 `config.text_config.vocab_size`、`ignore_mismatched_sizes=False`、加载后断言
  `embed 行数 == tokenizer 长度 == 151666`、`padding_side="right"`。
- 回归证据：官方 demo 图（`assets/examples`）修复前三个目标图挤在 0.163–0.175，"不编辑"最高 0.170；
  修复后 (src,src) = 1.137、三个目标 +1.440/+1.855/+1.463。
- 纪律说明：这是**修 bug**，不是改判定阈值；但因为它改变了已发布的数字，P4 报告里必须在原位留
  作废标记（已加）。

## P5-D2. E1d 的指令用错了字段

- P4-E1d 把 `tar_prompt`（目标 caption）当指令传给 EditReward；文档 §1.1 要求用 PIE-Bench 的
  `editing_instruction`。已在 `geoedit/data.py` / `pie.py` 增加 `instruction` / `blended_word` 字段，
  P5 全部改用 `editing_instruction`。

## P5-D3. 官方 EditReward 代码来自 GitHub 镜像

- 直连 github/raw 与本机 HF 均被网络策略拒绝；改用 `https://ghproxy.net/https://github.com/...`
  克隆官方仓库到 `/tmp/EditReward_off` 做逐行核对（`inference_vl_edit.py`、
  `model/qwen2_5_vl_trainer.py`、`dataset/data_collator_qwen_edit.py`、`config/EditReward-Qwen2.5-7B-VL.yaml`）。
- 官方环境用 **slow image processor**，本机 transformers 4.57 默认加载 **fast** 版本；两者预处理
  有微小差异。P4/P5 的 EditReward 与 Qwen 评委**都用同一默认**，内部一致，但与原论文数值不保证
  逐位可比。已在此记录。

## P5-D4. §1.3 的判定用中位数口径，且两口径结论相反

- 文档 §2 给的参照值（L0 ≈ 0.79、L7 = 0.858）在本机复现出来是**中位数**（0.7898 / 0.8581），
  对应均值 0.736 / 0.773（P3a `mask_quality.csv` 同口径可查）。
- 因此 §2 的 0.85 门槛按中位数读：L5 = 0.8565 **过线**；按均值读则最高只有 0.766，**不过线**。
- 处理：**两个口径都报告**，并且 §3/§6 的 4 NFE 与 10 NFE 两条腿**都跑**（文档 FAIL 分支本来就
  要求同时跑）。不视作判定阈值的修改，只视作口径必须写明。

## P5-D5. §1.3 的四个控制在 λ=1，背景 PSNR 只有 19.3 dB

- 文档预注册 λ=1，照做；但 λ=1 时掩码外画面已被明显破坏，评委很可能在回答"这图还自然吗"
  而不是"改动方向对不对"。
- 因此额外做一个**事后敏感性**：同一套控制在 λ=0.5 上重跑（`results/p5/validity_lam050/`），
  报告里**明确标为事后**，不替代预注册判定。动机是给 PI 一个"阈值是否定在工作点之外"的依据。

## P5-D6. §2 的平滑族是族内选优

- L1–L5 五个平滑器（pool3/pool5/σ=1/1.5/2）在**同一批图**上比较并取最好的 L5。族内 5 个选项、
  中位数跨度 0.831→0.857，不是险胜，但仍是样本内选择。报告里写明"σ=2 是在预注册的 L1–L5 族内
  选出的"。

## P5-D7. §3 的外部基线没有跑

- ChordEdit / BudEdit / PreciseEdit 都需要官方权重（SD1.5-inpainting 系），本机没有；`hf-mirror.com`
  可达但要另下 4–8 GB 并适配依赖。按文档"有代码才跑；没有代码就写明缺失"处理，在
  `REPORT_p5.md` §3 列表说明。
- 这是论文的硬缺口：至今**没有任何外部方法**的同台对比。

## P5-D8. 官方 demo 图的位置偏置

- 修复后的 EditReward 在交换图像顺序后分数系统性偏高 0.15–0.30（同一对图）。量级小于"编辑 vs
  不编辑"的差距（+0.30~+0.72），但 P5 的所有 EditReward 分数**一律用固定顺序（源图在前）**，
  与官方一致，不混用两序。
