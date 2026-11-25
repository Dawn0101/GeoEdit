# 交接文档 v3：P3 —— 用方差决定"在哪改"

- 日期：2026-11-25
- 发给：服务器上的执行 agent（DeepSeek）
- 前置：v2（`claude_to_deepseek_geoedit_20261124_v2_p1_p2.md`）已完成。本文档在 v2 的代码库 `geoedit/` 上**增量**开发，v2 的固定设置（§2）、纪律（§8）全部沿用，除非本文明确修改。
- 想法背景（不必细读）：`2026-11-25_0337_v3_push_pull_variance.md`

---

## 0. 目标

P2 的数据表明：单次噪声下，背景损伤约 90% 是随机的，而编辑区里的随机部分带着语义。P3 检验下面这个做法是否能把 CLIP–PSNR 前沿往外推：

> **用 N 次共享噪声的逐像素统计量决定"在哪改"（权重 w），用其中一次噪声的残差决定"改成什么"。**

P3 分三部分：

| 部分 | 内容 | 需要 UNet 吗 | 优先级 |
|---|---|---|---|
| **P3a** | 用 P1 已存的残差（每张图 4 seed × 3 t）离线构造各种 mask 和编辑，解码评测 | **不需要**，只解码 | 最高，先交中间报告 |
| P3b | 噪声侧的技巧：源自洽噪声（不动点）、对偶噪声 | 需要，量小 | 中 |
| P3c | 用模型自己的似然比图把粗 mask 收细（100 张图试点） | 需要，量小 | 低，探索性 |

---

## 1. 先修评测口径（阶段 A 必做，不通过不能继续）

P2 的 `clip_whole` 随编辑强度单调下降（recon 最高 26.1），说明它是对**源** prompt 算的；D9 又写了跳过 edit-part 指标。P3 必须输出与 PIE 官方 `evaluate.py` **同名、同定义**的三个 CLIP 量：

| 列名 | 定义（以官方代码为准） |
|---|---|
| `clip_target_whole` | 编辑后整图 vs 目标 prompt（官方 "CLIP Similarity Whole"） |
| `clip_target_edit` | 编辑后图像的编辑区（官方用 mask 处理的那部分）vs 目标 prompt（官方 "CLIP Similarity Edited"） |
| `clip_source_whole` | 编辑后整图 vs 源 prompt（非官方，仅作参考） |

- **主指标**：`clip_target_edit`（语义）和非编辑区 `psnr`（背景）。
- 在报告里写一段：P2 的 `clip_edited` 实际对应上表哪一列，附代码行号。
- 新测试 **T-X3**：在 20 张图上，三个 CLIP 量与直接调用官方 `evaluate.py` 的结果逐项一致（最大绝对差 < 1e-4）。

---

## 2. 记号与固定设置

- 全部沿用 v2 §2：SD-Turbo、无 CFG、**fp32**（v2 的 D1）、ε 由 `(image_id, seed)` 确定性生成并在源、目标分支之间共享、PIE 官方 mask 与评测代码。
- 第 i 次噪声（seed i）在噪声水平 t 下的残差：$R_i=\hat x_{\rm tar}(a x_s+b\varepsilon_i)-\hat x_{\rm src}(a x_s+b\varepsilon_i)$，形状 4×64×64。**P1 已经为每张图存了 t∈{0.9,0.7,0.5}、seed∈{0,1,2,3} 的 R**，P3a 直接读。若文件已删除，按 v2 的 `10_p1_compute.py` 重算（约 4 分钟）。
- 编辑：$x=x_s+\lambda\,\delta$，**直接解码（raw），不做 prox**。λ 网格 {0.5, 0.75, 1, 1.5, 2, 3}。
- latent GT mask：v2 的 `mask64`（8×8 平均池化 > 0.5）。**GT mask 只能用于 oracle 对照和评测，不能进入任何非 oracle 方法。**
- **任何方法组件都不能用 CLIP。**

### 2.1 逐像素统计量（`masks.py` 实现）

给定 N 个残差 $R_1..R_N$（N ≥ 2）：
1. 均值 $\hat m=\frac1N\sum_i R_i$；逐像素方差 $s^2_p=\frac{1}{4(N-1)}\sum_i\sum_{c=1}^{4}(R_{i,c,p}-\hat m_{c,p})^2$。
2. 用 3×3 盒滤波（`avg_pool`，stride 1，padding 1，边界按 `count_include_pad=False`）分别平滑 $\|\hat m_p\|^2$ 和 $s^2_p$，得到 $\overline{M}_p$、$\overline{S}_p$。
3. 信噪比 $T_p=\dfrac{N\,\overline{M}_p}{4\,\overline{S}_p+10^{-12}}$。没有信号时 $T\approx1$。
4. 权重（噪声校准的正部收缩）：$w_p=\max\big(0,\,1-c/T_p\big)$，**主设置 c=1**；c∈{0.5, 2} 作为敏感性分析，只报告，不用来挑结果。

另外几种 mask（都输出 [0,1] 的 64×64 图）：
- **DiffEdit 式**：$D_p=\frac1N\sum_i\|R_{i,p}\|$；除以本图最大值；> 0.5 记 1，否则 0。
- **单次能量**：$E_p=\|R_{0,p}\|$；除以本图最大值；> 0.5 记 1（类似 BudEdit 的能量支撑，不含注意力）。
- **跨 t 由粗到细**：$w^{\rm ct}=w(t{=}0.9)\odot w(t{=}0.5)$，两者都用 N=4、c=1。

---

## 3. P3a：离线实验（主实验）

### 3.1 变体（δ 的定义）

除 V2、V3 外，其他变体都不接触 GT mask。

| ID | δ | 含义 |
|---|---|---|
| V0 | $R_0$ | 基线（= P2 的 `lam`，用来在新口径下重画 λ 曲线） |
| V1 | $\hat m_4$ | 全局平均，N=4 |
| V2 | $M_{\rm gt}\odot R_0$ | oracle 位置 + 单次样本 |
| V3 | $M_{\rm gt}\odot \hat m_4$ | oracle 位置 + 平均 |
| **V4** | $w_4\odot R_0$ | **主方法**：SNR 权重（N=4）+ 单次样本 |
| V5 | $w_4\odot \hat m_4$ | SNR 权重 + 平均 |
| V6 | $w_2\odot R_0$ | N=2（seed 0、1） |
| V7 | $D_4\odot R_0$ | DiffEdit 式 mask |
| V8 | $E\odot R_0$ | 单次能量 mask |
| V10 | $w^{\rm ct}\odot R_{0}$（R 取 t=0.9） | 跨 t 由粗到细，只在 t=0.9 做 |
| V11 | $w_4\odot R_{i^\star}$，$i^\star=\arg\min_i\|(1-w_4)\odot R_i\|^2$ | 选样本（模型内部标准，不用 CLIP） |

网格：
- t=0.9：V0–V8、V10、V11（共 11 个）× 6 个 λ；
- t=0.5：V0–V8、V11（共 10 个）× 6 个 λ；
- t=0.7：V0、V2、V4、V7 × 6 个 λ；
- 另加 `recon`（λ=0）一次。

每张图约 150 次解码，合计约 10.5 万次。

### 3.2 描述性分析（不做判定，但必须输出）

1. **方差分解（latent 空间）**：每张图、每个 t，在 GT mask 内和 mask 外分别计算
   - 随机能量 $V=\mathrm{mean}_p(4s^2_p)$；
   - 系统能量的无偏估计 $S=\mathrm{mean}_p\big(\|\hat m_p\|^2-4s^2_p/N\big)$；
   - 随机占比 $V/(S+V)$。

   输出中位数和四分位数。
2. **mask 质量**：每种 mask 相对 `mask64` 的 AUROC（用连续值）和 IoU（连续值 > 0.5 二值化），按 t、按类别。
3. **λ 曲线图**：每个 t 一张 CLIP-Edited–PSNR 图，画出全部变体。

### 3.3 预注册假设与判定（看到数据之前固定，不允许修改）

**匹配方式**：全部在**均值层面**比较。对每个变体，用 6 个 λ 点的（平均 CLIP-Edited，平均 PSNR）做分段线性曲线，在**锚点 CLIP 值**处插值出 PSNR。
- 锚点 = 同一 t 下 V0 在 λ=0.75 和 λ=1 时的平均 CLIP-Edited。
- CI：以图为单位 bootstrap 1000 次。每次重采样图集合，重新计算全部均值曲线、锚点和插值，得到差值的分布。
- 锚点落在某条曲线 CLIP 范围之外时记 `out_of_range`，不外推。

| 假设 | 内容 | SUPPORTED | REFUTED |
|---|---|---|---|
| **H3a** | 编辑区里单次样本比平均更有语义（t=0.9） | $\max_\lambda$ CLIP-Edited(V2) − $\max_\lambda$ CLIP-Edited(V3) 的 CI > 0 | CI < 0 |
| **H3b（主）** | SNR 权重把前沿往外推（t=0.9） | 两个锚点上 ΔPSNR(V4 − V0) 的 CI 都 > 0 | 两个锚点上 CI 都 < 0 |
| **H3c** | 比 DiffEdit 式和单次能量 mask 好（t=0.9） | 两个锚点上 ΔPSNR(V4 − V7) 与 ΔPSNR(V4 − V8) 的 CI 都 > 0 | 任一的 CI < 0 |
| **H3d** | 随机能量在 mask 外的占比高于 mask 内（t=0.9） | 配对 Wilcoxon 单侧 p < 0.01 | 反方向 p < 0.01 |

其余情况记 INCONCLUSIVE。另外必须报告（无判定）：
- **缺口闭合率** $G=\frac{\mathrm{PSNR}(V4)-\mathrm{PSNR}(V0)}{\mathrm{PSNR}(V2)-\mathrm{PSNR}(V0)}$（在锚点处，附 CI）；
- V5 vs V4、V6 vs V4、V10 vs V4、V11 vs V4 的 ΔPSNR；
- t=0.5、0.7 上同样的比较；
- c 的敏感性。

主分析排除类别 8、9（背景和风格编辑，没有有意义的非编辑区），但它们的结果也要输出。

---

## 4. P3b：噪声侧（需要 UNet，t∈{0.9, 0.7}，700 张）

### 4.1 源自洽噪声（不动点）

- 初值 $\varepsilon^{(0)}$ = seed 0 的噪声。迭代 $\varepsilon^{(j+1)}=\hat\varepsilon_{\rm src}(a x_s+b\varepsilon^{(j)})$，j 最多到 8。
- 每步记录：
  - 源重建误差 $e_j=\|\hat x_{\rm src}(\varepsilon^{(j)})-x_s\|/\|x_s\|$；
  - 步长 $\|\varepsilon^{(j+1)}-\varepsilon^{(j)}\|/\|\varepsilon^{(j)}\|$；
  - $\varepsilon^{(j)}$ 的标准差和超额峰度（判断是否还像高斯）。
- 在 J∈{0, 2, 8} 时计算 $R^{(J)}$ 和 $A^{(J)}=\hat x_{\rm tar}(\varepsilon^{(J)})-x_s$，各做 λ∈{0.5, 1, 1.5, 2} 的编辑并评测。J=0 就是 V0。
- 检查恒等式：$\|A^{(J)}-R^{(J)}-e_{\rm src}\|$ 应在数值误差内为 0（测试 T-N3）。
- **H3e**（方向预注册，判定规则同 H3b）：t=0.9 时，在 V0 的两个锚点上，ΔPSNR($R^{(8)}$ − V0) 的 CI > 0。
- 描述：$e_j$ 随 j 的中位数曲线；是否收敛（定义：$e_8<0.5\,e_0$）的比例。

### 4.2 对偶噪声

- 计算 $R(-\varepsilon_0)$（t=0.9，2 NFE）。
- 用 $\{R(\varepsilon_0),R(-\varepsilon_0)\}$ 按 §2.1 算 $w_2^{\rm anti}$，与 seed 0、1 得到的 $w_2$ 比较 AUROC（描述性）。
- 做编辑 $w_2^{\rm anti}\odot R_0$，用 6 个 λ，和 V6 比 ΔPSNR（描述性）。

---

## 5. P3c：粗 mask 收细（试点，探索性，不做判定）

固定 100 张图：从排除类别 8/9 后的图里按 image_id 等间隔抽取，名单写进配置。t=0.9 的 R、λ=1。

1. 粗：$w_0=w_4$（t=0.9）。编辑 $x_1=x_s+w_0\odot R_0$。
2. 似然比图：在 $t_r=0.5$、2 个 seed（10、11）下，对 $x\in\{x_s,x_1\}$ 计算
   $\ell_p(x)=\frac12\sum_{i}\big(\|\hat\varepsilon_{\rm src}(z_i)-\varepsilon_i\|_p^2-\|\hat\varepsilon_{\rm tar}(z_i)-\varepsilon_i\|_p^2\big)$，
   其中 $z_i=a x+b\varepsilon_i$，$\|\cdot\|_p^2$ 是像素 p 上 4 个通道的平方和。共 8 NFE。
3. $\Delta\ell=\ell(x_1)-\ell(x_s)$。细化：$w_1=w_0\odot\mathbb 1[\overline{\Delta\ell}_p>0]$，$\overline{\cdot}$ 是 3×3 盒滤波。
4. 编辑 $x_2=x_s+w_1\odot R_0$，并做 λ∈{0.5, 1, 1.5, 2} 的曲线。

输出：$w_0$、$w_1$ 相对 GT 的 IoU 和 AUROC；与 V4 的曲线对比；12 张固定图的 mask 可视化。

---

## 6. 代码增量（在 v2 的 `geoedit/` 上）

```
geoedit/
├── configs/p3.yaml                 # λ 网格、变体表、c、池化尺寸、N、P3c 图像名单、锚点定义
├── geoedit/
│   ├── masks.py                    # 新：§2.1 的统计量和各种 mask；AUROC / IoU
│   ├── noise_ops.py                # 新：不动点噪声、对偶噪声、i* 选择
│   ├── reward.py                   # 新：§5 的似然比图
│   ├── metrics.py                  # 改：三个官方 CLIP 量（§1）
│   └── stats.py                    # 改：均值层面曲线匹配 + 图级 bootstrap（§3.3）
├── scripts/
│   ├── 30_p3a_build_and_eval.py    # 读 P1 的 R → 构造 V0–V11 → 解码 → 评测；按分片运行
│   ├── 31_p3a_analyze.py           # §3.2 描述 + §3.3 判定 + 作图
│   ├── 32_p3b_noise.py             # §4 生成 + 评测
│   ├── 33_p3c_refine_pilot.py      # §5
│   └── 34_p3_report.py             # 汇总成 REPORT_p3.md 的表格部分
└── results/p3/{a,b,c}/
```

| 文件 | 职责 | 验收 |
|---|---|---|
| `masks.py` | `snr_weight(Rs, c, pool)`、`diffedit_mask(Rs)`、`energy_mask(R)`、`auroc(w, gt)`、`iou(w, gt)` | T-K1–T-K4 |
| `noise_ops.py` | `fixed_point_eps(x_s, t, emb_src, eps0, J) → list[eps], diag`、`antithetic(eps)`、`select_draw(Rs, w)` | T-N1–T-N3 |
| `reward.py` | `lr_map(x, t, eps_list, emb_src, emb_tar) → 64×64` | T-R1 |
| `metrics.py` | 加入三个 CLIP 量，名称与官方一致 | T-X3 |
| `stats.py` | `mean_curve_match(curves, anchors, n_boot=1000)`：图级重采样，返回差值的点估计、CI、`out_of_range` | T-S3 |
| `30_p3a_build_and_eval.py` | 不保存 δ 的 latent（太多），只保存每行指标；对 §5 的 12 张固定图另存 PNG | 行数 = 700 ×（变体 × λ）+ recon；无 NaN（类别 8/9 的背景指标按 v2 的 D2 记 NaN） |
| `31_p3a_analyze.py` | 判定代码逐条注释引用 §3.3 的假设编号 | 判定表与 CSV 能互相核对 |

---

## 7. 测试

| ID | 内容 | 通过标准 |
|---|---|---|
| T-K1 | 零均值噪声：$R_i$ 为 N=4 个 iid 标准正态（4×64×64），c=1，重复 50 次 | $\mathrm{mean}(w)<0.1$（我这边的 numpy 模拟值为 0.087） |
| T-K2 | 强信号：N=4，$R_i=\mu+$ 标准正态噪声，每像素 4 通道合计 $\|\mu_p\|^2=10$，c=1 | $\mathrm{mean}(w)>0.85$（模拟值 0.91） |
| T-K3 | 把 mask 设为全 1 时，V2 的 δ 与 V0 逐元素相等；N=1 时 `snr_weight` 报错 | 通过 |
| T-K4 | AUROC：完美 mask 得 1，随机 mask 约 0.5 | 误差 < 0.02 |
| T-N1 | 高斯解析去噪器（对角 W∈(0,1)）上，不动点迭代误差等于 $(I-W)^J(\varepsilon^{(0)}-\varepsilon^\star)$ | 相对误差 < 1e-6 |
| T-N2 | `antithetic(eps)` 等于 −eps；与源、目标分支共享 | 通过 |
| T-N3 | 真实模型上 $A-R-e_{\rm src}\equiv0$ | 最大绝对值 < 1e-5 |
| T-R1 | 源 prompt = 目标 prompt 时 $\ell\equiv0$ | 最大绝对值 < 1e-6 |
| T-X3 | §1 的官方 CLIP 一致性 | 最大绝对差 < 1e-4 |
| T-S3 | 均值曲线匹配：两条已知直线的合成数据 | 插值正确；锚点越界时返回 `out_of_range` |

v2 的测试必须全部继续通过。

---

## 8. GPU 分配（4 × A100 40G，按 image 索引 mod 4 分片）

| 阶段 | 内容 | 预计耗时 | 进入下一阶段的条件 |
|---|---|---|---|
| A | 单元测试；T-X3；确认 P1 的 R 文件齐全（700×3×4）；在 20 张图上跑通 30 号脚本 | 约 30 分钟 | 全部测试通过 |
| B | P3a 解码与评测（约 10.5 万次），4 卡并行 | 约 2.5–3 小时 | 行数齐全 |
| C | P3a 分析 → **发中间报告**（H3a–H3d、G、方差分解、mask 质量） | 约 15 分钟（CPU） | — |
| D | P3b：不动点（每张图每个 t 约 13 NFE）+ 对偶噪声 + 评测（约 3.5 万次解码） | 约 1.5 小时 | — |
| E | P3c 试点（100 张） | 约 20 分钟 | — |
| F | 汇总报告 | CPU | — |

阶段 D、E 可以在 B 跑的时候用空闲卡提前生成 latent，但**分析必须在 P3a 中间报告发出之后再做**。

---

## 9. 纪律（在 v2 §8 基础上补充）

- §3.3 的锚点、c=1、3×3 池化、N、λ 网格、i* 的选择标准，都是在看到数据之前定的，不得修改。想加新变体可以，但要标为"事后探索"，单独成节。
- 任何方法组件都不能使用 CLIP 或 GT mask（V2、V3 除外，它们是 oracle）。
- 只报告均值层面的匹配；逐图插值的结果可以放附录，但不能作为判定依据（P2 的教训：逐图插值的 CI 会被很陡的曲线放大）。
- 负结果照常报告。

---

## 10. 交付物

`results/p3/`：
- `a/per_image.csv`：列 `image_id, category, t, variant, lambda, psnr, lpips, mse, ssim, struct_dist, clip_target_whole, clip_target_edit, clip_source_whole, mask_auroc, mask_iou`
- `a/summary.csv`：每个（t, variant, λ）的均值和 CI
- `a/variance_decomp.csv`：`image_id, category, t, region(in/out), V, S, frac_random`
- `a/mask_quality.csv`：`image_id, category, t, mask_type, auroc, iou`
- `a/hypotheses.json`：H3a–H3d 的判定、点估计、CI、p 值、n、`out_of_range`
- `b/fixed_point_diag.csv`：`image_id, t, j, e_src, step, eps_std, eps_kurt`
- `b/per_image.csv`、`b/hypotheses.json`（H3e）
- `c/pilot.csv`、`c/masks_png/`
- 图：每个 t 一张 Pareto 图（全部变体）、mask 示例网格（12 张固定图 × mask 类型）、$e_j$ 收敛曲线

`report/REPORT_p3.md`：
1. 环境；
2. 评测口径修正说明（§1）；
3. 判定表；
4. 方差分解与 mask 质量；
5. P3b、P3c 的描述；
6. 异常；
7. 偏离设计（`DEVIATIONS.md` 新增条目编号从 D10 开始）；
8. 运行时间与 NFE。

**中间报告**：阶段 C 结束后先交 `REPORT_interim_p3a.md`，放到 `deepseek_talk/p3/`。数据包格式同 P2：放汇总 CSV 和 json，不放逐图大文件。
