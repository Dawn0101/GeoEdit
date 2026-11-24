> **已归档（2026-11-24）。** 这是部分 OT 路线在立项当天写下的实施计划，对应原始任务书
> `../2026_11_24.md` 的第 6 节。该路线已停止投入：E1 实测证明 ChordEdit 的背景偏差是高频纹理重绘
> 而不是几何错位，OT 的空间容忍没有可发挥的余量。计划里的 E2/E3/E4/B2 **全部取消，一行没跑**。
> 保留本文件只作过程存档；当前状态见 `../docs/DECISION.md`。
> 其中仍然有效的部分：T0（环境 + 模型整形）、T1（求解器与单测）、T2（缓存）的设计与坑位记录。

---

# GeoEdit 实施计划（配套 `2026_11_24.md`）

- 生成日期：2026-11-24
- 服务器：8×A100-40GB / 96 核 / 754GB 内存 / `/data` 剩余 965GB
- 本文件只写"怎么做、按什么顺序做、每步怎么验收、哪里必须偏离任务书"。理论背景见 `2026_11_24.md`。

---

## 0. 结论（先说判断）

**idea 本身成立，可以开跑。** 三点依据：

1. 数学可验：LP 化简（`min<C−2τ,P>` 等价于带 TV 罚的部分 OT）正确；阈值性质 `C_ij>2τ ⟹ P_ij=0` 正确；`s_p→0` 退化为逐点残差阈值正确；`τ→0` 退化为 proposal（= ChordEdit）、`τ→∞` 退化为原图正确。也就是说方法覆盖了"逐点阈值"作为特例，消融是干净的。
2. 工程风险比预期小：**ChordEdit 官方代码已存在**（`github.com/ChordEdit/ChordEdit`，含 `run_pie_bench.py` + `pipeline_chord.py`），并且其默认超参与任务书 §5 完全一致（`t_start=0.90, t_delta=0.15, step_scale=1.0, t_end=0.30, noise_samples=1, cleanup=True`），核心位移也是在 x0 空间求的，所以任务书 §5 里"自己复现"的那一整套可以整段作废，改成"vendor 官方代码 + 校验数字"。基线复现从高风险项降为低风险项。
3. 实验设计的关键一环是对的：`B2_gtmask`（用 GT mask 当 σ 的上界）和 `M0_pointwise`（R=0）两个对照把"OT 到底有没有用"和"mask 估计离 oracle 差多少"分开了，这是审稿人最会问的两个问题。

**但收益上限要打折，原因是先验判断：**

- ChordEdit 的背景偏差来自 `û`，而 `û` 是两个时刻的 **x0 差**在高噪声段的加权平均，本身是**低频、平滑**的场。这种偏差更可能是"整幅重绘 / 色调漂移"，而不是"刚性平移几像素"。任务书 §3.3 已经指出，OT 的空间容忍**只在错位超过逐点容忍范围时才有效**。我的先验是 E1 落在"位移占比 < 20%"那一支的概率略大（≈60/40）。
- 因此本计划的执行策略是：**把 R=0（逐点阈值，但共享同一套代价/τ/合成规则）当作保底主方法**，R>0 当作"E1 支持才上量"。这样即使 E1 是负结果，产出仍然是一个完整的故事："统一代价视角 + 可解释价格 τ + 与逐点阈值的连续谱 + 更好的合成/特征"，而不是"OT 没用"。
- 真正的增益更可能来自**合成规则**（原位组合 / 像素域羽化合成）和**特征设计**（lat 对颜色敏感 vs dino 对结构敏感），而不是 OT 本身。计划里为这两块各加了 2 个几乎免费的消融。

**任务书里有 7 处需要修正，见 §5。** 其中第 1 条（`R` 与 `r_max` 不自洽）会直接毁掉 E3 扫描的可解释性，必须先改再跑。

---

## 1. 资产盘点（已核验）

### 1.1 已有，直接可用

| 资产 | 路径 | 说明 |
|---|---|---|
| PIE-Bench 全量 | `/data/lx/Dataset/PIE-Bench-Dataset` | 700 张 512×512，`mapping_file.json`（含 RLE mask、prompt、editing_type_id），`mask/` 下 `*_binary.png` / `*_mask.json` |
| CLIP **L/14** | `/data/lx/Model/CLIP` | 正是 PnPInversion 官方评测用的 `openai/clip-vit-large-patch14` |
| CLIP B/16 | `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch16` | 1.2GB，完整 |
| DINOv2-Large | `/data/lx/Model/DINOv2-Large` | ViT-L/14，供 E1 的 patch 特征（448 → 32×32） |
| 官方 ChordEdit | 需 vendor（已确认可 clone） | `ChordEdit/ChordEdit` |
| PnPInversion 评测代码 | 需 vendor（已确认可 clone） | `cure-lab/PnPInversion` → `evaluation/{evaluate.py,matrics_calculator.py}` |

### 1.2 缺，必须补

| 资产 | 体积 | 由谁提供 |
|---|---|---|
| `stabilityai/sd-turbo` | ~2.4GB（去掉 fp16 冗余后） | **用户下载**（见 §8） |
| LPIPS 线性权重 `squeeze.pth` | 6KB | agent 可用 ghproxy 自取（`raw.githubusercontent.com` 被墙，`ghproxy.net` 可达） |
| DINO ViT-B/8 权重 | ~330MB | `torch.hub` 自动下载；`dl.fbaipublicfiles.com` 可达（已验证 206） |

### 1.3 网络现状（实测）

| 目标 | 状态 |
|---|---|
| `huggingface.co` | ✗ 不可达 |
| `hf-mirror.com` | ✓ 200（API 与 resolve 均可用） |
| `github.com` / `codeload.github.com` | ✓ |
| `raw.githubusercontent.com` | ✗ 不可达 → 用 `ghproxy.net` 前缀 |
| `dl.fbaipublicfiles.com` | ✓ |
| `download.pytorch.org` | ✓ |
| `pypi.org`（pip 走 aliyun 镜像） | ✓ |

---

## 2. 环境：克隆 `trellis2`，不要从零建

现有 5 个带 torch 的环境里，`trellis2` 是最贴合的基座：

| 环境 | torch | diffusers | transformers | scipy | cv2 | timm | 判断 |
|---|---|---|---|---|---|---|---|
| **trellis2** | 2.6.0+cu124 | 0.31.0 | 4.57.6 | ✓ | ✓ | ✓ | **基座** |
| kairos | 2.12.0+cu126 | 0.31.0 | 4.45.2 | ✗ | ✓ | ✓ | 版本太激进 |
| flow_cycle | 2.6.0+cu124 | 0.31.0 | 4.45.2 | ✗ | ✗ | ✗ | 缺太多 |
| fusion_edit | 2.6.0+cu124 | 0.30.0 | 4.50.3 | ✗ | ✓ | ✗ | 缺太多 |

```bash
# 1) 克隆基座（6.4GB，几分钟）
conda create -n geoedit --clone trellis2 -y

# 2) 只补 6 个小包（走 aliyun 镜像，torch 不重装）
/data/lx/.conda/envs/geoedit/bin/pip install \
  pandas scikit-image matplotlib pytest \
  torchmetrics==1.8.2 lpips

# 3) 全局环境变量（写进 ~/.bashrc 或每个脚本的 env 头）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/lx/.cache/huggingface
export TORCH_HOME=/data/lx/.cache/torch
export PYTHONPATH=/data/lx/Code/GeoEdit/third_party/PnPInversion:$PYTHONPATH
```

验证（一次跑通才算 T0 完成）：

```bash
python -c "
import torch, diffusers, transformers, torchmetrics, lpips, skimage, pandas
print(torch.__version__, torch.cuda.device_count())
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
"
```

> 官方 ChordEdit 的 `requirement.txt` 写的是 torch 2.5 / py3.12。实测不需要严格对齐：`pipeline_chord.py` 只用到 `UNet2DConditionModel / DDPMScheduler / AutoencoderKL / CLIPTextModel`，diffusers 0.31 全部兼容。**不要**为此新建 torch 2.5 环境。

---

## 3. 目录结构

```
/data/lx/Code/GeoEdit/
├── 2026_11_24.md                 # 任务书（不动）
├── PLAN.md                       # 本文件
├── configs/                      # 每个 run 一份 yaml
├── third_party/
│   ├── ChordEdit/                # 官方代码，记录 pin 的 commit
│   └── PnPInversion/evaluation/  # 只用评测部分
├── src/geoedit/
│   ├── uot.py                    # 部分 OT 求解器（窗口 + 稠密参考）
│   ├── features.py               # lat-unfold / dino / both，特征归一化
│   ├── compose.py                # 原位组合、目标侧重心（对照）、门控 prox、像素域合成
│   ├── chord.py                  # 包官方 pipeline：transport / prox / 批量
│   ├── cache.py                  # latent 缓存读写
│   ├── metrics.py                # 包 PnPInversion MetricsCalculator
│   └── runner.py                 # 一个 config → 一个 results/<run_id>/
├── scripts/
│   ├── prepare_sd_turbo.py       # sd-turbo 目录整形 + 写 DDPM scheduler 配置（见 §5.4）
│   ├── build_cache.py            # 700 张 × {x_src, y, y_prox, noise}
│   ├── run_e0.py  run_e1.py  run_e2.py  run_e3.py  run_e4.py
│   ├── eval_run.py               # 对一个 run 目录算全部指标
│   └── collect.py                # 汇总大表 + Pareto csv
├── cache/                        # ~200MB
├── results/<run_id>/{per_image.csv,config.json,summary.json}
└── figures/                      # E4 的 12 张拼图
```

---

## 4. 任务分解

时间按"单人 + 8 卡"估。每项都写了**验收标准**，没通过不许进下一步。

### T0 环境与资产（0.5 天）

- 克隆环境、补包、env 变量（§2）。
- vendor 两个仓库，记录 commit hash 进 `third_party/PINS.md`。
- `prepare_sd_turbo.py`：把 sd-turbo 的 HF 目录整形成 `unet/ vae/ text_encoder/ tokenizer/ scheduler/` 五件套（`scheduler/` 必须替换成 DDPM 配置，见 §5.4）。
- 预取 `squeeze.pth` 到 `~/.cache/torch/hub/checkpoints/`，预取 DINO ViT-B/8。
- **验收**：3 张图跑通官方 pipeline，输出图肉眼正常；`MetricsCalculator` 能对一张图算出 psnr/lpips/ssim/clip/struct_dist 五个数。

### T1 求解器移植与单测（0.5 天，不需要 sd-turbo）

- 落 `src/geoedit/uot.py`（任务书 §4.1 代码 + 稠密 numpy 参考 §4.2）。
- 单测（`pytest tests/test_uot.py`）：
  1. 稠密 numpy vs 窗口 torch：`H=W=8, R=7` 与 `H=W=12, R=2`，40000 步，`max|Δσ| < 1e-4`（任务书给定阈值）。
  2. 与 LP 精确解（`scipy.optimize.linprog`）：值差 < 1e-2（ε=0.01），且 `C>2τ` 上的质量 < 1e-10。
  3. 单调性：τ ∈ {0.25,0.5,1,2,4} 上保留质量单调不减。
  4. 退化：`R=0` 时 σ ∈ {0,1} 且与 `C_kk < 2τ` 逐点一致。
  5. 数值：随机 50 组（含全 inf 边界、极小 s_f）不出现 NaN；记录迭代未收敛比例。
- **必须新增**：`R = ceil(s_p·√(2τ)) + 1` 的自动推导（§5.1），以及 `both` 特征的单位化（§5.3）。
- 测速：64×64、R=4、300 iters 的 GPU 单图耗时；D=768 时实测显存并决定分块大小。
- **验收**：全部单测绿；单图 OT < 30ms（64×64, R≤8）。

### T2 缓存生成 + 基线校准（0.5 天）—— **这是第一个 gate**

- `build_cache.py`：700 张，每张存 `x_src`(scaled latent)、`y`(transport、无 cleanup)、`y_prox`(transport + cleanup)、`noise`、`seed`、以及 `img_src / img_y / img_y_prox` 解码图。fp32，`.npz`，单卡 ~5 分钟，8 卡分片 <1 分钟。
- 用官方参数跑 `B0_chord_noprox`（`--no-cleanup`）与 `B1_chord`（默认，`cleanup=True`），走官方 `MetricsCalculator`。
- **Gate G1**：PSNR/CLIP-E 与论文对照

  | run | 论文 PSNR | 论文 CLIP-E | 允许偏差 |
  |---|---|---|---|
  | B0 | 23.89 | 21.87 | ±1.0 dB / ±0.5 |
  | B1 | 22.20 | 22.96 | ±1.0 dB / ±0.5 |

  不达标 → 按任务书 §5 扫 `λ ∈ {0.8,1.0,1.2,1.5}` 先对齐 CLIP-E，把最终 λ 和偏差写进 `docs/baseline.md`，**再**往后走。
- **验收**：`results/B0_chord_noprox/` 与 `results/B1_chord/` 各三件套齐全；G1 表格填好。

### T3 E1：proposal 背景偏差的性质（0.5 天）—— **第二个 gate，决定方向**

- 背景定义：GT mask 补集，在 64×64 latent 网格上再腐蚀 2px。
- 每个 8×8 背景 patch：`r0`（零位移 MSE）、`r*`（位移 ∈ [−3,3]² 块匹配最小 MSE）、`gain=1−r*/r0`、最优位移；另加 DINOv2 余弦距离（解码图）。
- 输出：显著位移占比（整体 + 10 类）、`r0`/`gain` 的 p25/p50/p75/p90、`2τ ∈ {2,4,8}` 下的背景误判率。
- **Gate G2 决策树**：
  - `占比 < 10%` 且 `median gain < 0.1` → **分支 A**：重点转 E3 的特征对比 + 合成规则，`R>0` 只跑 `feat=both, τ=4` 一列做交代。
  - `占比 ≥ 20%` → **分支 B**：`R>0` 全量扫，主推 M1。
  - 中间地带 → 按类别拆（`change style / change background / add object` 往往位移多，`change color / material` 往往漂移多），主表按类别报告。
- **验收**：`results/E1/report.json` + 一张 markdown 结论表，明确写出走 A 还是 B 以及依据。

### T4 E0 剩余 + E2 主结果 + 合成消融（0.5 天）

`B2_gtmask` 与 `M0–M5` 全部从缓存重算，每个配置只花 1 NFE（prox）+ 毫秒级 OT。

| run_id | 内容 | 相对任务书 |
|---|---|---|
| `B2_gtmask` | σ = 1 − GT mask（下采样 + 腐蚀） | 原样 |
| `M0_pointwise` | `R=0` | 原样 |
| `M1_uot` | 默认 | 原样，但默认改为 `feat=lat@64×64`（§5.2） |
| `M2_uot_target_bary` | 目标侧重心组合 | 原样（验证 §2.4 教训） |
| `M3_absdv_mask` | `|û|` 归一化阈值 mask | 原样 |
| **`M3b_area_matched`** | M1 的 τ 逐图标定，使编辑面积 = M3 的面积 | **新增**（见 §5.6） |
| **`M4_prox_then_compose`** | 先 `prox(y)`，再与 `x_src` 合成 | **新增**（见 §5.5） |
| **`M5_pixel_feather`** | 像素域羽化合成 | **新增**（见 §5.5） |

- 所有 run 共用同一份 `noise`（seed 固定），prox 噪声跨配置完全一致。
- **验收**：汇总大表（行 = run，列 = PSNR / LPIPS / MSE / SSIM / CLIP-W / CLIP-E / StructDist / 耗时 / NFE / 编辑面积）+ 至少 6 组定性对比图。

### T5 E3 扫描（0.5–1 天，8 卡并行）

- 主扫 `τ ∈ {1,2,4,8,16} × s_p ∈ {1,2,3} × feat ∈ {lat,dino,both}`，`R` 由 §5.1 自动推导（不再手工给 R）。
- 时间紧时的优先序：`feat=lat` 的 15 组 → `feat=both` 的 15 组 → `feat=dino` 的 15 组 → `ε ∈ {0.05,0.1,0.3}` 3 组。
- 并行方式：**按配置分卡**（每卡一个进程跑完整 700 张 + 当场算指标），比按图片分片简单且不会有写冲突。
- 预算：单配置 = prox 700 张（batch 32，约 15s）+ 指标（约 3min）≈ 4min；60 个配置 ≈ 4 小时单卡 → **8 卡约 30–40 分钟**。
- **验收**：`results/E3/pareto.csv`（config_id, tau, s_p, R, eps, feat, psnr, lpips, clip_e, type_id），以及按 10 类拆分的同结构文件。

### T6 E4 mask 质量 + 拼图 + 汇总（0.5 天）

- 对 M0/M1/M3/M3b 与 E3 全部配置：`1−σ>0.5` 与 GT mask 的 IoU、`mean(1−σ)`、GT 面积比例。
- 12 张固定图（每类 ≥1，列出 image id）拼 `[原图 | proposal | σ 热图 | GT mask | 输出]`。
- 汇总大表 + 与论文数字的差异说明 + 异常清单（NaN / 不收敛 / OOM / 与论文差距）。
- **验收**：对齐任务书 §8 的 7 项，逐项打勾。

### T7 撰写素材（0.5 天）

- 一张"τ–面积–PSNR–CLIP-E"的联合曲线图（说明 τ 确实是单调旋钮）。
- 一张 `R` 从 0 到 6 的 mask 热图演化（说明空间容忍在做什么）。
- 失败案例分析：proposal 本身没改成功的样本（`CLIP-E(proposal)` 低分位）。

**总计 ≈ 3.5–4 天**；如果只要"这个方向值不值得做"的结论，**T0–T3 三天内**就能给出（E1 + B0/B1 校准 + B2 上限）。

---

## 5. 相对任务书必须修正的 7 点

### 5.1 `R` 与 `r_max` 不自洽（**必须先改**）

任务书给 `r_max = s_p·√(2τ)`，但默认配置 `τ=4, s_p=2` 时 `r_max = 5.66 > R = 4`，E3 里 `s_p = R/2` 时 `r_max = 1.41R > R`。也就是说**窗口比理论搬运半径还小**，τ 的阈值语义被窗口截断，扫描出来的"τ 越大保留越多"就不再是价格效应。

改法：**由 `(τ, s_p)` 反推 `R = ceil(s_p·√(2τ)) + 1`**，并把实际 `R` 写进 `config.json`。E3 因此扫 `(τ, s_p)` 而不是 `(τ, R)`。

### 5.2 OT 网格统一到 64×64

任务书默认 `feat=both` 在 32×32 上解 OT，再上采样到 64×64 latent 去合成。32×32 的 σ 上采样后边界误差是 16 latent px（= 32 图像 px），太粗，而 toy 已经证明熵模糊是 OT 的主要劣势。

改法：**网格固定 64×64**（latent 原生），DINO patch 特征双线性上采样到 64×64 后拼接；`s_p`/`R` 统一以 64 网格为单位，σ 不再需要插值。注意 D=804 时 `window_cost` 的中间张量约 1GB/图，需按 8–16 个目标 token 分块。

### 5.3 `both` 特征的双重归一化

"各自按 `normalize_feature_scale` 归一化再拼接"会让代价变成两项之和，量纲翻倍，`τ_both ≈ 2τ_lat`，两个 feat 的 τ 不可直接比较。

改法：各自除以本 block 的 `s_f` 后**再除以 block 数**，使 `both` 的代价与单特征同量纲；并在 `config.json` 里记录每个 block 的 `s_f` 原值。

### 5.4 sd-turbo 的 scheduler 类不匹配

官方 `pipeline_chord.from_local_weights` 用 `DDPMScheduler.from_pretrained(<scheduler 目录>)`，而 sd-turbo 官方发布的是 `EulerDiscreteScheduler`，直接 `from_pretrained` 会因 `_class_name` 不符报错。

改法：`prepare_sd_turbo.py` 里新建 `sd-turbo/scheduler_ddpm/scheduler_config.json`：

```json
{"_class_name":"DDPMScheduler","num_train_timesteps":1000,
 "beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear",
 "trained_betas":null,"prediction_type":"epsilon","variance_type":"fixed_small",
 "clip_sample":false,"steps_offset":1,"timestep_spacing":"leading",
 "rescale_betas_zero_snr":false}
```

其余四个组件（unet/vae/text_encoder/tokenizer）用符号链接指向原目录。这样 `_max_unet_timestep=999`，与任务书 §5 的 `round(t·999)` 一致。

### 5.5 合成规则要单独消融（新增 M4 / M5）

toy 结论（§2.4）说明**合成规则的影响可能大于 OT 本身**，但任务书只在 E2 里放了 M2 一个对照。补两个几乎免费的：

- `M4_prox_then_compose`：`z = σ⊙x_src + (1−σ)⊙prox(y)`。避免"先把两幅 latent 混在一起再送进 UNet"造成的低频糊化。
- `M5_pixel_feather`：解码后在像素域用羽化 mask 合成。背景 PSNR 会显著抬高（VAE 往返误差被完全绕过），但可能出现色调断缝，看 LPIPS 与实际观感。注意报告时要说明它是像素域方法，与 latent 域方法不是同一个比较层级。

### 5.6 M1 vs M3 存在面积混淆（新增 M3b）

任务书让 M3 按"与 M1 相同的编辑面积比例"取阈值，但 M1 自己的面积并未被约束。于是 CLIP-E 的差异可能只是面积差异。补一个 `M3b_area_matched`：**对 M1 逐图标定 τ 使编辑面积等于 M3 的面积**。只有 M1 在等面积下仍优于 M3，才算真优。

### 5.7 P2 的斜率符号（文档小错，不影响结论）

目标值 `V(τ) = <C,P> + 2τ(1−m)`，按包络定理 `dV/dτ = 2(1−m(τ))`，是**递增**凹函数；任务书写的 `−2m(τ)` 差了一个 `2τ` 的常数项。结论（`m(τ)` 单调不减、两端点）不受影响，但写论文时别照抄。

### 5.8 额外实现约定（避免踩坑）

- **每张图单独解 OT**，不能把多图拼 batch（质量按图归一化）。
- 缓存里存 `noise`，所有配置共用；prox 的 UNet 调用**可以**跨图 batch（16–32），transport 不能（已在缓存里）。
- `x_src` 与 `y` 都在 **scaled** latent 空间（`vae.config.scaling_factor`），解码时除回去。
- σ 阈值取 `1−σ > 0.5` 记 IoU；同时报 `mean(1−σ)` 的连续面积，避免只看二值。
- 所有 run 的 `img_out` 只在 12 张拼图 + 每 run 8 张样例时落盘（JPEG q95），否则只留指标 → 60 个配置的图片不会吃掉几十 G。

---

## 6. Gating 一览

```
T0 环境 ──► T1 单测 ──► T2 缓存 ──► G1(PSNR/CLIP-E 对齐论文 ±1dB)
                                      │
                                      ├─ 不过 → 扫 λ，写偏差说明，再走
                                      ▼
                                   T3 E1 ──► G2 分支 A / B 决定后面的赌注
                                      ▼
                            T4 E0+E2(+M3b/M4/M5) ──► T5 E3 ──► T6 E4 ──► T7 素材
```

---

## 7. 交付物（对齐任务书 §8）

| # | 任务书要求 | 落地位置 |
|---|---|---|
| 1 | 每 run 三件套 | `results/<run_id>/{per_image.csv,config.json,summary.json}` |
| 2 | 汇总大表 | `docs/summary_table.md` |
| 3 | E1 统计 | `results/E1/{report.json,report.md}` |
| 4 | E3 Pareto csv | `results/E3/pareto.csv` + `pareto_by_type.csv` |
| 5 | 12 张拼图 + IoU | `figures/panel_*.jpg`、`results/E4/iou.csv` |
| 6 | 基线复现说明 | `docs/baseline.md`（官方代码 commit、是否官方、最终 λ、与论文差值） |
| 7 | 异常清单 | `docs/anomalies.md`（NaN 率、不收敛率、OOM、与论文差距大的数字） |

---

## 8. 需要用户提供的唯一资产

`stabilityai/sd-turbo` → `/data/lx/Model/sd-turbo`（~2.4GB，只要 fp32）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
/data/lx/miniconda3/envs/hf_download/bin/hf download stabilityai/sd-turbo \
  --local-dir /data/lx/Model/sd-turbo \
  --include "model_index.json" "unet/*" "vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" \
  --exclude "*.fp16.safetensors" "*.h5" "*.msgpack" "*.bin" "*.png" "*.jpg"
```

如果已经有 sd-turbo（或等价的 SD2.1-base 单步蒸馏模型），直接给路径，T0 里做整形即可。
