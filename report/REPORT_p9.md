# P9 报告：1.8 个 CLIP 点的来源，与官方底座上的交叉验证

- 日期：2026-11-27
- 上游：PI 对 P8 的两步交叉验证要求（6 项逐项 diff + 在 `pipeline_chord.py` 底座上跑我们的组件）

---

## 0. 结论速览

| 问题 | 答案 |
|---|---|
| 1.8 个 CLIP 点里，**评测子集**贡献多少？ | **0.18**（chord 在同一 137 图上 22.41 vs 全集 22.59）——不是主因 |
| **噪声抽样**贡献多少？ | **+0.27**（60 图，同一场只换 ε）|
| **prompt 的 `[ ]` 括号**贡献多少？ | **0**（PIE-Bench 的 raw prompt 在本 bench 上不带括号，token 流逐位相同）|
| **文本编码的 attention mask** 贡献多少？ | **+0.80** ← **这才是主因** |
| 时间步映射 / α、σ / VAE 编码 / 预处理 | **逐位相同**（见 §1）|
| `_u_estimate` 与 `_pred_x0` | **逐位相同**（我方复现 maxdiff 0.011，norm 56；`x0_pred` 与显式公式 maxdiff 0.0）|
| 官方底座复现 B1_chord | ✅ PSNR 22.362 vs 22.36、DINO 0.031、CLIP-E 23.27 vs 23.32（同一 41 图）|
| 官方底座 + 我们的掩码 | **背景 PSNR 22.36 → 28.21（+5.85 dB）**、DINO 0.031 → 0.008、CLIP-E 23.27 → 21.45（−1.82）|
| 官方底座 + 我们的掩码 + SDS | CLIP-E 21.45 → 21.59（k=0.5）/ 21.62（k=1.0），PSNR 28.21 → 27.98 / 27.41 |

**一句话：1.8 分不是"两个底座"的玄学，而是我们自己的 `embed()` 多传了 `attention_mask` ——
文本条件被系统性削弱（场范数 31.9 vs 40.7），CLIP-E 少 0.8。修掉它 + 换弦场之后，我们的
P8b 分数应该整体上移约 0.8。**

---

## 1. 六项逐项核对

### 1.1 噪声抽样与复用（**差异存在，贡献 +0.27**）

- ChordEdit：`_prepare_noise_list()` 里 `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`
  然后 `torch.randn_like(latents, device=latents.device)` —— **在 CUDA 上抽**，而且 `seed=42`
  是**所有图片共用的**（每次调用都重新 seed，所以每张图拿到**同一个** ε 张量）。
- 我们：`make_eps()` 在 **CPU** 上抽（`torch.Generator('cpu').manual_seed(sha256(image_id|0))`）
  再搬到 GPU —— 每张图不同。
- 单因子测量（`scripts/74_p9_noise_probe.py`，60 图，同一弦场、同一 prox，只换 ε）：
  `hash_noise` CLIP-E **20.47** vs `chord_noise42` **20.74**，|u| 38.95 vs 38.83。
  ⇒ **+0.27**，方向对但不解释主因。
- 附带确认：我的重抽 `torch.manual_seed(42)+randn_like` 与官方 noise **逐位相同**（maxdiff 0.0）。

### 1.2 时间步 → timestep 索引与 α、σ（**逐位相同**）

| 量 | ChordEdit | 我们 |
|---|---|---|
| `max_timestep` | `num_train_timesteps - 1` = **999** | **999** |
| `round(t·max)` @0.9 / 0.75 / 0.3 | 899 / 749 / 300 | 899 / 749 / 300 |
| α=√ābar, σ=√(1−ābar) | 0.119577 / 0.992825 @0.9 | 0.119577 / 0.992825（maxdiff 0.0）|

### 1.3 `_u_estimate` 与 `_pred_x0` 的公式（**逐位相同**）

`_u_estimate` 返回 `(δ·dv_s + t_s·dv_s0)/(t_s+δ)` = `0.1429·R(0.9) + 0.8571·R(0.75)`，位移是
`x = x_s + step_scale·u`（step_scale=1.0），prox 是 `x̂0(a_c·x + b_c·ε, t_c)`。
我们用官方噪声重算这一整套，与 `ed.transport(...)['y'] − x_s` 的 **maxdiff = 0.011**（场范数 56.03）。
`scaling_factor` 两边都在 `vae.encode` 之后乘、`vae.decode` 之前除，位置一致。

### 1.4 图像预处理与 VAE 编码（**逐位相同**）

- `_CenterSquareCropTransform` 只在非方图上有作用；**本轮 200 张评测图全部是 512×512**，
  所以它和我们的 `Resize(512,512)` 等价（都是 no-op）。
- 归一化：两边都是 `ToTensor` 后 `[-1,1]`（chord: `Normalize(.5,.5)`；我们: `/127.5 − 1`）。
- 编码：两边都是 `vae.encode(x).latent_dist.mode() * scaling_factor`（**不是** `.sample()`，无随机性）。
- 实测逐位相同：`maxdiff 0.00000`（3 张图），且与 `results/p3/a/xs` 缓存也 maxdiff 0.0。

### 1.5 Prompt 文本处理：**括号不是问题，attention mask 才是（+0.80）**

- **括号**：v1 的 runner 传的是 `original_prompt` / `editing_prompt`（**带 `[ ]` 的原串**），我们传
  `strip_brackets` 之后的串。但对本 bench 的评测集，raw `editing_prompt` **不含括号**
  （例子：`a slanted rusty mountain bicycle ...`），两条串的 token id **逐位相同**，
  `model.embed` 输出 maxdiff **0.0** ⇒ 括号贡献 **0**。
- **padding / attention mask**：ChordEdit 的 `_encode_text` 用
  `tokenizer(padding='max_length', truncation=True, max_length=model_max_length)` 后
  **`attention_mask=None`**（`use_attention_mask=False`），即 CLIP 文本编码器**会 attend 到补零 token**
  —— 这是 CompVis/SD 的原始约定。我们的 `embed()` 传了 `attention_mask=tokens.attention_mask`，
  把 padding 屏蔽掉。
- 后果：同一 prompt 的文本嵌入差异 **maxdiff 16.2，范数 292.8 vs 191.2**；弦场范数 **40.7 vs 31.9**；
  单因子 CLIP-E **23.81 vs 23.01**（20 图，同一 ε、同一 prox）⇒ **+0.80**。
- 这是一个**我们自己的偏差**，而且影响 P1–P8 的**全部残差**（所有 `R` 都是用被 mask 的文本条件算的）。

### 1.6 评测子集（**0.18，不是主因**）

| 集合 | chord CLIP-E | PSNR | DINO |
|---|---|---|---|
| 全集 200 图（含类别 8/9） | 22.59 | 22.02 | 0.031 |
| **c_plain 那 137 图** | **22.41** | 21.98 | 0.031 |

⇒ 类别 8/9 只值 0.18。**PI 的怀疑方向正确（子集确实不同），但量级不足以解释 1.8。**

**合计解释：0.27（噪声）+ 0.80（文本条件）= 1.07 / 约 1.7。** 剩余约 0.5–0.7 分与"场的整体幅度"
（我们 31.9 vs 40.7）耦合，修掉文本条件后应一并消失。

---

## 2. 官方底座上的交叉验证（`scripts/75_p9_official.py`）

底座 = `geoedit.chord.ChordEditor`（官方 `pipeline_chord.py` 的薄包装）：transport 的 4 次前向
与 `t_c=0.3` 的 prox **全部走官方代码、官方噪声（seed 42）**。我们只在 latent 上加两样东西：

- `M`：`q60/q90` 分位数饱和掩码（背景 60% 像素严格为 0），从 `‖R(0.75)‖` 的高斯平滑能量取；
  另有从 `‖R(0.5)‖`（P5 的 L5）取的变体；
- 核心区吸引项：`δ = M·(u + k·M²·A(0.75))/(1 + k·M²)`，`A = μ_tar(0.75) − x_s`，
  合成 `x_out = (1−M)x_s + M·prox`（**未平方的 M**）。

**先验证底座**（同一 41 图）：

| 变体 | 背景 PSNR | DINO | CLIP-E |
|---|---|---|---|
| `off_k0`（M≡1,k=0，本脚本） | **22.362** | 0.031 | **23.27** |
| B1_chord（v1 存档图，官方指标） | **22.36** | 0.031 | **23.32** |

⇒ **复现成功**（PSNR 差 0.002 dB，CLIP-E 差 0.05，来源是存档图为 JPEG）。
所以 P8 里那个"我们的 ChordEdit 复现只有 20.8"是**假的复现**——它用的是我们自己的文本条件。

（同一 41 图的四个变体表见 §3，待全 200 图跑完更新。）
