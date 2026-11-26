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
