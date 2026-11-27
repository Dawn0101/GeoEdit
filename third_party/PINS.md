# Vendored / fetched third-party code (pinned)

| repo | commit | 如何处理 | 用到的文件 |
|---|---|---|---|
| https://github.com/ChordEdit/ChordEdit | `a33d8cc26108851c6545b998c0e939006f89c9bd` | **入库**（MIT，保留 `third_party/ChordEdit/LICENSE`）；P6 从 `origin/v1` 移植 | `pipeline_chord.py`, `run_pie_bench.py`, `utils.py` |
| https://github.com/cure-lab/PnPInversion | `07f97f448150e2ca220bebd54c8f687c5c50c67a` | **不入库**（无 LICENSE → 版权保留），用 `third_party/fetch.sh` 拉取 | `evaluation/evaluate.py`, `evaluation/matrics_calculator.py` |
| https://github.com/Xiao0219/BudEdit | — | **无法入库：官方代码未发布**（README 明写 "Code Coming soon"，2026-11-27 核对） | — |
| PreciseEdit（arXiv 2610.23535） | — | **无法入库：未找到官方代码**（论文正文/项目页均无仓库链接，2026-11-27 核对） | — |

不要改动 `third_party/` 下的任何文件；需要额外行为时请在外面包一层
（见 `geoedit/chord.py` 与 `geoedit/metrics.py`）。

## P6 的基线状态（2026-11-27）

* **ChordEdit**：`geoedit/chord.py` 已从 `origin/v1` 移植（`ROOT` 从 `parents[2]` 改为
  `parents[1]`，其余未动）。v1 用它跑过 700 图官方指标，**输出图仍在
  `cache/images/{B0_chord_noprox,B1_chord}/`**（各 700 张），P6 直接复用、不重跑。
* **BudEdit**：最接近我们的方法（残差能量支撑 + 注意力门 + 两次 UNet），但官方代码未发布，
  按交接文档"官方代码和官方超参，不自己重写"的规定，P6 **不跑**，在报告里记缺失。
* **PreciseEdit**：与我们的 k 项同族（见 `REPORT_p6.md` §3.2），无代码，P6 **不跑**，只点名。

## 为什么这样分

* ChordEdit 是 MIT，可以随仓库分发，保留 LICENSE 即可，这样基线部分仓库自包含。
* PnPInversion 仓库里**没有 LICENSE 文件**，按默认版权保留处理；把它原样提交到公开仓库不合适。
  我们实际上只 import 它的 `evaluation/matrics_calculator.py`，所以改成"用时拉取"。

## ChordEdit 的演示素材不随仓库分发

`images/`、`chord_show.gif`、`chord_app.png`（共 15 MB，README 里的展示图）已从本仓库剔除，
只保留代码。需要的话：

```bash
git -C third_party/ChordEdit checkout  # 若曾 clone 过
# 或直接看上游：https://github.com/ChordEdit/ChordEdit
```
