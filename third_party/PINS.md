# Vendored / fetched third-party code (pinned)

| repo | commit | 如何处理 | 用到的文件 |
|---|---|---|---|
| https://github.com/ChordEdit/ChordEdit | `a33d8cc26108851c6545b998c0e939006f89c9bd` | **入库**（MIT，保留 `third_party/ChordEdit/LICENSE`） | `pipeline_chord.py`, `run_pie_bench.py`, `utils.py` |
| https://github.com/cure-lab/PnPInversion | `07f97f448150e2ca220bebd54c8f687c5c50c67a` | **不入库**（无 LICENSE → 版权保留），用 `third_party/fetch.sh` 拉取 | `evaluation/evaluate.py`, `evaluation/matrics_calculator.py` |

不要改动 `third_party/` 下的任何文件；需要额外行为时请在外面包一层
（见 `src/geoedit/chord.py` 与 `src/geoedit/metrics.py`）。

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
