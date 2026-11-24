#!/usr/bin/env bash
# 按 PINS.md 记录的 commit 拉取第三方代码。
#
# ChordEdit 是 MIT,已随本仓库分发（third_party/ChordEdit，保留其 LICENSE）。
# PnPInversion 没有 LICENSE 文件（默认版权保留），因此**不入库**，由本脚本就地拉取。
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch() {
  local repo="$1" dir="$2" commit="$3"
  if [ -d "$here/$dir/.git" ]; then
    echo "[skip] $dir 已存在"
    return 0
  fi
  rm -rf "$here/$dir"
  echo "[clone] $repo @ $commit"
  git clone --quiet "https://github.com/$repo" "$here/$dir"
  git -C "$here/$dir" checkout --quiet "$commit"
  echo "[ok] $dir"
}

fetch cure-lab/PnPInversion PnPInversion 07f97f448150e2ca220bebd54c8f687c5c50c67a
echo
echo "只用它的 evaluation/：我们的 src/geoedit/metrics.py 会 import"
echo "  evaluation/matrics_calculator.py（未改动）"
