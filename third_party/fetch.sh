#!/usr/bin/env bash
# 按 PINS.md 记录的 commit 拉取第三方代码。
#
# ChordEdit 是 MIT，代码已随本仓库分发（third_party/ChordEdit，保留 LICENSE；演示素材被剔除）。
# PnPInversion 没有 LICENSE（默认版权保留），因此**不入库**，由本脚本就地拉取。
# 注意：我们只用它的 evaluation/（36 KB），不拉 models/ 与 scripts/（47 MB）。
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="cure-lab/PnPInversion"
dir="$here/PnPInversion"
commit="07f97f448150e2ca220bebd54c8f687c5c50c67a"

if [ -d "$dir/evaluation" ]; then
  echo "[skip] $dir/evaluation 已存在"
  exit 0
fi

rm -rf "$dir"
echo "[clone] $repo @ $commit (sparse: evaluation only)"
# 本机直连 github 被拒；仓库里已配好 ghproxy.net 镜像（见 .git/config remote.mirror）。
# 需要直连时加 USE_DIRECT=1。
if [ "${USE_DIRECT:-0}" = "1" ]; then
  url="https://github.com/$repo"
else
  url="https://ghproxy.net/https://github.com/$repo"
fi
echo "[url] $url"
git clone --quiet --depth 1 --filter=blob:none --sparse "$url" "$dir"
git -C "$dir" sparse-checkout set evaluation
head_sha="$(git -C "$dir" rev-parse HEAD)"
if [ "$head_sha" != "$commit" ]; then
  echo "[warn] HEAD=$head_sha 与 pin 的 $commit 不一致；如需严格复现请手动 checkout"
fi
rm -rf "$dir/.git"
echo "[ok] $dir/evaluation ($(du -sh "$dir" | cut -f1))"
