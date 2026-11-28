"""P9 viewer: a fully STATIC HTML page (no JS, no fetch) comparing all methods side by side.

Static because a JS/fetch page is all-or-nothing: if the data request or a script fails, the page is
blank.  Here Python writes every row into the HTML, so what you see is what the server sent.

One row per source image, six columns (source | ChordEdit | ours mask+DDS | +SDS 0.5 | +SDS 1.0 |
ours mask from t=0.5), the three prompts above each row, PSNR / DINO / CLIP-Edited under each cell.

Serve with: python -m http.server 3081 --directory results/p9/viewer
"""
from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml  # noqa: E402

# (image key, label html, image dir relative to cache/images, metrics variant key)
COLUMNS = [
    ("src", "source", None, None),
    ("off_k0", "ChordEdit<br><span class=nm>official 5 NFE</span>", "p9/off_k0", "off_k0"),
    ("m075_k0", "Ours P9<br><span class=nm>soft mask, double-gated</span>", "p9/m075_k0", "m075_k0"),
    ("plat_out", "Ours P10<br><span class=nm>plateau mask, single gate</span>", "p10/plat_out", "plat_out"),
    ("erchord_mu1", "ER-Chord<br><span class=nm>equal-energy (mu=1)</span>", "p11/erchord_mu1", "erchord_mu1"),
    ("budedit", "BudEdit<br><span class=nm>energy budget</span>", "p11/budedit", "budedit"),
    ("precise_a1", "PreciseEdit<br><span class=nm>SCT, gamma=1</span>", "p11/precise_a1", "precise_a1"),
    ("scadA_a1", "Ours P11<br><span class=nm>SCAD-A alpha=1, 5 NFE</span>", "p11/scadA_a1", "scadA_a1"),
    ("scad_cleanbg", "Ours P11<br><span class=nm>SCAD + clean background</span>", "p11/scad_cleanbg", "scad_cleanbg"),
]
KEYS = [c[0] for c in COLUMNS]

CSS = """
:root{--ink:#1a1d23;--dim:#6b7280;--line:#e5e7eb;--soft:#f6f7f9;--head:#f0f2f5;
      --good:#0a7d43;--bad:#c0392b;--chip:#eef2ff;--chipfg:#3b4a8c}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:16px 20px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:17px}
.sub{color:var(--dim);font-size:12px}
table.summary{border-collapse:collapse;margin-top:10px;font-size:12px}
table.summary th,table.summary td{border:1px solid var(--line);padding:4px 10px;text-align:right}
table.summary th:first-child,table.summary td:first-child{text-align:left}
table.summary th{background:var(--head);font-weight:600}
table.summary tr.hl td{background:#fffbeb}
main{padding:14px 20px 80px}
.grid{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-bottom:10px}
.grid>div{background:var(--head);padding:7px 8px;text-align:center;font-weight:600;font-size:12px}
.nm{font-weight:400;color:var(--dim);font-size:11px}
.row{border:1px solid var(--line);border-radius:10px;margin-bottom:14px;overflow:hidden}
.meta{padding:8px 12px;background:var(--soft);border-bottom:1px solid var(--line)}
.l1{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;font-size:12px}
.mid{font-family:ui-monospace,Menlo,monospace;font-weight:700}
.chip{background:var(--chip);color:var(--chipfg);border-radius:5px;padding:0 7px;font-size:11px}
.ab{font-size:11px;color:var(--dim)}
.pos{color:var(--good);font-weight:700}.neg{color:var(--bad);font-weight:700}
.pr{margin-top:6px;font-size:12px;line-height:1.75}
.pr i{display:inline-block;min-width:76px;color:var(--dim);font-style:normal}
.cells{display:grid;gap:1px;background:var(--line)}
.cell{background:#fff}
.cell img{display:block;width:100%;height:auto;background:#fff}
.m{padding:3px 6px;font-size:11px;color:var(--dim);display:flex;gap:8px;font-variant-numeric:tabular-nums;white-space:nowrap}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="results/p9/viewer")
    ap.add_argument("--width", type=int, default=190)
    args = ap.parse_args()
    base = load_yaml(ROOT / args.config)
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")

    import glob
    frames = [pd.read_csv(ROOT / base["output_root"] / "p9" / "frontier" / "per_image.csv",
                          dtype={"image_id": str})]
    frames += [pd.read_csv(p, dtype={"image_id": str}) for p in
               sorted(glob.glob(str(ROOT / base["output_root"] / "p10" / "frontier" /
                                     "per_image_shard*.csv")))]
    frames += [pd.read_csv(p, dtype={"image_id": str}) for p in
               sorted(glob.glob(str(ROOT / base["output_root"] / "p11" / "frontier" /
                                     "per_image_shard*.csv")))]
    per = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image_id", "variant"])
    ab = pd.read_csv(ROOT / base["output_root"] / "p9" / "ab.csv",
                     dtype={"image_id": str}).drop_duplicates(subset=["image_id", "pair"])
    summary = pd.concat([
        pd.read_csv(ROOT / base["output_root"] / "p9" / "frontier" / "summary.csv"),
        pd.read_csv(ROOT / base["output_root"] / "p10" / "frontier" / "summary.csv"),
        pd.read_csv(ROOT / base["output_root"] / "p11" / "frontier" / "summary.csv")],
        ignore_index=True).drop_duplicates(subset=["variant"], keep="last")

    ids = sorted(per.image_id.unique())
    linked, scores, abmap = {}, {i: {} for i in ids}, {i: {} for i in ids}
    for image_id in ids:
        sample = dataset[image_id]
        files = {}
        sub = out / "src" / f"{image_id}{sample.image_path.suffix}"
        sub.parent.mkdir(parents=True, exist_ok=True)
        if not sub.exists():
            os.symlink(sample.image_path.resolve(), sub)
        files["src"] = f"src/{image_id}{sample.image_path.suffix}"
        for key, _lab, subdir, _mk in COLUMNS[1:]:
            p = ROOT / "cache" / "images" / subdir / f"{image_id}.png"
            files[key] = None
            if p.exists():
                d = out / key
                d.mkdir(parents=True, exist_ok=True)
                dst = d / f"{image_id}.png"
                if not dst.exists():
                    os.symlink(p.resolve(), dst)
                files[key] = f"{key}/{image_id}.png"
        linked[image_id] = files
    for _, r in per.iterrows():
        scores[r["image_id"]][r["variant"]] = (float(r["psnr_unedit"]),
                                               float(r["struct_dist"]),
                                               float(r["clip_target_edit"]))
    for _, r in ab.iterrows():
        abmap[r["image_id"]][r["pair"]] = float(r["s"])

    label = {c[0]: c[1] for c in COLUMNS}
    trows = []
    for key in ["off_k0", "erchord_mu1", "budedit", "precise_a1", "scadA_a1", "scad_cleanbg"]:
        r = summary[summary.variant == key]
        if r.empty:
            continue
        r = r.iloc[0]
        trows.append(f"<tr class='{'hl' if key == 'off_k0' else ''}'><td>{label[key]}</td>"
                     f"<td>{r.psnr_unedit:.2f}</td><td>{r.lpips_unedit:.3f}</td>"
                     f"<td>{r.struct_dist:.4f}</td><td>{r.clip_target_edit:.2f}</td></tr>")
    summary_html = ("<table class='summary'><tr><th>方法</th><th>背景 PSNR↑</th><th>LPIPS↓</th>"
                    "<th>全图 DINO↓</th><th>CLIP-Edited↑</th></tr>" + "".join(trows) + "</table>")

    head = "".join(f"<div>{c[1]}</div>" for c in COLUMNS)
    body = []
    for image_id in ids:
        sample = dataset[image_id]
        abi = abmap[image_id]
        ab_txt = "　".join(
            f"{t} <b class='{'pos' if v > 0 else 'neg'}'>{v:+.2f}</b>"
            for t, v in (("A/B mask k=0", abi.get("m075_k0_vs_off_k0")),
                         ("A/B mask k=1", abi.get("m075_k10_vs_off_k0"))) if v is not None)
        cells = []
        for key in KEYS:
            f = linked[image_id].get(key)
            sc = scores[image_id].get(key)
            if f is None:
                cells.append("<div class='cell'><div class='m'>无数据</div></div>")
                continue
            if sc is None:
                mt = "<div class='m'>原图</div>"
            else:
                mt = (f"<div class='m'><span>PSNR {sc[0]:.1f}</span>"
                      f"<span>DINO {sc[1]:.3f}</span><span>C-E {sc[2]:.1f}</span></div>")
            cells.append(f"<div class='cell'><img loading='lazy' src='{f}' alt='{key}'>{mt}</div>")
        body.append(f"""
<div class="row">
  <div class="meta">
    <div class="l1"><span class="mid">{image_id}</span><span class="chip">类别 {sample.category}</span>
      <span class="ab">{ab_txt}</span></div>
    <div class="pr">
      <div><i>编辑指令</i>{html.escape(sample.instruction or '—')}</div>
      <div><i>源图提示词</i>{html.escape(sample.src_prompt)}</div>
      <div><i>目标提示词</i>{html.escape(sample.tar_prompt)}</div>
    </div>
  </div>
  <div class="cells" style="grid-template-columns:repeat({len(COLUMNS)},{args.width}px);width:{args.width * len(COLUMNS) + len(COLUMNS) - 1}px">
    {''.join(cells)}
  </div>
</div>""")

    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>P9 各方法横向对比</title><style>{CSS}</style></head><body>
<header>
  <h1>P9 各方法横向对比（官方 ChordEdit 底座，200 图）</h1>
  <div class="sub">每行 = 一张源图，每列 = 一种方法；格内下方：背景 PSNR / 全图 DINO / CLIP-Edited。
    行首 A/B 为正表示配对比较中<b>我们被偏好</b>（相对 ChordEdit）。</div>
  {summary_html}
</header>
<main>
  <div class="grid" style="grid-template-columns:repeat({len(COLUMNS)},{args.width}px);width:{args.width * len(COLUMNS) + len(COLUMNS) - 1}px">{head}</div>
  {''.join(body)}
</main></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    n_imgs = sum(1 for v in linked.values() for f in v.values() if f)
    print(f"wrote {out}/index.html | rows={len(ids)} | images={n_imgs} | {len(page)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
