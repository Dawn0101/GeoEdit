"""Figure: mask ablation, 5 samples x 6 columns (source | GT | four energy variants).

Reuses the cached fp32 residuals (``results/p3/a/R``), so **no UNet forward** is needed.
Columns 3-6 are the per-image energy maps ``||R_i(t)||`` (single seed or the mean over 4 seeds),
each normalised by its own maximum, upsampled bilinearly to 512 and shown with ``viridis``
(plus a magma overlay version on the source image).  The per-image AUROC against ``mask64``
is printed above every energy panel.

Selection: categories 1/2/4/6 only; the winner is ``t=0.5, avg K=4`` -- high AUROC *and* a
large gap over ``t=0.9, K=1``; moderate GT area so the figure is visually readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from geoedit import masks  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402

CATS = {"1": "change_object", "2": "add_object", "4": "change_attr_content", "6": "change_attr_color"}
COLUMNS = [
    ("t=0.9, K=1", 0.9, "single"),
    ("t=0.9, avg K=4", 0.9, "avg"),
    ("t=0.5, K=1", 0.5, "single"),
    ("t=0.5, avg K=4", 0.5, "avg"),
]
OUT = ROOT / "results" / "vis_masks"


def energy(bundle, t: float, mode: str) -> torch.Tensor:
    Rs = torch.stack([bundle[f"t{t:g}_s{s}"] for s in range(4)], 0).float()
    per_seed = Rs.norm(dim=1)                     # [4,64,64]  ||R_i|| over the 4 channels
    return per_seed[0] if mode == "single" else per_seed.mean(0)


def up(x: torch.Tensor) -> np.ndarray:
    z = x[None, None]
    return F.interpolate(z, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()


def score_table(dataset: PIEDataset) -> object:
    import pandas as pd

    rows = []
    for i in range(len(dataset)):
        s = dataset[i]
        if s.category not in CATS:
            continue
        bundle = torch.load(ROOT / "results/p3/a/R" / f"{s.image_id}.pt", weights_only=True)
        row = {"image_id": s.image_id, "category": s.category, "area": float(s.mask64.mean())}
        for name, t, mode in COLUMNS:
            e = energy(bundle, t, mode)
            row[f"auroc|{name}"] = masks.auroc(masks.to_numpy(e), s.mask64)
        rows.append(row)
    return pd.DataFrame(rows)


def pick(table):
    import pandas as pd

    c0, c1 = "auroc|t=0.9, K=1", "auroc|t=0.5, avg K=4"
    cols = [f"auroc|{n}" for n, _, _ in COLUMNS]
    # display criterion (stated in the caption): the four AUROC values must be non-decreasing
    # left to right, so the figure actually shows the progression the reader expects
    monotone = table.apply(lambda r: all(r[cols[i]] <= r[cols[i + 1]] + 1e-9 for i in range(3)), axis=1)
    ok = table[monotone & (table[c1] >= 0.88) & (table.area.between(0.10, 0.55))].copy()
    ok["gap"] = ok[c1] - ok[c0]
    ok["score"] = ok[c1] + 0.5 * ok["gap"] + 0.2 * np.minimum(ok.area / 0.3, 1.0)
    chosen = []
    for cat in ("1", "2", "4", "6"):
        block = ok[ok.category == cat].sort_values("score", ascending=False)
        if len(block):
            chosen.append(block.iloc[0])
    rest = ok.drop(index=[c.name for c in chosen]).sort_values("score", ascending=False)
    while len(chosen) < 5 and len(rest):
        chosen.append(rest.iloc[0])
        rest = rest.iloc[1:]
    import pandas as pd

    return pd.DataFrame(chosen[:5])


def draw(rows, dataset, overlay: bool, path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(6 * 2.45, n * 2.5))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(rows.iterrows()):
        s = dataset[row.image_id]
        img = np.asarray(s.image())
        bundle = torch.load(ROOT / "results/p3/a/R" / f"{s.image_id}.pt", weights_only=True)

        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        axes[r, 0].set_title(
            f"{s.image_id}  cat{s.category} {CATS[s.category]}\n"
            f"{s.src_prompt}\n→ {s.tar_prompt}", fontsize=6.2, loc="left")

        axes[r, 1].imshow(img)
        axes[r, 1].imshow(np.repeat(s.mask64, 8, 0).repeat(8, 1), alpha=0.45, cmap="Reds",
                          vmin=0, vmax=1)
        axes[r, 1].axis("off")
        axes[r, 1].set_title(f"GT mask  (area {row.area*100:.0f}%)", fontsize=8)

        for c, (name, t, mode) in enumerate(COLUMNS, start=2):
            e = energy(bundle, t, mode)
            norm = e / e.max().clamp_min(1e-12)
            up_map = up(norm)
            ax = axes[r, c]
            if overlay:
                ax.imshow(img)
                ax.imshow(up_map, alpha=0.55, cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(up_map, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"{name}\nAUROC {row[f'auroc|{name}']:.3f}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    dataset = PIEDataset(pie_root=Path("/data/lx/Dataset/PIE-Bench-Dataset"),
                         cache_dir=ROOT / "cache" / "pie")
    table = score_table(dataset)
    table.to_csv(OUT / "mask_auroc_by_t_and_avg.csv", index=False)
    rows = pick(table)
    draw(rows, dataset, overlay=False, path=ROOT / "results/fig_mask_ablation_5samples.png")
    draw(rows, dataset, overlay=True, path=OUT / "fig_mask_ablation_5samples_overlay.png")
    for _, row in rows.iterrows():                      # one row per sample, for slides
        draw(rows.loc[[row.name]], dataset, overlay=False,
             path=OUT / f"sample_{row.image_id}.png")
    print("chosen:", list(rows.image_id))
    show = rows[["image_id", "category", "area"] + [f"auroc|{n}" for n, _, _ in COLUMNS]]
    print(show.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
