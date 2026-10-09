"""Draw docs/figures/platform-overview.png: the two halves of the platform.

  .venv/bin/python docs/figures/make_platform_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_name("platform-overview.png")

INK, MUTED = "#1f2933", "#52606d"
ETL_FILL, ETL_EDGE = "#e3edf7", "#3b6ea5"
AD_FILL, AD_EDGE = "#fdeee0", "#c56a1b"
SIDE_FILL, SIDE_EDGE = "#eef0f2", "#8a96a3"

W, H, GAP = 1.72, 1.2, 0.38          # box width, height, horizontal gap
STEP = W + GAP


def box(ax, x, y, title, body, fill, edge, w=W, h=H):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=8.2,
            color=MUTED, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
                                 lw=1.3, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def lane(ax, y, x0, stages, fill, edge):
    """Boxes left to right with arrows between; returns each box's x."""
    xs = []
    for i, (title, body) in enumerate(stages):
        x = x0 + i * STEP
        box(ax, x, y, title, body, fill, edge)
        if i:
            arrow(ax, x - GAP, y + H / 2, x, y + H / 2, color=edge)
        xs.append(x)
    return xs


def header(ax, y, title, sub, color, x=0.0):
    ax.text(x, y, title, fontsize=13, fontweight="bold", color=color, va="top")
    ax.text(x, y - 0.36, sub, fontsize=9.5, color=MUTED, va="top")


fig, ax = plt.subplots(figsize=(17, 6.4), dpi=160)
ax.set_xlim(-0.15, 16.75)
ax.set_ylim(-0.1, 6.55)
ax.axis("off")
fig.patch.set_facecolor("white")

# -- backend ETL lane ------------------------------------------------------
Y_ETL = 3.95
header(ax, 6.5, "Backend ETL", "src/avionics_etl  ·  medallion layers as partitioned Parquet in build/", ETL_EDGE)

etl = lane(ax, Y_ETL, 0.0, [
    ("Source", "C-MAPSS\nFD001–FD004\ndata/*.txt"),
    ("Raw", "byte copy\n+ SHA-256\nper file"),
    ("Bronze", "adapter → canonical\nmodel, chunked,\nappend-only"),
    ("Validate", "channel contracts\n(configs/\nchannels.yaml)"),
    ("Silver", "regime z-score,\nfolds by asset,\noutlier flags"),
    ("Labels", "RUL per row,\nclipped at 125"),
    ("Gold", "causal rolling\nfeatures\n(5/10/20 cycles)"),
    ("Warehouse", "DuckDB star\nschema: views\nover Parquet"),
], ETL_FILL, ETL_EDGE)

# Lineage band over the lane.
yb = Y_ETL + H + 0.17
ax.plot([etl[1], etl[-1] + W], [yb, yb], color=ETL_EDGE, lw=1, ls=(0, (3, 3)))
ax.text((etl[1] + etl[-1] + W) / 2, yb + 0.06,
        "lineage on every table: _run_id · _config_hash · _stage  (manifests in build/artifacts/)",
        ha="center", va="bottom", fontsize=8.5, color=ETL_EDGE, style="italic")

# Quarantine hangs below Validate.
qw, qh = W - 0.2, 0.85
qx, qy = etl[3] + 0.1, Y_ETL - 0.4 - qh
box(ax, qx, qy, "Quarantine", "failed rows,\nnever dropped", SIDE_FILL, SIDE_EDGE, w=qw, h=qh)
arrow(ax, etl[3] + W / 2, Y_ETL, etl[3] + W / 2, qy + qh, color=SIDE_EDGE)

# -- anomaly detection lane (same column grid) -------------------------------
Y_AD = 0.0
header(ax, 2.05, "Anomaly detection", "src/avionics_ad  ·  Streamlit app (./run_demo.sh)", AD_EDGE,
       x=STEP * 2)

box(ax, etl[0], Y_AD, "Synthetic LV", "checkout telemetry\ngenerator\n(always available)", SIDE_FILL, SIDE_EDGE)
ad = lane(ax, Y_AD, etl[1], [
    ("Datasets", "lv_synthetic\nor cmapss (FD001)\n+ injected faults"),
    ("Windows", "min-max scaled\non clean data,\nwindow × channels"),
    ("Fit", "8 detectors\non clean\nwindows only"),
    ("Score", "one anomaly\nscore per\ntest window"),
    ("Evaluate", "threshold from\ntraining scores →\nrecall, F2, per kind"),
    ("Streamlit app", "configure → run →\nresults, model docs,\nproject overview"),
], AD_FILL, AD_EDGE)
arrow(ax, etl[0] + W, Y_AD + H / 2, ad[0], Y_AD + H / 2, color=SIDE_EDGE)

# Silver feeds the cmapss dataset: down, left under the quarantine box, down.
sx, dx, ye = etl[4] + W / 2, ad[0] + W / 2, qy - 0.32
ax.plot([sx, sx, dx], [Y_ETL, ye, ye], color=ETL_EDGE, lw=1.3)
arrow(ax, dx, ye, dx, Y_AD + H, color=ETL_EDGE)
ax.text(sx + 0.1, (Y_ETL + ye) / 2, "silver/measurements\n(FD001, *_z channels)",
        fontsize=8.2, color=ETL_EDGE, va="center")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print(f"wrote {OUT}")
