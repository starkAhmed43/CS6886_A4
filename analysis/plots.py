"""Generate report figures for CS6886 Assignment 4.

Reads the deterministic roofline CSVs produced by sparten.c runs and
renders three vector PDFs: roofline.pdf, tile_sweep.pdf, bw_sweep.pdf.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

PEAK = 32.0  # ops/cycle
DEFAULT_BW = 16.0  # bytes/cycle

INNER_COLOR = "#2a78d6"
OUTER_COLOR = "#eb6834"
TEXT_COLOR = "#333333"
GRID_COLOR = "#dddddd"

LAYERS = ["conv1", "conv2", "fc1", "fc2"]
DATAFLOWS = ["inner", "outer"]

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rows_by_layer_dataflow(rows):
    out = {}
    for r in rows:
        out[(r["layer"], r["dataflow"])] = r
    return out


# ---------------------------------------------------------------------------
# Figure 1: roofline (E1)
# ---------------------------------------------------------------------------
def fig_roofline():
    rows = read_csv(os.path.join(RESULTS, "E1", "roofline.csv"))
    data = rows_by_layer_dataflow(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ridge = PEAK / DEFAULT_BW  # ops/byte
    x_lo, x_hi = 0.05, 10.0
    y_lo, y_hi = 0.5, 50.0

    # memory roof
    mem_x = [x_lo, ridge]
    mem_y = [DEFAULT_BW * x for x in mem_x]
    ax.plot(mem_x, mem_y, color="#888888", linewidth=1.3, zorder=1)
    # compute roof
    ax.plot([ridge, x_hi], [PEAK, PEAK], color="#888888", linewidth=1.3, zorder=1)
    # ridge line
    ax.axvline(ridge, color="#bbbbbb", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(ridge * 1.07, y_lo * 1.3, f"ridge = {ridge:.2g} ops/byte",
             fontsize=7.5, color="#777777", rotation=90, va="bottom", ha="left")

    table_rows = []
    # Outer labels sit to the left of their point, inner labels to the right
    # -- each with a short, non-crossing leader line, so the dense cluster
    # around x=0.25-0.32 stays unambiguous. Most offsets are relative
    # (points); conv1-outer is pinned to an absolute data position on the
    # left side, away from conv1-inner's label, to avoid long parallel
    # leaders that are hard to tell apart.
    label_layout = {
        ("conv1", "inner"): dict(xytext=(34, -10), textcoords="offset points", ha="left", va="center"),
        ("conv2", "inner"): dict(xytext=(20, -12), textcoords="offset points", ha="left", va="top"),
        ("fc1", "inner"): dict(xytext=(20, -10), textcoords="offset points", ha="left", va="top"),
        ("fc2", "inner"): dict(xytext=(20, -10), textcoords="offset points", ha="left", va="top"),
        ("conv1", "outer"): dict(xytext=(0.135, 3.3), textcoords="data", ha="right", va="center"),
        ("conv2", "outer"): dict(xytext=(-45, -4), textcoords="offset points", ha="right", va="center"),
        ("fc1", "outer"): dict(xytext=(-25, -18), textcoords="offset points", ha="right", va="top"),
        ("fc2", "outer"): dict(xytext=(28, -18), textcoords="offset points", ha="left", va="top"),
    }

    for dataflow, color, marker in (("inner", INNER_COLOR, "o"), ("outer", OUTER_COLOR, "s")):
        xs, ys = [], []
        for layer in LAYERS:
            r = data[(layer, dataflow)]
            x, y = float(r["intensity"]), float(r["achieved"])
            xs.append(x)
            ys.append(y)
            table_rows.append((layer, dataflow, x, y, float(r["attainable"]), r["bound"]))
            layout = label_layout[(layer, dataflow)]
            ax.annotate(layer, xy=(x, y), xycoords="data",
                        xytext=layout["xytext"], textcoords=layout["textcoords"],
                        fontsize=7.5, color=TEXT_COLOR, ha=layout["ha"], va=layout["va"],
                        arrowprops=dict(arrowstyle="-", color="#999999",
                                        linewidth=0.6, shrinkA=0, shrinkB=3),
                        zorder=4)
        ax.scatter(xs, ys, color=color, marker=marker, s=38, edgecolor="#222222",
                   linewidth=0.6, label=dataflow, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("operational intensity (ops/byte)")
    ax.set_ylabel("performance (ops/cycle)")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, table_rows


# ---------------------------------------------------------------------------
# Figure 2: tile sweep (E2)
# ---------------------------------------------------------------------------
TILE_SHAPES = ["1x16", "2x8", "4x4", "8x2", "16x1"]


def fig_tile_sweep():
    data = {}
    for shape in TILE_SHAPES:
        rows = read_csv(os.path.join(RESULTS, "E2", f"tile_{shape}", "roofline.csv"))
        data[shape] = rows_by_layer_dataflow(rows)

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.8), sharex=True)

    table_rows = []
    lines_for_legend = {}
    for ax, layer in zip(axes, LAYERS):
        default_idx = TILE_SHAPES.index("4x4")
        ax.axvspan(default_idx - 0.5, default_idx + 0.5, color="#f0f0f0", zorder=0)
        for dataflow, color, marker, ls in (
            ("inner", INNER_COLOR, "o", "-"),
            ("outer", OUTER_COLOR, "s", "--"),
        ):
            ys = [float(data[shape][(layer, dataflow)]["cycles_per_image"]) for shape in TILE_SHAPES]
            for shape, y in zip(TILE_SHAPES, ys):
                table_rows.append((layer, dataflow, shape, y))
            line, = ax.plot(range(len(TILE_SHAPES)), ys, color=color, marker=marker,
                             linestyle=ls, markersize=5, linewidth=1.3,
                             markeredgecolor="#222222", markeredgewidth=0.5)
            lines_for_legend[dataflow] = line
        ax.set_xticks(range(len(TILE_SHAPES)))
        ax.set_xticklabels(TILE_SHAPES, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(bottom=0)
        ax.set_title(layer, fontsize=9, pad=3)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.margins(x=0.08)

    axes[0].set_ylabel("cycles per image")
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.legend(lines_for_legend.values(), lines_for_legend.keys(), loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.text(0.5, 0.01, "tile shape (TILE_ROWS × TILE_COLS)", ha="center", va="bottom",
              fontsize=9, color=TEXT_COLOR)
    return fig, table_rows


# ---------------------------------------------------------------------------
# Figure 3: bandwidth sweep (E3)
# ---------------------------------------------------------------------------
BW_VALUES = [4, 8, 16, 32, 64]


def fig_bw_sweep():
    data = {}
    for bw in BW_VALUES:
        rows = read_csv(os.path.join(RESULTS, "E3", f"bw_{bw}", "roofline.csv"))
        data[bw] = rows_by_layer_dataflow(rows)

    fig, ax = plt.subplots(figsize=(6.5, 2.9))

    table_rows = []
    plateau_y = {}
    for dataflow, color, marker, ls in (
        ("inner", INNER_COLOR, "o", "-"),
        ("outer", OUTER_COLOR, "s", "--"),
    ):
        ys = [float(data[bw][("total", dataflow)]["cycles_per_image"]) for bw in BW_VALUES]
        for bw, y in zip(BW_VALUES, ys):
            table_rows.append((dataflow, bw, y))
        ax.plot(BW_VALUES, ys, color=color, marker=marker, linestyle=ls, markersize=6,
                 linewidth=1.3, markeredgecolor="#222222", markeredgewidth=0.6, label=dataflow)
        plateau_y[dataflow] = ys

    # Point directly at the bandwidth where each curve stops falling.
    plateau_bw = {"inner": 16, "outer": 32}
    plateau_text = {
        "inner": ("inner: flat from 16 B/cycle", (0, -28), "top"),
        "outer": ("outer: flat from 32 B/cycle", (0, 28), "bottom"),
    }
    for dataflow in ("inner", "outer"):
        bw = plateau_bw[dataflow]
        y = plateau_y[dataflow][BW_VALUES.index(bw)]
        text, (dx, dy), va = plateau_text[dataflow]
        ax.annotate(text, xy=(bw, y), xycoords="data", xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color="#777777",
                    ha="center", va=va,
                    arrowprops=dict(arrowstyle="-", color="#999999",
                                    linewidth=0.6, shrinkA=0, shrinkB=4))

    ax.set_xscale("log", base=2)
    ax.set_xticks(BW_VALUES)
    ax.set_xlim(3.3, 110)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("bandwidth (bytes/cycle)")
    ax.set_ylabel("total cycles per image")
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, table_rows


def print_table(name, rows, header):
    print(f"\n=== {name} ===")
    widths = [max(len(str(h)), max((len(f"{r[i]:.4g}" if isinstance(r[i], float) else str(r[i])) for r in rows), default=0)) for i, h in enumerate(header)]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in r]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))


def main():
    fig1, t1 = fig_roofline()
    fig1.savefig(os.path.join(FIGDIR, "roofline.pdf"))

    fig2, t2 = fig_tile_sweep()
    fig2.savefig(os.path.join(FIGDIR, "tile_sweep.pdf"))

    fig3, t3 = fig_bw_sweep()
    fig3.savefig(os.path.join(FIGDIR, "bw_sweep.pdf"))

    scratch_png = os.environ.get("FIGS_PNG_DIR")
    if scratch_png:
        os.makedirs(scratch_png, exist_ok=True)
        fig1.savefig(os.path.join(scratch_png, "roofline.png"), dpi=200)
        fig2.savefig(os.path.join(scratch_png, "tile_sweep.png"), dpi=200)
        fig3.savefig(os.path.join(scratch_png, "bw_sweep.png"), dpi=200)

    print_table("roofline (E1)", t1, ["layer", "dataflow", "intensity", "achieved", "attainable", "bound"])
    print_table("tile sweep (E2)", t2, ["layer", "dataflow", "tile_shape", "cycles_per_image"])
    print_table("bw sweep (E3)", t3, ["dataflow", "bw_bytes_per_cycle", "total_cycles_per_image"])


if __name__ == "__main__":
    main()
