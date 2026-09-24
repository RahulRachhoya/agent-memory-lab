"""README charts from docs/results.json, in a light and a dark variant for GitHub's two themes.

    uv run python -m agent_memory_lab.charts

Style: Qdrant in one hue, Postgres in a second, direct labels, recessive grid.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

RESULTS = Path("docs/results.json")
CHART_DIR = Path("docs/charts")

THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "qdrant": "#2a78d6", "postgres": "#c2620a"},
    "dark": {"surface": "#0d1117", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "qdrant": "#3987e5", "postgres": "#d9731a"},
}


def store(name: str) -> str:
    return "qdrant" if name.startswith("qdrant") else "postgres"


def style(ax, t, grid_axis):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, t, title, subtitle):
    fig.text(0.02, 0.96, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.895, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def legend(fig, t, y=0.83):
    x = 0.02
    for key, label in (("qdrant", "Qdrant"), ("postgres", "Postgres + pgvector")):
        fig.patches.append(Rectangle((x, y - 0.012), 0.014, 0.028, transform=fig.transFigure,
                                     facecolor=t[key], linewidth=0))
        fig.text(x + 0.02, y, label, color=t["ink2"], fontsize=9, va="center")
        x += 0.03 + 0.0105 * len(label)


def hbar(ax, y, width, height, color, radius_px):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.bbox.width
    y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height
    r = radius_px * x_per_px
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=y_per_px / x_per_px, facecolor=color, linewidth=0))
    ax.add_patch(Rectangle((0, y - height / 2), min(width, 2 * r), height, facecolor=color, linewidth=0))


def chart_quality(res, t, path):
    rows = sorted(res, key=lambda n: res[n]["ndcg@10"])
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.74, bottom=0.12)
    style(ax, t, "x")
    top = max(r["ndcg@10"] for r in res.values())
    xmax = 0.1 * (int(top * 10) + 1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.canvas.draw()
    bar_h = 18 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.bbox.height
    for i, name in enumerate(rows):
        v = res[name]["ndcg@10"]
        hbar(ax, i, v, bar_h, t[store(name)], 4)
        ax.text(v + xmax * 0.012, i, f"{v:.3f}", va="center", color=t["ink"], fontsize=9.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.5)
    titles(fig, t, "Retrieval quality on BEIR SciFact",
           "nDCG@10 over 300 test claims, 5,183 abstracts. Same embeddings in both stores. Higher is better.")
    legend(fig, t)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


# Label placement (points offset, alignment, leader line) for the crowded top-left cluster.
LABELS = {
    "pgvector dense": (-9, 0, "right", False),
    "qdrant bm25": (-9, 0, "right", False),
    "qdrant dense": (22, -24, "left", True),
}


def chart_latency(res, t, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=t["surface"])
    fig.subplots_adjust(left=0.09, right=0.96, top=0.74, bottom=0.2)
    style(ax, t, "both")
    ax.set_xscale("log")
    for name, r in res.items():
        x, y = r["p50_ms"], r["ndcg@10"]
        ax.plot([x, r["p95_ms"]], [y, y], color=t[store(name)], linewidth=2, alpha=0.45, zorder=2,
                solid_capstyle="round")
        ax.scatter([x], [y], s=80, color=t[store(name)], edgecolors=t["surface"], linewidths=2, zorder=3)
        dx, dy, ha, leader = LABELS.get(name, (0, 10, "center", False))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center" if ha != "center" else "bottom", fontsize=8.5, color=t["ink"],
                    arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.8,
                                "shrinkA": 2, "shrinkB": 6} if leader else None)
    xs = [r["p50_ms"] for r in res.values()] + [r["p95_ms"] for r in res.values()]
    ax.set_xlim(min(xs) / 3.5, max(xs) * 1.6)  # room for the right-aligned labels on the left
    ys = [r["ndcg@10"] for r in res.values()]
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.set_xlabel("Latency per query, log scale (dot = p50, line extends to p95)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("nDCG@10", color=t["ink2"], fontsize=9)
    titles(fig, t, "Quality vs latency: top-left is best",
           "Database time only: query embeddings are precomputed. The reranker runs on CPU.")
    legend(fig, t)
    fig.text(0.02, 0.03, "Single laptop, all services in Docker, sequential queries, 10 warm-up queries excluded.",
             color=t["muted"], fontsize=8)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main():
    # Charts show one row per method; exact-search and REST reference rows stay in the table.
    res = {k: v for k, v in json.loads(RESULTS.read_text())["results"].items()
           if "(exact)" not in k and "(REST)" not in k}
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    for name, fn in [("quality", chart_quality), ("latency", chart_latency)]:
        for theme, t in THEMES.items():
            fn(res, t, CHART_DIR / f"{name}-{theme}.png")
    print(f"wrote charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
