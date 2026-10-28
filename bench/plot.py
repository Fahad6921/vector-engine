"""Plot recall@10 vs QPS from bench/results/<dataset>.json.

Usage:
    python bench/plot.py sift-128-euclidean glove-100-angular
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
STYLE = {
    "vecengine": {"color": "#d1495b", "marker": "o", "linewidth": 2.2},
    "hnswlib": {"color": "#00798c", "marker": "s"},
    "faiss-hnsw": {"color": "#edae49", "marker": "^"},
}


def plot(name: str) -> None:
    data = json.loads((RESULTS / f"{name}.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lib, res in data["results"].items():
        pts = sorted(res["points"], key=lambda p: p["recall"])
        ax.plot(
            [p["recall"] for p in pts],
            [p["qps"] for p in pts],
            label=lib,
            markersize=4,
            **STYLE.get(lib, {}),
        )
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (1 thread)")
    ax.set_title(f"{name}, M={data['m']}, efConstruction={data['ef_construction']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS / f"{name}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        plot(name)
