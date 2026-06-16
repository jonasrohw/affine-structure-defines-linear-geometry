"""Visualize LayerNorm in 2D: 3x3 grid showing input vectors and their normalized versions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

def layernorm_2d(v, eps=1e-5):
    mean = np.mean(v)
    var = np.var(v)
    return (v - mean) / np.sqrt(var + eps)

vectors = [
    np.array([3.0, 1.0]),
    np.array([1.0, 3.0]),
    np.array([5.0, 4.5]),
    np.array([0.5, 0.1]),
    np.array([10.0, 2.0]),
    np.array([-1.0, -3.0]),
    np.array([2.0, -1.0]),
    np.array([-0.3, 0.3]),
    np.array([4.0, 4.0]),
]

labels = [
    "(3, 1)",
    "(1, 3)",
    "(5, 4.5) — near diagonal",
    "(0.5, 0.1) — small",
    "(10, 2) — large",
    "(-1, -3) — both negative",
    "(2, -1) — mixed sign",
    "(-0.3, 0.3) — small, symmetric",
    "(4, 4) — exactly on diagonal",
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for idx, (ax, v, label) in enumerate(zip(axes.flat, vectors, labels)):
    vn = layernorm_2d(v)

    lim = max(np.abs(v).max(), np.abs(vn).max()) * 1.4
    lim = max(lim, 1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color="steelblue", lw=2))
    ax.annotate("", xy=vn, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color="crimson", lw=2))

    ax.plot(*v, "o", color="steelblue", ms=5)
    ax.plot(*vn, "o", color="crimson", ms=5)

    ax.set_title(f"input {label}", fontsize=9)
    ax.text(0.02, 0.98, f"LN = ({vn[0]:+.2f}, {vn[1]:+.2f})",
            transform=ax.transAxes, fontsize=8, va="top", color="crimson")

axes[0, 0].annotate("input", xy=(0.5, 0.5), xytext=(0.5, 0.5),
                     fontsize=0)
fig.legend(
    handles=[
        plt.Line2D([0], [0], color="steelblue", lw=2, label="input vector"),
        plt.Line2D([0], [0], color="crimson", lw=2, label="after LayerNorm"),
    ],
    loc="upper center", ncol=2, fontsize=11, frameon=False,
)

fig.suptitle("LayerNorm in 2D: all vectors collapse to (±1, ∓1)", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).parent.parent / "results" / "layernorm_2d_visualization.pdf"
fig.savefig(out)
print(f"Saved: {out}")
plt.close()
