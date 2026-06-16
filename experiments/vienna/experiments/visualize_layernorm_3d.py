"""Visualize LayerNorm in 3D: 3x3 grid showing input vectors and their normalized versions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

def layernorm_3d(v, eps=1e-5):
    mean = np.mean(v)
    var = np.var(v)
    return (v - mean) / np.sqrt(var + eps)

vectors = [
    np.array([3.0, 1.0, 0.0]),
    np.array([1.0, 3.0, 1.0]),
    np.array([5.0, 5.0, 1.0]),
    np.array([0.2, 0.1, 0.5]),
    np.array([10.0, 2.0, 6.0]),
    np.array([-1.0, -3.0, -2.0]),
    np.array([2.0, -1.0, 0.5]),
    np.array([-2.0, 0.0, 2.0]),
    np.array([4.0, 4.0, 4.0]),
]

labels = [
    "(3, 1, 0)",
    "(1, 3, 1)",
    "(5, 5, 1) — two equal",
    "(0.2, 0.1, 0.5) — small",
    "(10, 2, 6) — large",
    "(-1, -3, -2) — all negative",
    "(2, -1, 0.5) — mixed sign",
    "(-2, 0, 2) — symmetric",
    "(4, 4, 4) — on diagonal",
]

fig = plt.figure(figsize=(14, 14))

for idx, (v, label) in enumerate(zip(vectors, labels)):
    ax = fig.add_subplot(3, 3, idx + 1, projection="3d")
    vn = layernorm_3d(v)

    lim = max(np.abs(v).max(), np.abs(vn).max()) * 1.3
    lim = max(lim, 1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color="steelblue", lw=2)
    ax.scatter(*v, color="steelblue", s=40, zorder=5)

    ax.plot([0, vn[0]], [0, vn[1]], [0, vn[2]], color="crimson", lw=2)
    ax.scatter(*vn, color="crimson", s=40, zorder=5)

    ax.set_title(f"input {label}\nLN = ({vn[0]:+.2f}, {vn[1]:+.2f}, {vn[2]:+.2f})",
                 fontsize=8)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=25, azim=135)

fig.legend(
    handles=[
        plt.Line2D([0], [0], color="steelblue", lw=2, label="input vector"),
        plt.Line2D([0], [0], color="crimson", lw=2, label="after LayerNorm"),
    ],
    loc="upper center", ncol=2, fontsize=11, frameon=False,
)

fig.suptitle("LayerNorm in 3D: vectors project onto a circle in the plane x+y+z=0",
             fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).parent.parent / "results" / "layernorm_3d_visualization.pdf"
fig.savefig(out)
print(f"Saved: {out}")
plt.close()
