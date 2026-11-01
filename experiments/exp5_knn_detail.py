import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from src.datasets.synthetic import flip_labels, make_linear, train_val_test_split
from src.eval.metrics import accuracy
from src.eval.plotting import C_MISS, C_NEG, C_POS, R_NEG, R_POS, setup_style
from src.models.knn import KNN
from src.run import CSV_DIR, FIG_DIR, save_csv

# 1.1 实例级邻居
Q = np.zeros(2)  # 待分类查询点（真实 y = +1，黑色五角星）

rng = np.random.default_rng(42)

def ring(n, r_lo, r_hi):
    r = rng.uniform(r_lo, r_hi, n)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

# 刻意构造的局部邻域（按到查询点的距离从近到远）：
X_NB = np.vstack(
    [
        ring(2, 0.40, 0.55),
        ring(3, 1.30, 1.70),
        ring(5, 2.00, 3.00),
        ring(1, 2.20, 2.90),
        ring(16, 3.80, 5.20),
        ring(4, 4.00, 5.40),
    ]
)
Y_NB = np.array([1] * 2 + [-1] * 3 + [1] * 5 + [-1] * 1 + [1] * 16 + [-1] * 4)

def neighbor_figure(out_path):
    """k = 5 时均匀投票与距离加权投票的对照（各一格）。"""
    setup_style()
    k = 5
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.75), constrained_layout=True)
    for ax, weights in zip(axes, ("uniform", "distance")):
        d = np.sqrt(((X_NB - Q) ** 2).sum(axis=1))
        r_k = np.sort(d)[k - 1]
        inside = d <= r_k
        pred = KNN(k=k, weights=weights).fit(X_NB, Y_NB).predict(Q[None])[0]
        for val, c in ((1, C_POS), (-1, C_NEG)):
            m = Y_NB == val
            ax.scatter(*(X_NB[m & ~inside]).T, c=c, s=8, alpha=0.30, linewidths=0.15, edgecolors="white")
            ax.scatter(*(X_NB[m & inside]).T, c=c, s=13, alpha=0.95, linewidths=0.25, edgecolors="white")
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        ax.plot(r_k * np.cos(t), r_k * np.sin(t), ls="--", color="k", lw=1.1)
        ax.scatter(*Q, marker="*", s=90, c=C_MISS, zorder=5)
        vote = "均匀投票" if weights == "uniform" else "距离加权"
        outcome = "正确" if pred == 1 else "误分类"
        ax.set_title(f"{vote}，{outcome}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markersize=5, label="y = +1"),
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markersize=5, label="y = −1"),
        Line2D([], [], marker="*", ls="none", color=C_MISS, markersize=9, label="查询点（真实 +1）"),
        Line2D([], [], color="k", ls="--", lw=1.1, label="至第 k 近邻的距离"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, columnspacing=1.2, handlelength=1.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

neighbor_figure(FIG_DIR / "exp5_knn_neighbors.png")

# 1.2 k 扫描，k = 1 / 11 / 21 的决策区域
X, y = make_linear(900, mu=2.0, sigma=1.5, seed=42)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=42)
y_fit = flip_labels(y_train, eta=0.08, seed=42)

def knn_grid_predict(model, G, chunk=8192):
    return np.concatenate([model.predict(G[i : i + chunk]) for i in range(0, len(G), chunk)])

span = X.max(axis=0) - X.min(axis=0)
lims = np.stack([X.min(axis=0) - 0.12 * span, X.max(axis=0) + 0.12 * span], axis=1)
xs = np.linspace(*lims[0], 220)
ys = np.linspace(*lims[1], 220)
xx, yy = np.meshgrid(xs, ys)
G = np.stack([xx.ravel(), yy.ravel()], axis=1)

setup_style()
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35), constrained_layout=True)
records = []
for ax, k in zip(axes, (1, 11, 21)):
    model = KNN(k=k).fit(X_train, y_fit)
    Z = knn_grid_predict(model, G).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1.5, 0.0, 1.5], colors=[R_NEG, R_POS], zorder=0)
    for val, c in ((1, C_POS), (-1, C_NEG)):
        m = y_fit == val
        ax.scatter(X_train[m, 0], X_train[m, 1], c=c, s=5, alpha=0.85, linewidths=0.1, edgecolors="white")
    records.append(
        {
            "k": k,
            "train_accuracy": accuracy(y_fit, model.predict(X_train)),
            "val_accuracy": accuracy(y_val, model.predict(X_val)),
            "test_accuracy": accuracy(y_test, model.predict(X_test)),
        }
    )
    ax.set_title(f"k = {k}")
    ax.set_xlabel("$x_1$")
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_aspect("equal", adjustable="box")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("$x_2$")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

handles = [
    Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markersize=5, label="训练点 y = +1"),
    Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markersize=5, label="训练点 y = −1"),
    Patch(facecolor=R_POS, label="预测 +1 区域"),
    Patch(facecolor=R_NEG, label="预测 −1 区域"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, columnspacing=1.2, handlelength=1.6)
fig.savefig(FIG_DIR / "exp5_knn_ksweep.png", dpi=300, bbox_inches="tight")
plt.close(fig)

save_csv(records, CSV_DIR / "exp5_knn_detail.csv")
for r in records:
    print(f"k={r['k']:<3d} train={r['train_accuracy']:.3f} val={r['val_accuracy']:.3f} test={r['test_accuracy']:.3f}")
