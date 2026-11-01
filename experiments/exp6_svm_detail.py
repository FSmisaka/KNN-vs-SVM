import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from src.datasets.synthetic import make_linear, train_val_test_split
from src.eval.metrics import accuracy
from src.eval.plotting import C_EDGE, C_MISS, C_NEG, C_POS, C_TRUE, setup_style
from src.run import CSV_DIR, FIG_DIR, build_model, save_csv

BAND = "#E1E6EE"       # 间隔带底色
ACC_TRAIN = "#111827"  # accuracy 曲线：训练
ACC_VAL = "#4B5563"    # accuracy 曲线：验证
ACC_TEST = "#9CA3AF"   # accuracy 曲线：测试

def margin_panel(ax, model, X, y, title, arrow=False):
    w, b = model.w_, model.b_
    wnorm = np.linalg.norm(w)
    span = X.max(axis=0) - X.min(axis=0)
    lims = np.stack([X.min(axis=0) - 0.12 * span, X.max(axis=0) + 0.12 * span], axis=1)
    xs = np.linspace(*lims[0], 300)
    ys = np.linspace(*lims[1], 300)
    xx, yy = np.meshgrid(xs, ys)
    F = w[0] * xx + w[1] * yy + b

    ax.contourf(xx, yy, F, levels=[-1.0, 1.0], colors=[BAND], zorder=0)
    ax.contour(xx, yy, F, levels=[-1.0, 1.0], colors=[C_TRUE], linestyles="--", linewidths=1.1, zorder=1.5)
    ax.contour(xx, yy, F, levels=[0.0], colors=[C_EDGE], linewidths=1.6, zorder=1.6)
    for val, c in ((1, C_POS), (-1, C_NEG)):
        ax.scatter(X[y == val, 0], X[y == val, 1], c=c, s=8, alpha=0.85, linewidths=0.2, edgecolors="white", zorder=2)
    sv = y * (X @ w + b) <= 1.0
    for val, c in ((1, C_POS), (-1, C_NEG)):
        m = (y == val) & sv
        ax.scatter(X[m, 0], X[m, 1], s=62, facecolors="none", edgecolors=c, linewidths=1.1, zorder=3)
    if arrow:
        n_hat = w / wnorm
        p0 = -b * n_hat / wnorm                    # 超平面上离原点最近的点
        half = n_hat / wnorm                       # 沿法向到间隔边界的距离 1/||w||
        ax.annotate("", xy=p0 + half, xytext=p0 - half, arrowprops=dict(arrowstyle="<->", color=C_MISS, lw=1.0), zorder=4)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_aspect("equal", adjustable="box")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return int(sv.sum())

# 2.1 间隔
X_A, y_A = make_linear(160, mu=2.0, sigma=1.0, seed=42)
C_ANAT = 100.0

setup_style()
fig, ax = plt.subplots(figsize=(3.0, 2.75), constrained_layout=True)
model_A = build_model("svm", C=C_ANAT).fit(X_A, y_A)
n_sv = margin_panel(ax, model_A, X_A, y_A, "")
ax.set_title(f"C = {C_ANAT:g}，间隔宽 {2 / np.linalg.norm(model_A.w_):.2f}")
handles = [
    Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markersize=5, label="y = +1"),
    Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markersize=5, label="y = −1"),
    Line2D([], [], color=C_EDGE, lw=1.6, label="超平面 w·x + b = 0"),
    Line2D([], [], color=C_TRUE, lw=1.1, ls="--", label="间隔边界 w·x + b = ±1"),
    Patch(facecolor=BAND, label="间隔带"),
    Line2D([], [], marker="o", ls="none", color="none", markerfacecolor="none", markeredgecolor="#6B7280", markersize=6.5, label="支持向量 y·(w·x + b) ≤ 1"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, columnspacing=1.0, handlelength=1.4)
fig.savefig(FIG_DIR / "exp6_svm_margin.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2.2 C 扫描，间隔宽度 / 支持向量数 / accuracy 随 C 的变化
X, y = make_linear(1500, mu=2.0, sigma=1.5, seed=42)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=42)

C_GRID = np.logspace(-2, 6, 17)
SHOW_CS = (0.1, 10.0, 10000.0)

records = []
models = {}
for C in C_GRID:
    m = build_model("svm", C=float(C)).fit(X_train, y_train)
    models[C] = m
    wnorm = np.linalg.norm(m.w_)
    records.append(
        {
            "C": C,
            "margin_width": 2.0 / wnorm,
            "n_sv": int((y_train * (X_train @ m.w_ + m.b_) <= 1.0).sum()),
            "train_accuracy": accuracy(y_train, m.predict(X_train)),
            "val_accuracy": accuracy(y_val, m.predict(X_val)),
            "test_accuracy": accuracy(y_test, m.predict(X_test)),
        }
    )
save_csv(records, CSV_DIR / "exp6_svm_detail.csv")

setup_style()
fig, axes = plt.subplots(1, 3, figsize=(5.7, 2.15), constrained_layout=True)

# 三条指标曲线（圆点标出 C = 0.1 / 10 / 10^4 三个参考取值）
def curve_panel(ax, ys, ylabel, color):
    ax.plot(C_GRID, ys, color=color, lw=1.4)
    ax.scatter(SHOW_CS, np.interp(SHOW_CS, C_GRID, ys), color=color, s=16, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

curve_panel(axes[0], [r["margin_width"] for r in records], "间隔宽 2/||w||", "#3E5F8A")
curve_panel(axes[1], [r["n_sv"] for r in records], "支持向量数", "#3E5F8A")
curve_panel(axes[2], [r["train_accuracy"] for r in records], "准确率", ACC_TRAIN)
axes[2].plot(C_GRID, [r["val_accuracy"] for r in records], color=ACC_VAL, lw=1.4, ls="--")
axes[2].plot(C_GRID, [r["test_accuracy"] for r in records], color=ACC_TEST, lw=1.4, ls=":")
axes[1].tick_params(labelleft=False)
axes[2].tick_params(labelleft=False)

handles = [
    Line2D([], [], color=ACC_TRAIN, lw=1.4, label="训练准确率"),
    Line2D([], [], color=ACC_VAL, lw=1.4, ls="--", label="验证准确率"),
    Line2D([], [], color=ACC_TEST, lw=1.4, ls=":", label="测试准确率"),
    Line2D([], [], marker="o", ls="none", color="#3E5F8A", markersize=5, label="C = 0.1 / 10 / 10⁴ 取值点"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, columnspacing=1.0, handlelength=1.6)
fig.savefig(FIG_DIR / "exp6_svm_c_sweep.png", dpi=300, bbox_inches="tight")
plt.close(fig)

for r in records:
    print(
        f"C={r['C']:<10g} margin={r['margin_width']:.3f} SV={r['n_sv']:<4d} "
        f"train={r['train_accuracy']:.3f} val={r['val_accuracy']:.3f} test={r['test_accuracy']:.3f}"
    )
