from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# 配色
C_POS = "#3572C6"      # label = +1 的点
C_NEG = "#C94F4F"      # label = -1 的点
R_POS = "#D8E6F8"      # 预测 +1 的决策区域底色
R_NEG = "#F9DEDE"      # 预测 -1 的决策区域底色
C_EDGE = "#1F2937"     # 模型决策边界
C_TRUE = "#4B5563"     # 真实（Bayes）边界
C_MISS = "#111827"     # 误分类标记

_LATIN_CANDIDATES = ("Times New Roman",)
_CJK_CANDIDATES = ("Songti SC", "SimSun", "STSong")

def setup_style():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    latin = [f for f in _LATIN_CANDIDATES if f in avail]
    cjk = [f for f in _CJK_CANDIDATES if f in avail]
    rcParams["font.family"] = latin + cjk + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["mathtext.fontset"] = "stix"  # 与 Times 风格一致的数学字体
    rcParams.update(
        {
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.edgecolor": "#8A8F98",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": "#444950",
            "ytick.color": "#444950",
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _make_lifter(X_ref, dims):
    """返回 lift(G2)：把展示平面上的点补全为完整维度的向量。

    二维数据直接恒等映射；高维数据把未展示的维度固定为参考集均值（切片）。
    """
    X_ref = np.asarray(X_ref, dtype=float)
    if X_ref.shape[1] == 2:
        return lambda G: np.asarray(G, dtype=float)
    base = X_ref.mean(axis=0)
    i, j = dims

    def lift(G):
        G = np.asarray(G, dtype=float)
        F = np.broadcast_to(base, (len(G), len(base))).copy()
        F[:, i] = G[:, 0]
        F[:, j] = G[:, 1]
        return F

    return lift

def _predict_grid(model, G2, lift, chunk=2048):
    out = np.empty(len(G2), dtype=int)
    for i in range(0, len(G2), chunk):
        out[i : i + chunk] = model.predict(lift(G2[i : i + chunk]))
    return out

def _grid(X_disp, n_grid=240, pad=0.12):
    lo = X_disp.min(axis=0)
    hi = X_disp.max(axis=0)
    span = hi - lo
    lims = np.stack([lo - pad * span, hi + pad * span], axis=1)
    xs = np.linspace(lims[0, 0], lims[0, 1], n_grid)
    ys = np.linspace(lims[1, 0], lims[1, 1], n_grid)
    xx, yy = np.meshgrid(xs, ys)
    G2 = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return xx, yy, G2, lims

def plot_test_comparison(
    X_test,
    y_true,
    y_pred,
    model=None,
    out_path="figure.png",
    dims=(0, 1),
    feature_names=("$x_1$", "$x_2$"),
    true_score=None,
    boundary_label="模型决策边界",
    ref=None,
    n_grid=240,
    equal_aspect=True,
    y_display=None,
    true_title="真实标签",
):
    """生成左右对比图并保存到 out_path，返回路径。

    X_test      : 完整维度测试特征（n, d），展示时只取 dims 两列
    y_true      : 真实标签（±1），用于计算误分类
    y_pred      : 模型预测标签（±1）
    model       : 已训练模型，用于网格预测得到决策区域与边界
    true_score  : callable(G2)->score，其 0 等值线即真实（Bayes）边界，
                  仅在左子图以虚线绘制
    y_display   : 左子图染色所用标签，默认与 y_true 相同；
                  如需展示加噪后的观测标签可另行传入（不影响误分类计算）
    """
    setup_style()
    X_test = np.asarray(X_test, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_disp = y_true if y_display is None else np.asarray(y_display)
    if ref is None:
        ref = X_test
    i, j = dims
    P = X_test[:, [i, j]]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), constrained_layout=True)
    ax_true, ax_pred = axes

    xx, yy, G2, lims = _grid(P, n_grid=n_grid)
    for ax in axes:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- 左子图：真实标签 ----
    m = y_disp == 1
    ax_true.scatter(P[m, 0], P[m, 1], c=C_POS, s=16, alpha=0.85, linewidths=0.3, edgecolors="white")
    ax_true.scatter(P[~m, 0], P[~m, 1], c=C_NEG, s=16, alpha=0.85, linewidths=0.3, edgecolors="white")
    if true_score is not None:
        S = true_score(G2).reshape(xx.shape)
        ax_true.contour(xx, yy, S, levels=[0.0], colors=[C_TRUE], linewidths=1.8, linestyles="dashed")
    ax_true.set_title(f"测试集：{true_title}（n = {len(y_true)}）")

    # ---- 右子图：模型预测 + 决策区域/边界 ----
    if model is not None:
        lift = _make_lifter(ref, dims)
        Z = _predict_grid(model, G2, lift).reshape(xx.shape)
        ax_pred.contourf(xx, yy, Z, levels=[-1.5, 0.0, 1.5], colors=[R_NEG, R_POS], zorder=0)
        if (Z == 1).any() and (Z == -1).any():
            ax_pred.contour(xx, yy, Z, levels=[0.0], colors=[C_EDGE], linewidths=2.0, zorder=1.5)

    m = y_pred == 1
    ax_pred.scatter(P[m, 0], P[m, 1], c=C_POS, s=16, alpha=0.9, linewidths=0.3, edgecolors="white", zorder=3)
    ax_pred.scatter(P[~m, 0], P[~m, 1], c=C_NEG, s=16, alpha=0.9, linewidths=0.3, edgecolors="white", zorder=3)

    mis = y_pred != y_true
    if mis.sum() > 0:
        ax_pred.scatter(P[mis, 0], P[mis, 1], marker="x", c=C_MISS, s=46, linewidths=1.5, zorder=4)

    ax_pred.set_title("测试集：模型预测")

    # ---- 图例：统一放整张图下方（图外） ----
    handles = [
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markeredgecolor="#B9C6D9", markersize=7, label="y = +1"),
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markeredgecolor="#DDB3B3", markersize=7, label="y = −1"),
        Patch(facecolor=R_POS, label="预测 +1 区域"),
        Patch(facecolor=R_NEG, label="预测 −1 区域"),
        Line2D([], [], color=C_TRUE, lw=1.8, ls="--", label="真实（Bayes）边界"),
        Line2D([], [], color=C_EDGE, lw=2.0, label=boundary_label),
    ]
    if mis.sum() > 0:
        handles.append(Line2D([], [], marker="x", ls="none", color=C_MISS, markersize=8, label="误分类测试点"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, columnspacing=1.6, handlelength=1.8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
