from pathlib import Path

import itertools

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 配色
C_POS = "#3572C6"      # label = +1 的点
C_NEG = "#C94F4F"      # label = -1 的点
R_POS = "#D8E6F8"      # 预测 +1 的决策区域底色
R_NEG = "#F9DEDE"      # 预测 -1 的决策区域底色
C_EDGE = "#1F2937"     # 模型决策边界
C_TRUE = "#4B5563"     # 真实（Bayes）边界
C_MISS = "#111827"     # 误分类标记
P_FACE = "#A8C3E8"     # 3D 超平面填充色
P_EDGE = "#3E5F8A"     # 3D 超平面描边色

SAVE_DPI = 300

_LATIN_CANDIDATES = ("Times New Roman",)
_CJK_CANDIDATES = ("Songti SC", "SimSun", "STSong")

def setup_style():
    """印刷尺寸优先的样式：图按论文中的物理宽度生成，字号与正文字号相当。"""
    avail = {f.name for f in font_manager.fontManager.ttflist}
    latin = [f for f in _LATIN_CANDIDATES if f in avail]
    cjk = [f for f in _CJK_CANDIDATES if f in avail]
    rcParams["font.family"] = latin + cjk + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["mathtext.fontset"] = "stix"  # 与 Times 风格一致的数学字体
    rcParams.update(
        {
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#8A8F98",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#444950",
            "ytick.color": "#444950",
            "legend.fontsize": 9,
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

def plot_boundary_comparison(
    X_test,
    y_true,
    models,
    out_path="figure.png",
    dims=(0, 1),
    feature_names=("$x_1$", "$x_2$"),
    true_score=None,
    figsize=(5.5, 2.55),
    n_grid=240,
    equal_aspect=True,
    extra_points=None,
    ref=None,
):
    """单实验合并图：[真实标签 | 各模型预测]，KNN 与 SVM 同图对比，保存到 out_path。

    models      : [(名称, {"pred": 预测标签, "model": 已训练模型}), ...]，通常为 KNN 与 SVM
    true_score  : callable(G2)->score，其 0 等值线即真实（Bayes）边界，仅在首格以虚线绘制
    extra_points: (X_extra, y_extra, 图例文字)，在各预测格以空心三角叠加展示
    """
    setup_style()
    X_test = np.asarray(X_test, dtype=float)
    y_true = np.asarray(y_true)
    if ref is None:
        ref = X_test
    i, j = dims
    P = X_test[:, [i, j]]
    n_panels = 1 + len(models)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)

    xx, yy, G2, lims = _grid(P, n_grid=n_grid)
    for ax in axes:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(feature_names[0])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(feature_names[1])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # ---- 首格：真实标签 ----
    m = y_true == 1
    axes[0].scatter(P[m, 0], P[m, 1], c=C_POS, s=9, alpha=0.85, linewidths=0.2, edgecolors="white")
    axes[0].scatter(P[~m, 0], P[~m, 1], c=C_NEG, s=9, alpha=0.85, linewidths=0.2, edgecolors="white")
    if true_score is not None:
        S = true_score(G2).reshape(xx.shape)
        axes[0].contour(xx, yy, S, levels=[0.0], colors=[C_TRUE], linewidths=1.4, linestyles="dashed")

    # ---- 其余格：各模型预测 + 决策区域/边界 ----
    for ax, (name, art) in zip(axes[1:], models):
        model, y_pred = art["model"], np.asarray(art["pred"])
        lift = _make_lifter(ref, dims)
        Z = _predict_grid(model, G2, lift).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1.5, 0.0, 1.5], colors=[R_NEG, R_POS], zorder=0)
        if (Z == 1).any() and (Z == -1).any():
            ax.contour(xx, yy, Z, levels=[0.0], colors=[C_EDGE], linewidths=1.6, zorder=1.5)

        m = y_pred == 1
        ax.scatter(P[m, 0], P[m, 1], c=C_POS, s=9, alpha=0.9, linewidths=0.2, edgecolors="white", zorder=3)
        ax.scatter(P[~m, 0], P[~m, 1], c=C_NEG, s=9, alpha=0.9, linewidths=0.2, edgecolors="white", zorder=3)
        mis = y_pred != y_true
        if mis.sum() > 0:
            ax.scatter(P[mis, 0], P[mis, 1], marker="x", c=C_MISS, s=26, linewidths=1.1, zorder=4)
        if extra_points is not None:
            X_extra, y_extra, _ = extra_points
            E = np.asarray(X_extra, dtype=float)[:, [dims[0], dims[1]]]
            y_extra = np.asarray(y_extra)
            me = y_extra == 1
            ax.scatter(E[me, 0], E[me, 1], marker="^", s=24, facecolors="none", edgecolors=C_POS, linewidths=1.0, zorder=2.5)
            ax.scatter(E[~me, 0], E[~me, 1], marker="^", s=24, facecolors="none", edgecolors=C_NEG, linewidths=1.0, zorder=2.5)

    # ---- 图例：整图下方 ----
    handles = [
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markeredgecolor="#B9C6D9", markersize=5, label="y = +1"),
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markeredgecolor="#DDB3B3", markersize=5, label="y = −1"),
        Line2D([], [], color=C_TRUE, lw=1.4, ls="--", label="真实边界"),
        Line2D([], [], color=C_EDGE, lw=1.6, label="模型边界"),
        Line2D([], [], marker="x", ls="none", color=C_MISS, markersize=6, label="误分类点"),
    ]
    if extra_points is not None:
        handles.append(Line2D([], [], marker="^", ls="none", color="none", markerfacecolor="none", markeredgecolor="#6B7280", markersize=6, label=extra_points[2]))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(3, len(handles)), frameon=False, columnspacing=1.4, handlelength=1.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_search_curves_combined(
    knn_records,
    svm_records,
    knn_best,
    svm_best,
    out_path,
    dataset_label="",
):
    """真实数据单数据集合并搜索曲线：左格 KNN（k 轴，uniform/distance 两种加权），
    右格 SVM（C 对数轴）。每格画训练/验证准确率，星标验证集选出的最优点。

    *_records : 网格点记录（含 k 或 C、weights、train/val_accuracy）
    *_best    : 验证集选出的最优参数组合，用于打星
    """
    setup_style()
    split_color = {"train": "#C94F4F", "val": "#3572C6"}
    series_style = {"uniform": ("-", "o"), "distance": ("--", "s")}

    fig, (ax_knn, ax_svm) = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)

    # ---- 左格：KNN，x = k ----
    xs = sorted({r["k"] for r in knn_records})
    for series in ("uniform", "distance"):
        ls, mk = series_style[series]
        for split, metric in (("train", "train_accuracy"), ("val", "val_accuracy")):
            ys = [next(r[metric] for r in knn_records if r["k"] == x and str(r["weights"]) == series) for x in xs]
            ax_knn.plot(
                xs, ys, color=split_color[split], ls=ls, marker=mk, ms=3, lw=1.3 if split == "val" else 0.9,
                markeredgewidth=0.9, markerfacecolor="white", alpha=1.0 if split == "val" else 0.55,
                label=f"{'均匀加权' if series == 'uniform' else '距离加权'}·{'训练' if split == 'train' else '验证'}",
            )
    bx = knn_best["k"]
    b_series = str(knn_best.get("weights", "uniform"))
    yv = next(r["val_accuracy"] for r in knn_records if r["k"] == bx and str(r["weights"]) == b_series)
    ax_knn.scatter([bx], [yv], marker="*", s=150, color="#F2B01E", edgecolors="#7A5C07", linewidths=0.7, zorder=5)
    ax_knn.set_xlabel("k（邻居数）")
    ax_knn.set_ylabel("准确率")
    ax_knn.set_title("KNN 的 k 扫描")

    # ---- 右格：SVM，x = C（对数） ----
    xs = sorted({r["C"] for r in svm_records})
    for split, metric in (("train", "train_accuracy"), ("val", "val_accuracy")):
        ys = [next(r[metric] for r in svm_records if r["C"] == x) for x in xs]
        ax_svm.plot(
            xs, ys, color=split_color[split], marker="o", ms=3, lw=1.3 if split == "val" else 0.9,
            markeredgewidth=0.9, markerfacecolor="white", alpha=1.0 if split == "val" else 0.55,
            label="训练" if split == "train" else "验证",
        )
    bx = svm_best["C"]
    yv = next(r["val_accuracy"] for r in svm_records if r["C"] == bx)
    ax_svm.scatter([bx], [yv], marker="*", s=150, color="#F2B01E", edgecolors="#7A5C07", linewidths=0.7, zorder=5)
    ax_svm.set_xscale("log")
    ax_svm.set_xlabel("C（对数刻度）")
    ax_svm.set_title("SVM 的 C 扫描")

    for ax in (ax_knn, ax_svm):
        ax.grid(True, ls=":", lw=0.6, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if dataset_label:
        fig.suptitle(dataset_label, fontsize=10)

    handles = [
        Line2D([], [], color="black", lw=1.3, marker="o", ms=3, markerfacecolor="white", label="均匀加权"),
        Line2D([], [], color="black", lw=1.3, ls="--", marker="s", ms=3, markerfacecolor="white", label="距离加权"),
        Line2D([], [], color=split_color["train"], lw=1.3, label="训练"),
        Line2D([], [], color=split_color["val"], lw=1.3, label="验证"),
        Line2D([], [], marker="*", ls="none", color="none", markerfacecolor="#F2B01E", markeredgecolor="#7A5C07", markersize=9, label="验证集最优点"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, columnspacing=1.4, handlelength=1.6)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path

# 3D 版本：展示三个维度，SVM 超平面绘制为平面（与展示长方体精确裁剪）

def _plane_box_polygon(W, c, lo, hi):
    """平面 {p: W·p + c = 0} 与长方体 [lo, hi]^3 的交集多边形。

    对长方体的 12 条棱求交点，去重后在平面内按极角排序；不相交返回 None。
    """
    W = np.asarray(W, dtype=float)
    pts = []
    for i in range(3):  # 棱的方向轴
        others = [ax for ax in range(3) if ax != i]
        for j, k in itertools.product((0, 1), (0, 1)):
            q0 = np.empty(3)
            q0[others[0]] = (lo, hi)[j][others[0]]
            q0[others[1]] = (lo, hi)[k][others[1]]
            q0[i] = lo[i]
            q1 = q0.copy()
            q1[i] = hi[i]
            f0, f1 = W @ q0 + c, W @ q1 + c
            if f1 == f0:
                continue
            t = -f0 / (f1 - f0)
            if 0.0 <= t <= 1.0:
                pts.append(q0 + t * (q1 - q0))
    uniq = []
    for p in pts:
        if not any(np.allclose(p, q, atol=1e-9) for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None
    P = np.array(uniq)
    n = W / np.linalg.norm(W)
    a = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    ctr = P.mean(axis=0)
    d = P - ctr
    ang = np.arctan2(d @ e2, d @ e1)
    return P[np.argsort(ang)]

def _svm_plane_in_display(model, ref, dims):
    """把线性 SVM 的超平面映射到展示三维坐标系：返回 (W, c)，平面为 W·p + c = 0。

    未展示的维度固定为参考集均值（切片），W 为 w 在展示维度上的分量，
    c = b + w 在其余维度上的贡献。
    """
    base = np.asarray(ref, dtype=float).mean(axis=0)
    w = np.asarray(model.w_, dtype=float)
    b = float(model.b_)
    W = w[list(dims)]
    c = b + float(w @ base - W @ base[list(dims)])
    return W, c

def _style_3d_axis(ax, box_aspect=(1, 1, 1)):
    # 展示框三条轴的相对长度（默认立方体）；比例尺不一定等于数据真实比例
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=18, azim=-62)
    ax.zaxis.set_rotate_label(True)  # 「维度 3」竖排，避免横向占位导致的遮挡/裁剪
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("#D5D9E0")
    ax.tick_params(colors="#444950", labelsize=8.5)

def _front_back_alpha(P, plane, a_front=0.95, a_back=0.30):
    """按点在平面的哪一侧返回逐点透明度：w·x + b > 0 一侧醒目，< 0 一侧变浅。"""
    W = np.asarray(plane[0], dtype=float)
    c = float(plane[1])
    front = P @ W + c > 0
    return np.where(front, a_front, a_back)

def _rgba(color, alphas):
    """把单色 + 逐点透明度拼成 (n, 4) RGBA 数组（scatter 逐点透明度）。"""
    alphas = np.asarray(alphas, dtype=float)
    rgb = np.tile(np.array(to_rgb(color)), (len(alphas), 1))
    return np.column_stack([rgb, alphas])

def plot_boundary_comparison_3d(
    X_test,
    y_true,
    models,
    out_path="figure_3d.png",
    dims=(0, 1, 2),
    feature_names=("$x_1$", "$x_2$", "$x_3$"),
    true_plane=None,
    boundary_label="SVM 超平面切片",
    ref=None,
    pad=0.12,
    box_aspect=(1, 1, 1),
    figsize=(5.5, 3.25),
):
    """3D 合并图：[真实标签 | 各模型预测]，SVM 面板叠加超平面切片。

    models     : [(名称, {"pred": 预测标签, "model": 已训练模型}), ...]
    true_plane : (W, c)，展示坐标系中真实（Bayes）边界的平面方程 W·p + c = 0，
                 以虚线线框绘制在首格；若模型具有 w_/b_（线性 SVM），其超平面切片
                 绘制在该模型的预测格
    box_aspect : 展示框三条轴（x, y, z）的相对长度，默认立方体
    """
    setup_style()
    X_test = np.asarray(X_test, dtype=float)
    y_true = np.asarray(y_true)
    if ref is None:
        ref = X_test
    P = X_test[:, list(dims)]
    n_panels = 1 + len(models)

    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(1, n_panels, i + 1, projection="3d") for i in range(n_panels)]

    span = P.max(axis=0) - P.min(axis=0)
    lo = P.min(axis=0) - pad * span
    hi = P.max(axis=0) + pad * span
    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel(feature_names[0], labelpad=3)
        ax.set_ylabel(feature_names[1], labelpad=7)
        ax.set_zlabel(feature_names[2], labelpad=7)
        _style_3d_axis(ax, box_aspect)

    # 按 w·x+b 的符号分层透明度：> 0 一侧的点不透明醒目，< 0 一侧的点变浅
    alpha_true = _front_back_alpha(P, true_plane) if true_plane is not None else np.full(len(P), 0.85)

    # ---- 首格：真实标签 + 真实边界平面（虚线线框） ----
    m = y_true == 1
    axes[0].scatter(*P[m].T, c=_rgba(C_POS, alpha_true[m]), s=7, depthshade=False, linewidths=0.15, edgecolors="white")
    axes[0].scatter(*P[~m].T, c=_rgba(C_NEG, alpha_true[~m]), s=7, depthshade=False, linewidths=0.15, edgecolors="white")
    if true_plane is not None:
        verts = _plane_box_polygon(true_plane[0], true_plane[1], lo, hi)
        if verts is not None:
            coll = Poly3DCollection([verts], facecolors="none", edgecolors=C_TRUE, linewidths=1.0, alpha=0.9)
            coll.set_linestyle((0, (4, 3)))
            axes[0].add_collection3d(coll)

    # ---- 其余格：各模型预测（SVM 追加超平面切片） ----
    has_plane = False
    for ax, (name, art) in zip(axes[1:], models):
        model, y_pred = art["model"], np.asarray(art["pred"])
        alpha_pred = np.full(len(P), 0.9)
        if hasattr(model, "w_"):
            W, c = _svm_plane_in_display(model, ref, dims)
            verts = _plane_box_polygon(W, c, lo, hi)
            if verts is not None:
                plane = Poly3DCollection([verts], facecolors=P_FACE, edgecolors=P_EDGE, linewidths=0.8, alpha=0.30)
                ax.add_collection3d(plane)
                has_plane = True
                alpha_pred = _front_back_alpha(P, (W, c))

        m = y_pred == 1
        ax.scatter(*P[m].T, c=_rgba(C_POS, alpha_pred[m]), s=7, depthshade=False, linewidths=0.15, edgecolors="white")
        ax.scatter(*P[~m].T, c=_rgba(C_NEG, alpha_pred[~m]), s=7, depthshade=False, linewidths=0.15, edgecolors="white")
        mis = y_pred != y_true
        if mis.sum() > 0:
            ax.scatter(*P[mis].T, marker="x", c=C_MISS, s=20, linewidths=0.9)


    # ---- 图例：整图下方（手动预留空间避免与 3D 内容重叠） ----
    handles = [
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_POS, markeredgecolor="#B9C6D9", markersize=5, label="y = +1"),
        Line2D([], [], marker="o", ls="none", color="none", markerfacecolor=C_NEG, markeredgecolor="#DDB3B3", markersize=5, label="y = −1"),
    ]
    if true_plane is not None:
        handles.append(Line2D([], [], color=C_TRUE, lw=1.0, ls="--", label="真实边界"))
    if has_plane:
        handles.append(Patch(facecolor=P_FACE, edgecolor=P_EDGE, label=boundary_label))
    handles.append(Line2D([], [], marker="x", ls="none", color=C_MISS, markersize=6, label="误分类测试点"))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.26, wspace=0.24)
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, columnspacing=1.2, handlelength=1.6)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 3D 轴标签与图例需显式加入 tight 裁剪范围，避免被裁掉（不影响布局）
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", bbox_extra_artists=[leg, *[ax.zaxis.label for ax in axes]])
    plt.close(fig)
    return out_path
