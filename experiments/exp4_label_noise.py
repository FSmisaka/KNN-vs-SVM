import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import inject_structured_flips, make_linear_rule, train_val_test_split
from src.eval.metrics import accuracy, evaluate
from src.eval.plotting import plot_test_comparison
from src.run import FIG_DIR, CSV_DIR, build_model, format_table, save_csv

N = 2500
ETAS = [0.0, 0.02, 0.05, 0.08, 0.12]
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

# 结构性错标点：+1 类注入点沿真实边界的投影 t ~ N(+3, 0.5^2)，
# -1 类 t ~ N(-3, 0.5^2)（同标准差、均值显著不同），深度 s ~ U(1.0, 1.6)
MU = 3.0
SIGMA_T = 1
S_LO, S_HI = 1.0, 1.6

# 决策边界图展示的设置
SHOW_ETA = 0.02

# 协议：仅训练集被注入结构性错标点；验证/测试标签干净，
# 指标衡量学到的边界偏离真实规则的程度
X, y = make_linear_rule(N, seed=1)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=0)

records = []
best = {"knn": None, "svm": None}  # (val_accuracy, model, pred)
show_flips = None
for eta in ETAS:
    X_fit, y_fit, X_flip, y_flip_obs = inject_structured_flips(
        X_train, y_train, eta, mu=MU, sigma_t=SIGMA_T, s_lo=S_LO, s_hi=S_HI
    )
    if eta == SHOW_ETA:
        show_flips = (X_flip, y_flip_obs)
    for k in K_GRID:
        model = build_model("knn", k=k).fit(X_fit, y_fit)
        val_acc = accuracy(y_val, model.predict(X_val))
        pred = model.predict(X_test)
        if eta == SHOW_ETA and (best["knn"] is None or val_acc > best["knn"][0]):
            best["knn"] = (val_acc, model, pred)
        records.append(
            {
                "model": "knn",
                "eta": eta,
                "k": k,
                "val_accuracy": round(val_acc, 4),
                **evaluate(y_test, pred),
            }
        )
    for C in C_GRID:
        model = build_model("svm", C=C).fit(X_fit, y_fit)
        val_acc = accuracy(y_val, model.predict(X_val))
        pred = model.predict(X_test)
        if eta == SHOW_ETA and (best["svm"] is None or val_acc > best["svm"][0]):
            best["svm"] = (val_acc, model, pred)
        records.append(
            {
                "model": "svm",
                "eta": eta,
                "C": C,
                "val_accuracy": round(val_acc, 4),
                **evaluate(y_test, pred),
            }
        )

for eta in ETAS:
    print(f"exp4: structured label noise, eta={eta} (flips injected at t~N(±{MU},{SIGMA_T}^2) along true boundary; val/test clean)")
    print(format_table([r for r in records if r["eta"] == eta]))
    print()
save_csv(records, CSV_DIR / "exp4_label_noise.csv")

BOUNDARY_LABELS = {
    "knn": "KNN 决策边界（多数投票切换处）",
    "svm": "SVM 超平面（w·x + b = 0）",
}
for name, (_, model, pred) in best.items():
    plot_test_comparison(
        X_test, y_test, pred,
        model=model,
        out_path=FIG_DIR / f"exp4_label_noise_{name}.png",
        feature_names=("$x_1$", "$x_2$"),
        true_score=lambda G: G[:, 0] + 0.5 * G[:, 1],
        boundary_label=BOUNDARY_LABELS[name],
        extra_points=(*show_flips, "训练集注入的错标点（按观测标签）"),
    )
