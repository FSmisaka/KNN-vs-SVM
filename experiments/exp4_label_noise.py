import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import flip_labels, make_linear_rule, train_val_test_split
from src.eval.metrics import accuracy, evaluate
from src.eval.plotting import plot_test_comparison
from src.run import FIG_DIR, CSV_DIR, build_model, format_table, save_csv

N = 2500
ETAS = [0.0, 0.1, 0.2, 0.3]
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
C_GRID = [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

# 决策边界图展示的设置：噪声最强（eta=0.3）时，对翻转标签的容忍效果最直观
SHOW_ETA = 0.3

X, y = make_linear_rule(N, seed=1)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=0)

records = []
best = {"knn": None, "svm": None}  # (val_accuracy, model)
for eta in ETAS:
    y_fit = flip_labels(y_train, eta=eta, seed=2)
    for k in K_GRID:
        model = build_model("knn", k=k).fit(X_train, y_fit)
        val_acc = accuracy(y_val, model.predict(X_val))
        if eta == SHOW_ETA and (best["knn"] is None or val_acc > best["knn"][0]):
            best["knn"] = (val_acc, model)
        records.append(
            {
                "model": "knn",
                "eta": eta,
                "k": k,
                "val_accuracy": round(val_acc, 4),
                **evaluate(y_test, model.predict(X_test)),
            }
        )
    for C in C_GRID:
        model = build_model("svm", C=C).fit(X_train, y_fit)
        val_acc = accuracy(y_val, model.predict(X_val))
        if eta == SHOW_ETA and (best["svm"] is None or val_acc > best["svm"][0]):
            best["svm"] = (val_acc, model)
        records.append(
            {
                "model": "svm",
                "eta": eta,
                "C": C,
                "val_accuracy": round(val_acc, 4),
                **evaluate(y_test, model.predict(X_test)),
            }
        )

for eta in ETAS:
    print(f"exp4: label noise, eta={eta} (train labels flipped, val/test labels clean)")
    print(format_table([r for r in records if r["eta"] == eta]))
    print()
save_csv(records, CSV_DIR / "exp4_label_noise.csv")

BOUNDARY_LABELS = {
    "knn": "KNN 决策边界（多数投票切换处）",
    "svm": "SVM 超平面（w·x + b = 0）",
}
# 左子图染色用「观测标签」：对测试标签施加同比例翻转噪声（仅用于展示，
# 误分类标记与 CSV 指标仍相对干净的真实标签计算）
y_test_noisy = flip_labels(y_test, eta=SHOW_ETA, seed=3)
for name, (_, model) in best.items():
    plot_test_comparison(
        X_test, y_test, model.predict(X_test),
        model=model,
        out_path=FIG_DIR / f"exp4_label_noise_{name}.png",
        feature_names=("$x_1$", "$x_2$"),
        true_score=lambda G: G[:, 0] + 0.5 * G[:, 1],
        boundary_label=BOUNDARY_LABELS[name],
        y_display=y_test_noisy,
        true_title=f"观测标签（含 {SHOW_ETA:.0%} 翻转噪声）",
    )
