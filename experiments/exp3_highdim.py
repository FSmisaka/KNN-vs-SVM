import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import make_sparse_highdim, train_val_test_split
from src.eval.plotting import plot_test_comparison
from src.run import FIG_DIR, CSV_DIR, format_table, run_experiment, save_csv

DIMS = [2, 10, 50, 100, 200, 500, 1000]
N = 1000

# 决策边界图展示的设置：d=200 时 SVM(0.935) 与 KNN(0.850) 差距已拉开且趋势稳定
SHOW_D = 200

records = []
arts = None
for d in DIMS:
    X, y = make_sparse_highdim(N, d=d, mu=1.5, sigma=1.0, seed=42)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=0)
    if d == SHOW_D:
        rec, arts = run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            context={"d": d, "n_informative": 1}, return_artifacts=True,
        )
        records += rec
        show_split = (X_test, y_test)
    else:
        records += run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test, context={"d": d, "n_informative": 1}
        )

print("exp3: high-dim data, only dim-1 informative")
print(format_table(records))
save_csv(records, CSV_DIR / "exp3_highdim.csv")

BOUNDARY_LABELS = {
    "knn": "KNN 决策边界（多数投票切换处）",
    "svm": "SVM 超平面（w·x + b = 0）",
}
for name, a in arts.items():
    plot_test_comparison(
        *show_split, a["pred"],
        model=a["model"],
        out_path=FIG_DIR / f"exp3_highdim_{name}.png",
        dims=(0, 1),
        feature_names=("维度 1（唯一有效维度）", "维度 2"),
        true_score=lambda G: G[:, 0],
        boundary_label=BOUNDARY_LABELS[name],
    )
