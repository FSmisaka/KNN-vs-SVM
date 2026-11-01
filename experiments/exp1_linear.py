import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import make_linear, train_val_test_split
from src.eval.plotting import plot_boundary_comparison
from src.run import FIG_DIR, CSV_DIR, format_table, run_experiment, save_csv

MU = 2.0
SIGMAS = [0.5, 1.0, 1.5]
N = 1200

# 决策边界图展示的设置：噪声最大（类别重叠最严重）时，边界的放置最有信息量
SHOW_SIGMA = 1.5

records = []
arts = None
for sigma in SIGMAS:
    X, y = make_linear(N, mu=MU, sigma=sigma, seed=42)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=42)
    if sigma == SHOW_SIGMA:
        rec, arts = run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            context={"mu": MU, "sigma": sigma}, return_artifacts=True,
        )
        records += rec
        show_split = (X_test, y_test)
    else:
        records += run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test, context={"mu": MU, "sigma": sigma}
        )

print("exp1: linearly (near-)separable data")
print(format_table(records))
save_csv(records, CSV_DIR / "exp1_linear.csv")

plot_boundary_comparison(
    *show_split,
    [(name.upper(), {"pred": a["pred"], "model": a["model"]}) for name, a in arts.items()],
    out_path=FIG_DIR / "exp1_linear.png",
    feature_names=("$x_1$", "$x_2$"),
    true_score=lambda G: G[:, 0],
)
