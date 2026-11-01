import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.datasets.synthetic import make_circles, train_val_test_split
from src.eval.plotting import plot_boundary_comparison
from src.run import FIG_DIR, CSV_DIR, format_table, run_experiment, save_csv

N = 1200

X, y = make_circles(N, r_in=1.0, r_out_lo=1.4, r_out_hi=2.4, seed=42)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=42)
records, arts = run_experiment(
    X_train, y_train, X_val, y_val, X_test, y_test,
    context={"n": N, "r_in": 1.0, "r_out": "1.4~2.4"}, return_artifacts=True,
)

print("exp2: nonlinear boundary (disk vs annulus)")
print(format_table(records))
save_csv(records, CSV_DIR / "exp2_circle.csv")

R_BAYES = 1.2  # 单位圆 r<=1.0 与外环 r>=1.4 间隙的中点，即 Bayes 最优边界半径

plot_boundary_comparison(
    X_test, y_test,
    [(name.upper(), {"pred": a["pred"], "model": a["model"]}) for name, a in arts.items()],
    out_path=FIG_DIR / "exp2_circle.png",
    feature_names=("$x_1$", "$x_2$"),
    true_score=lambda G: R_BAYES - np.sqrt((G ** 2).sum(axis=1)),
)
