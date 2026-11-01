import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.eval.plotting import plot_search_curves_combined
from src.run import CSV_DIR, FIG_DIR


def load_rows():
    with open(CSV_DIR / "exp7_real_search.csv") as f:
        return list(csv.DictReader(f))


def to_records(rows, model):
    """把 CSV 行转回 plot_search_curves_combined 需要的记录格式（数值/字符串还原）。"""
    recs = []
    for r in rows:
        if r["model"] != model:
            continue
        rec = {
            "train_accuracy": float(r["train_accuracy"]),
            "val_accuracy": float(r["val_accuracy"]),
        }
        if model == "knn":
            rec["k"] = int(r["k"])
            rec["weights"] = r["weights"]
        else:
            rec["C"] = float(r["C"])
        recs.append(rec)
    return recs


def pick_best(recs, model):
    key = (lambda r: (r["val_accuracy"],))  # 并列时 plot 的星标取首个最大值即可
    best_rec = max(recs, key=key)
    if model == "knn":
        return {"k": best_rec["k"], "weights": best_rec["weights"]}
    return {"C": best_rec["C"]}


def main():
    rows = load_rows()
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        n_used = sub[0]["n_used"]
        sampling = sub[0]["sampling"]
        knn, svm = to_records(sub, "knn"), to_records(sub, "svm")
        out = plot_search_curves_combined(
            knn, svm, pick_best(knn, "knn"), pick_best(svm, "svm"),
            out_path=FIG_DIR / f"exp7_{ds}.png",
            dataset_label=f"{ds}（n={n_used}，{'全量' if sampling == 'full' else '分层抽样，种子 42'}）",
        )
        print(f"figure -> {out}")


if __name__ == "__main__":
    main()
