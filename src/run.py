import importlib
from pathlib import Path

from src.eval.evaluate import grid_search
from src.eval.metrics import evaluate
from src.models.knn import KNN
from src.models.svm import SVM
from src.train import Trainer

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"

REGISTRY = {
    "knn": (KNN, "config.knn"),
    "svm": (SVM, "config.svm"),
}

def load_config(name):
    _, module = REGISTRY[name]
    mod = importlib.import_module(module)
    return dict(mod.CONFIG), dict(mod.GRID)

def build_model(name, **overrides):
    cls, _ = REGISTRY[name]
    cfg, _ = load_config(name)
    cfg.update(overrides)
    return cls(**cfg)

def run_experiment(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    models=("knn", "svm"),
    context=None,
    return_artifacts=False,
):
    records = []
    artifacts = {}
    for name in models:
        _, grid = load_config(name)
        best, gs_records = grid_search(build_model(name), grid, X_train, y_train, X_val, y_val)
        trainer = Trainer(build_model(name, **best))
        trainer.train(X_train, y_train)
        pred = trainer.predict(X_test)
        record = {
            "model": name,
            **best,
            **evaluate(y_test, pred),
            "train_s": round(trainer.train_time, 4),
            "predict_s": round(trainer.predict_time, 4),
        }
        records.append({**context, **record} if context else record)
        artifacts[name] = {
            "model": trainer.model,
            "best": best,
            "pred": pred,
            "val_accuracy": max(r["val_accuracy"] for r in gs_records),
            "record": records[-1],
        }
    return (records, artifacts) if return_artifacts else records

def format_table(records):
    cols = list(dict.fromkeys(k for r in records for k in r))
    rows = []
    for r in records:
        rows.append(
            [
                (
                    f"{r.get(c):.4f}"
                    if isinstance(r.get(c), float)
                    else "" if r.get(c) is None else str(r.get(c))
                )
                for c in cols
            ]
        )
    header = [str(c) for c in cols]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(cols))
    ]
    lines = [" | ".join(header[i].ljust(widths[i]) for i in range(len(cols)))]
    lines.append("-+-".join("-" * w for w in widths))
    lines.extend(" | ".join(rows[i][j].rjust(widths[j]) for j in range(len(cols))) for i in range(len(rows)))
    return "\n".join(lines)

def save_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in records for k in r))
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
