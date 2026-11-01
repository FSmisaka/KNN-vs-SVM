import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.real import DATA_INFO, LOADERS, stratified_subsample, Standardizer
from src.datasets.synthetic import train_val_test_split
from src.eval.evaluate import grid_search_curves
from src.eval.metrics import evaluate
from src.eval.plotting import plot_search_curves_combined
from src.run import FIG_DIR, CSV_DIR, format_table, save_csv
from src.train import Trainer
from src.models.knn import KNN
from src.models.svm import SVM

SUBSEED = 42
SPLITSEED = 42

# 每个数据集：抽样量（None=全量）与网格；train 越大最优 k 通常越大，HEPMASS 的 k 上界相应放宽
DATASETS = {
    "webspam": {
        "n_sub": None,
        "knn": {"k": [1, 3, 5, 7, 9, 11, 15, 21, 31, 51], "weights": ["uniform", "distance"]},
        # 本实现 hinge 按样本数取均值（等效 C_eff = C/n），C 的网格据此跨越强正则到近硬间隔
        "svm": {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]},
    },
    "hepmass": {
        "n_sub": 100000,
        "knn": {"k": [1, 3, 5, 9, 15, 21, 31, 51, 71, 101], "weights": ["uniform", "distance"]},
        "svm": {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]},
    },
}
MODELS = {"knn": KNN, "svm": SVM}


def run_dataset(ds, data_dir):
    t0 = time.perf_counter()
    X, y = LOADERS[ds](data_dir)
    n_full, pos_full = len(y), (y == 1).mean()
    cfg = DATASETS[ds]
    if cfg["n_sub"] is None:
        Xs, ys, sampling = X, y, "full"
    else:
        Xs, ys = stratified_subsample(X, y, cfg["n_sub"], seed=SUBSEED)
        sampling = f"stratified(seed={SUBSEED})"
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(Xs, ys, seed=SPLITSEED)
    scaler = Standardizer().fit(X_train)  # 预处理统计量仅由训练集估计
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    print(
        f"[{ds}] 全量 {n_full} 条（正类 {pos_full:.1%}） -> {sampling} 使用 {len(ys)} 条"
        f" -> 切分 {len(y_train)}/{len(y_val)}/{len(y_test)}（seed={SPLITSEED}），"
        f"标准化后 {X_train.shape[1]} 维（训练集零方差列已丢弃），加载+预处理 {time.perf_counter() - t0:.1f}s"
    )

    ctx = {
        "dataset": ds,
        "n_full": n_full,
        "sampling": sampling,
        "n_used": len(ys),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
    }
    search_rows, test_rows, records_by_model, best_by_model = [], [], {}, {}
    for name in ("knn", "svm"):
        best, records = grid_search_curves(
            MODELS[name](), cfg[name], X_train, y_train, X_val, y_val
        )
        search_rows += [{**ctx, "model": name, **r} for r in records]
        records_by_model[name] = records
        best_by_model[name] = best

        trainer = Trainer(MODELS[name](**best))  # 仅训练集重训
        trainer.train(X_train, y_train)
        pred = trainer.predict(X_test)
        test_rows.append(
            {
                **ctx,
                "model": name,
                **best,
                **evaluate(y_test, pred),
                "train_s": round(trainer.train_time, 4),
                "predict_s": round(trainer.predict_time, 4),
            }
        )
        print(f"  {name}: best={best} -> test acc {test_rows[-1]['accuracy']:.4f}, f1 {test_rows[-1]['f1']:.4f}")

    f = plot_search_curves_combined(
        records_by_model["knn"], records_by_model["svm"],
        best_by_model["knn"], best_by_model["svm"],
        out_path=FIG_DIR / f"exp7_{ds}.png",
        dataset_label=f"{ds}（n={len(ys)}，{'全量' if sampling == 'full' else '分层抽样'}，切分种子 {SPLITSEED}）",
    )
    print(f"  figure -> {f}")
    return search_rows, test_rows


def main(datasets):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    all_search, all_test = [], []
    for ds in datasets:
        s, t = run_dataset(ds, data_dir / DATA_INFO[ds]["file"])
        all_search += s
        all_test += t

    save_csv(all_search, CSV_DIR / "exp7_real_search.csv")
    save_csv(all_test, CSV_DIR / "exp7_real_test.csv")
    print("\nexp7: 真实数据集（最终测试集指标，超参数由验证集选出）")
    print(format_table(all_test))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in LOADERS]
    main(args or ["webspam", "hepmass"])
