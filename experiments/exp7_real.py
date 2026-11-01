"""exp7 真实数据集：KNN vs SVM 的超参数搜索与最终测试（任务 c）。

协议（对两个数据集、两种模型完全一致，保证比较公平）：
1. 样本量：webspam（30k，课程已从原始 350k 抽取）直接使用全量；HEPMASS（500k）
   按类别分层、无放回抽取 100,000 条（seed=42，全量的 1/5，类别比例与全量一致）。
   HEPMASS 不用全量的原因：自实现 KNN 的预测需计算 n_query×n_train 稠密距离矩阵，
   全量 500k 切分后一次训练集自预测即 300k×300k≈720 GB，网格 20 个组合需数小时，
   故按作业说明以固定种子子集进行（webspam 与 HEPMASS 用同一套协议，仅样本量不同）。
2. train/val/test = 6:2:2（train_val_test_split, seed=42，与合成实验同一切分逻辑）。
3. 标准化的均值/标准差仅在训练集上估计；训练集零方差列（webspam 未出现的原始特征）被丢弃，
   同一变换作用于验证/测试集。
4. 网格搜索记录每个网格点的训练/验证 accuracy 与 F1；超参数仅由验证集 accuracy 选择
   （并列时取 F1 更高者），测试集不参与任何选择。
5. 选定超参数后仅在训练集上重训，测试集做一次最终评估。

指标选择理由：
- accuracy：两数据集的类别不平衡都不极端（webspam 正类 60.7%、HEPMASS 50.1%），
  accuracy 是与文献可比的总体指标；
- F1：webspam 中"把垃圾页漏判为正常页"与"误杀正常页"代价不对称，且正类为关注类，
  F1 对少数类的查准/查全更敏感，可防止 accuracy 被多数类主导的假象；
- HEPMASS 近乎平衡，F1 与 accuracy 接近，报告二者仅为协议统一。

运行：python experiments/exp7_real.py [webspam|hepmass ...]（默认两个数据集都跑）
参考耗时（M4 / 24GB）：webspam 约 4 分钟，hepmass 约 35 分钟（KNN 训练集自预测为主）。
"""

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
