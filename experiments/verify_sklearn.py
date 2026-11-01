import csv
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.datasets.real import LOADERS, Standardizer, stratified_subsample
from src.datasets.synthetic import (
    inject_structured_flips,
    make_circles,
    make_linear,
    make_linear_rule,
    make_sparse_highdim,
    train_val_test_split,
)
from src.eval.metrics import evaluate
from src.models.knn import KNN
from src.models.svm import SVM
from src.run import CSV_DIR

TOL = 0.02  # SVM 指标一致性容差
N = {"exp1": 1200, "exp2": 1200, "exp3": 1000, "exp4": 2500}
MU, SIGMA_T, S_LO, S_HI = 3.0, 1, 1.0, 1.6


def read_rows(name):
    with open(CSV_DIR / name) as f:
        return list(csv.DictReader(f))


def split_synthetic(X, y):
    return train_val_test_split(X, y, seed=42)


def knn_sklearn(k, weights):
    return KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm="brute")


def compare(name, setting, X_train, y_train, X_test, y_test, kind, params):
    """同一份数据上分别拟合自实现与 sklearn，返回一行对照记录。"""
    n_train = len(y_train)
    if kind == "knn":
        ours = KNN(k=params["k"], weights=params["weights"]).fit(X_train, y_train)
        sk = knn_sklearn(params["k"], params["weights"]).fit(X_train, y_train)
        obj_ours = obj_sk = ""
    else:
        ours = SVM(C=params["C"]).fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = SVC(kernel="linear", C=params["C"] / n_train, tol=1e-4).fit(X_train, y_train)
        m = SVM(C=params["C"])
        obj_ours = m._loss(X_train, y_train, ours.w_, ours.b_)
        obj_sk = m._loss(X_train, y_train, sk.coef_[0], sk.intercept_[0])

    print(f"  fitting {name} {setting} {kind.upper()} ...", flush=True)
    m_ours = evaluate(y_test, ours.predict(X_test))
    m_sk = evaluate(y_test, sk.predict(X_test))
    d_acc = abs(m_ours["accuracy"] - m_sk["accuracy"])
    d_f1 = abs(m_ours["f1"] - m_sk["f1"])
    ok = (d_acc == 0 and d_f1 == 0) if kind == "knn" else (d_acc <= TOL and d_f1 <= TOL)
    hp = f"k={params['k']},{params['weights']}" if kind == "knn" else f"C={params['C']:g}"
    return {
        "experiment": name,
        "setting": setting,
        "model": kind.upper(),
        "hyperparams": hp,
        "ours_accuracy": round(m_ours["accuracy"], 4),
        "ours_f1": round(m_ours["f1"], 4),
        "sklearn_accuracy": round(m_sk["accuracy"], 4),
        "sklearn_f1": round(m_sk["f1"], 4),
        "diff_accuracy": round(d_acc, 4),
        "diff_f1": round(d_f1, 4),
        "ours_train_obj": round(obj_ours, 6) if obj_ours != "" else "",
        "sklearn_train_obj": round(obj_sk, 6) if obj_sk != "" else "",
        "consistent": ok,
    }


def synthetic_rows():
    rows = []

    # 实验 1：三档 sigma，选中超参取自 exp1_linear.csv
    for r in read_rows("exp1_linear.csv"):
        sig = float(r["sigma"])
        X, y = make_linear(N["exp1"], mu=2.0, sigma=sig, seed=42)
        Xtr, Xv, Xte, ytr, yv, yte = split_synthetic(X, y)
        setting = f"sigma={sig:g}"
        if r["model"] == "knn":
            rows.append(compare("实验1", setting, Xtr, ytr, Xte, yte, "knn",
                                {"k": int(r["k"]), "weights": "uniform"}))
        else:
            rows.append(compare("实验1", setting, Xtr, ytr, Xte, yte, "svm", {"C": float(r["C"])}))

    # 实验 2
    X, y = make_circles(N["exp2"], r_in=1.0, r_out_lo=1.4, r_out_hi=2.4, seed=42)
    Xtr, Xv, Xte, ytr, yv, yte = split_synthetic(X, y)
    for r in read_rows("exp2_circle.csv"):
        if r["model"] == "knn":
            rows.append(compare("实验2", "单位圆对外环", Xtr, ytr, Xte, yte, "knn",
                                {"k": int(r["k"]), "weights": "uniform"}))
        else:
            rows.append(compare("实验2", "单位圆对外环", Xtr, ytr, Xte, yte, "svm", {"C": float(r["C"])}))

    # 实验 3：七档维度
    for r in read_rows("exp3_highdim.csv"):
        d = int(r["d"])
        X, y = make_sparse_highdim(N["exp3"], d=d, mu=1.5, sigma=0.5, seed=42)
        Xtr, Xv, Xte, ytr, yv, yte = split_synthetic(X, y)
        setting = f"d={d}"
        if r["model"] == "knn":
            rows.append(compare("实验3", setting, Xtr, ytr, Xte, yte, "knn",
                                {"k": int(r["k"]), "weights": "uniform"}))
        else:
            rows.append(compare("实验3", setting, Xtr, ytr, Xte, yte, "svm", {"C": float(r["C"])}))

    # 实验 4：五档 eta，选中超参按验证集准确率复选（与原脚本同序、同并列规则）
    X, y = make_linear_rule(N["exp4"], seed=42)
    Xtr, Xv, Xte, ytr, yv, yte = split_synthetic(X, y)
    grid = read_rows("exp4_label_noise.csv")
    for eta in sorted({float(r["eta"]) for r in grid}):
        sub = [r for r in grid if float(r["eta"]) == eta]
        X_fit, y_fit, _, _ = inject_structured_flips(
            Xtr, ytr, eta, mu=MU, sigma_t=SIGMA_T, s_lo=S_LO, s_hi=S_HI, seed=42
        )
        setting = f"eta={eta:g}"
        best_knn = max((r for r in sub if r["model"] == "knn"), key=lambda r: float(r["val_accuracy"]))
        best_svm = max((r for r in sub if r["model"] == "svm"), key=lambda r: float(r["val_accuracy"]))
        rows.append(compare("实验4", setting, X_fit, y_fit, Xte, yte, "knn",
                            {"k": int(best_knn["k"]), "weights": "uniform"}))
        rows.append(compare("实验4", setting, X_fit, y_fit, Xte, yte, "svm", {"C": float(best_svm["C"])}))
    return rows


def real_rows():
    rows = []
    test = read_rows("exp7_real_test.csv")
    sel = {}
    for r in test:
        sel.setdefault(r["dataset"], {})[r["model"]] = r

    for ds, n_sub in (("webspam", None), ("hepmass", 100000)):
        X, y = LOADERS[ds](Path(__file__).resolve().parents[1] / "data" / (
            "webspam_30k.svm" if ds == "webspam" else "HEPMASS_500k.csv"))
        if n_sub is None:
            Xs, ys = X, y
        else:
            Xs, ys = stratified_subsample(X, y, n_sub, seed=42)
        Xtr, Xv, Xte, ytr, yv, yte = train_val_test_split(Xs, ys, seed=42)
        scaler = Standardizer().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

        kn = sel[ds]["knn"]
        rows.append(compare("实验7", ds, Xtr, ytr, Xte, yte, "knn",
                            {"k": int(kn["k"]), "weights": kn["weights"]}))
        sv = sel[ds]["svm"]
        rows.append(compare("实验7", ds, Xtr, ytr, Xte, yte, "svm", {"C": float(sv["C"])}))
    return rows


def main():
    rows = synthetic_rows() + real_rows()
    out = CSV_DIR / "verify_sklearn.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    knn_rows = [r for r in rows if r["model"] == "KNN"]
    svm_rows = [r for r in rows if r["model"] == "SVM"]
    print(f"写入 {out}，共 {len(rows)} 行")
    print(f"KNN: {len(knn_rows)} 行，最大准确率差异 {max(r['diff_accuracy'] for r in knn_rows):.4f}，"
          f"全部一致 {all(r['consistent'] for r in knn_rows)}")
    print(f"SVM: {len(svm_rows)} 行，最大准确率差异 {max(r['diff_accuracy'] for r in svm_rows):.4f}，"
          f"最大 F1 差异 {max(r['diff_f1'] for r in svm_rows):.4f}，"
          f"全部一致 {all(r['consistent'] for r in svm_rows)}")
    print(f"SVM 目标值: 我方不劣于 sklearn 的行数 "
          f"{sum(r['ours_train_obj'] <= r['sklearn_train_obj'] + 1e-6 for r in svm_rows)}/{len(svm_rows)}")


if __name__ == "__main__":
    main()
