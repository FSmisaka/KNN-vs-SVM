import numpy as np

from src.datasets.synthetic import train_val_test_split  # noqa: F401  (re-export, 保持调用入口一致)


DATA_INFO = {
    "webspam": {
        "file": "webspam_30k.svm",
        # 原始 254 维 unigram 特征中，课程抽取版实际仅出现 0..128 共 129 维
        "n_features": 129,
        "positive": "+1 = 网页垃圾（spam）",
    },
    "hepmass": {
        "file": "HEPMASS_500k.csv",
        # CSV 每行 28 列 = label + 27 特征（表头只写了 27 个名字，以数据为准）
        "n_features": 27,
        "positive": "+1 = 信号事件（signal）",
    },
}


def load_webspam(path):
    """解析 LIBSVM 格式的 webspam_30k.svm 为稠密 (X, y)，标签已是 ±1。"""
    n_features = DATA_INFO["webspam"]["n_features"]
    X = np.zeros((30000, n_features))
    y = np.empty(30000)
    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.split()
            y[i] = float(parts[0])
            for tok in parts[1:]:
                idx, val = tok.split(":")
                X[i, int(idx)] = float(val)
    return X, y


def load_hepmass(path):
    """解析 HEPMASS_500k.csv 为稠密 (X, y)；标签 0/1 映射为 -1/+1（1 = 信号）。"""
    rows = []
    with open(path) as f:
        f.readline()  # 表头
        for ln in f:
            rows.append(np.fromstring(ln, sep=","))
    A = np.array(rows)
    return A[:, 1:], np.where(A[:, 0] == 1, 1.0, -1.0)


LOADERS = {"webspam": load_webspam, "hepmass": load_hepmass}


def stratified_subsample(X, y, n, seed=42):
    """按类别分层、无放回抽取 n 条（各类别配额按 round(n * 类别占比) 分配，余量给多数类）。

    返回 (X_sub, y_sub)，行顺序按类别分块拼接后再整体打乱，保证与随机划分兼容。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    quota = {c: int(round(n * (y == c).mean())) for c in classes}
    quota[classes[np.argmax([quota[c] for c in classes])]] += n - sum(quota.values())
    parts = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        take = rng.choice(idx, size=quota[c], replace=False)
        parts.append(take)
    idx = rng.permutation(np.concatenate(parts))
    return X[idx], y[idx]


class Standardizer:
    """z-score 标准化。fit 仅在训练集上估计均值/标准差；训练集零方差（含全零）列直接丢弃。"""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.keep_ = std > 0.0
        self.std_ = np.where(self.keep_, std, 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return ((X - self.mean_) / self.std_)[:, self.keep_]

    def fit_transform(self, X):
        return self.fit(X).transform(X)
