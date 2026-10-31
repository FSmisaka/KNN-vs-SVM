import numpy as np

class KNN:
    def __init__(self, k: int = 5):
        self.k = k

    def get_params(self):
        return {"k": self.k}

    def set_params(self, **params):
        merged = {**self.get_params(), **params}
        self.__init__(**merged)
        return self

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._k = min(self.k, len(self._y))
        return self

    def _distances(self, X):
        # 距离矩阵 = X norm + self._X norm - 2 * X * self._X 转置
        d2 = (X ** 2).sum(axis=1)[:, None] + (self._X ** 2).sum(axis=1)[None, :]
        d2 -= 2.0 * (X @ self._X.T)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = self._distances(X)
        # 每行取前 k 个距离最近的
        nn = np.argpartition(d, self._k - 1, axis=1)[:, :self._k]
        nb_y = self._y[nn]
        score = nb_y.sum(axis=1)
        return np.where(score >= 0, 1, -1)
