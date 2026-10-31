import numpy as np

class SVM:
    def __init__(self, C: float = 1.0, lr: float = 0.1, n_epochs: int = 500, tol: float = 1e-6):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.tol = tol

    def get_params(self):
        return {"C": self.C, "lr": self.lr, "n_epochs": self.n_epochs, "tol": self.tol}

    def set_params(self, **params):
        merged = {**self.get_params(), **params}
        self.__init__(**merged)
        return self

    def _loss(self, X, y, w, b):
        return 0.5 * w @ w + self.C * np.maximum(0.0, 1.0 - y * (X @ w + b)).mean()

    def _grads(self, X, y, w, b):
        viol = y * (X @ w + b) < 1.0
        gw = w.copy()
        gb = 0.0
        if viol.any():
            gw = w - self.C * (X[viol] * y[viol, None]).sum(axis=0) / len(y)
            gb = -self.C * y[viol].sum() / len(y)
        return gw, gb

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.zeros(X.shape[1])
        b = 0.0
        lr = self.lr
        loss = self._loss(X, y, w, b)
        self.loss_history_ = [loss]
        for _ in range(self.n_epochs):
            gw, gb = self._grads(X, y, w, b)

            # 梯度裁剪
            gnorm = np.sqrt(gw @ gw + gb * gb)
            cap = 10.0 * (1.0 + np.sqrt(w @ w + b * b))
            if gnorm > cap:
                gw, gb = gw * (cap / gnorm), gb * (cap / gnorm)

            # 回溯线搜索
            accepted = False
            for _ in range(60):
                w_new = w - lr * gw
                b_new = b - lr * gb
                loss_new = self._loss(X, y, w_new, b_new)
                if np.isfinite(loss_new) and loss_new <= loss + 1e-12 * (1.0 + abs(loss)):
                    accepted = True
                    break
                lr *= 0.5
            if not accepted: break
        
            w, b, loss = w_new, b_new, loss_new

            # 早停
            self.loss_history_.append(loss)
            if self.loss_history_[-2] - loss < self.tol * (1.0 + loss):
                break

        self.w_ = w
        self.b_ = b
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(X @ self.w_ + self.b_ >= 0, 1, -1)
