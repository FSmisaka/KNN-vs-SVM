import numpy as np


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, seed=0):
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(len(y) * test_size))
    n_val = int(round(len(y) * val_size))
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]


def make_linear(n, mu=2.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    X = np.empty((n, 2))
    X[:, 0] = y * mu + rng.normal(0.0, sigma, n)
    X[:, 1] = rng.normal(0.0, sigma, n)
    return X, y


def make_circles(n, r_in=1.0, r_out_lo=1.4, r_out_hi=2.4, seed=0):
    rng = np.random.default_rng(seed)
    n_in = n // 2
    n_out = n - n_in
    r1 = r_in * np.sqrt(rng.random(n_in))
    t1 = 2 * np.pi * rng.random(n_in)
    r2 = r_out_lo + (r_out_hi - r_out_lo) * rng.random(n_out)
    t2 = 2 * np.pi * rng.random(n_out)
    X = np.concatenate(
        [
            np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1),
            np.stack([r2 * np.cos(t2), r2 * np.sin(t2)], axis=1),
        ]
    )
    y = np.concatenate([np.ones(n_in), -np.ones(n_out)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_sparse_highdim(n, d=50, mu=2.0, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    X = rng.normal(0.0, 5.0, (n, d))
    X[:, 0] = y * mu + rng.normal(0.0, sigma, n)
    return X, y


def make_linear_rule(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 2))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] >= 0, 1, -1)
    return X, y


def flip_labels(y, eta, seed=0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y)) < eta
    return np.where(flip, -y, y)


# make_linear_rule 的真实超平面 x1 + 0.5*x2 = 0 的坐标：n̂ 为单位法向，v 为沿边界单位方向
_N_HAT = np.array([1.0, 0.5]) / np.sqrt(1.25)
_V_ALONG = np.array([-0.5, 1.0]) / np.sqrt(1.25)


def inject_structured_flips(X, y, eta, mu=3.0, sigma_t=2, s_lo=1.0, s_hi=1.6, seed=7):
    """向训练集注入「结构性错标」样本并返回增广后的 (X_aug, y_aug, X_flip, y_flip_obs)。

    机制：+1 类的错标点位于真实边界正侧、沿边界坐标 t ~ N(+mu, sigma_t^2)，
    深度 s ~ U(s_lo, s_hi)；-1 类对称地位于 t ~ N(-mu, sigma_t^2)。两类错标点
    的 t 分布标准差相同、均值显著不同（+mu vs -mu），在真实边界远端形成方向
    一致的「扭矩」。eta 为注入数量占原训练集的比例。错标点带相反的观测标签。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    m = int(round(eta * len(y)))
    X_flip, y_flip = [], []
    for c, mu_c in ((1, mu), (-1, -mu)):
        t = rng.normal(mu_c, sigma_t, m // 2)
        s = c * rng.uniform(s_lo, s_hi, m // 2)
        X_flip.append(np.outer(t, _V_ALONG) + np.outer(s, _N_HAT))
        y_flip.append(np.full(m // 2, -c, dtype=float))
    X_flip = np.concatenate(X_flip)
    y_flip = np.concatenate(y_flip)
    return np.vstack([X, X_flip]), np.concatenate([y.astype(float), y_flip]), X_flip, y_flip
