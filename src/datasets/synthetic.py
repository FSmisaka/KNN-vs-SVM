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


def make_sparse_highdim(n, d=50, mu=2.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    X = rng.normal(0.0, 1.0, (n, d))
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
