import itertools
import time

from src.eval.metrics import accuracy, f1


def clone(model):
    return type(model)(**model.get_params())


def grid_search(model, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid)
    combos = (
        [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]
        if keys
        else [{}]
    )
    records = []
    for params in combos:
        m = clone(model).set_params(**params)
        m.fit(X_train, y_train)
        records.append({**params, "val_accuracy": accuracy(y_val, m.predict(X_val))})
    best = max(records, key=lambda r: r["val_accuracy"])
    return {k: best[k] for k in keys}, records


def grid_search_curves(model, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid)
    combos = (
        [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]
        if keys
        else [{}]
    )
    records = []
    for params in combos:
        m = clone(model).set_params(**params)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_train = m.predict(X_train)
        train_predict_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_val = m.predict(X_val)
        val_predict_s = time.perf_counter() - t0
        records.append(
            {
                **params,
                "train_accuracy": accuracy(y_train, pred_train),
                "train_f1": f1(y_train, pred_train),
                "val_accuracy": accuracy(y_val, pred_val),
                "val_f1": f1(y_val, pred_val),
                "fit_s": fit_s,
                "train_predict_s": train_predict_s,
                "val_predict_s": val_predict_s,
            }
        )
    best = max(records, key=lambda r: (r["val_accuracy"], r["val_f1"]))
    return {k: best[k] for k in keys}, records
