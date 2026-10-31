import itertools

from src.eval.metrics import accuracy


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
