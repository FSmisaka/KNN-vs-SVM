from typing import Any, Dict, List

CONFIG: Dict[str, Any] = {
    "C": 1.0,
    "lr": 0.1,
    "n_epochs": 500,
    "tol": 1e-6,
}

GRID: Dict[str, List[Any]] = {
    "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
}
