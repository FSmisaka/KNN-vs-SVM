import numpy as np

def confusion_counts(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    tn = int(((y_pred != positive) & (y_true != positive)).sum())
    return tp, fp, fn, tn

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def precision(y_true, y_pred, positive=1):
    tp, fp, _, _ = confusion_counts(y_true, y_pred, positive)
    return tp / (tp + fp) if tp + fp > 0 else 0.0

def recall(y_true, y_pred, positive=1):
    tp, _, fn, _ = confusion_counts(y_true, y_pred, positive)
    return tp / (tp + fn) if tp + fn > 0 else 0.0

def f1(y_true, y_pred, positive=1):
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

def evaluate(y_true, y_pred, positive=1):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, positive),
        "recall": recall(y_true, y_pred, positive),
        "f1": f1(y_true, y_pred, positive),
    }
