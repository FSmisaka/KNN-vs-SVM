import time
from src.eval.metrics import evaluate

class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_time = None
        self.predict_time = None

    def train(self, X, y):
        start = time.perf_counter()
        self.model.fit(X, y)
        self.train_time = time.perf_counter() - start
        return self

    def predict(self, X):
        start = time.perf_counter()
        pred = self.model.predict(X)
        self.predict_time = time.perf_counter() - start
        return pred

    def evaluate(self, X_test, y_test):
        pred = self.predict(X_test)
        return {
            **evaluate(y_test, pred),
            "train_s": self.train_time,
            "predict_s": self.predict_time,
        }
