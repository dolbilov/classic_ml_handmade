import numpy as np
import pandas as pd
from regression_metrics import RegressionMetrics


class MyLineReg:
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        metric: str | None = None
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.best_score: float | None = None
        self.metric = metric
        self.metric_function = RegressionMetrics.get_metric_by_name(metric) if metric else None

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X_: pd.DataFrame, y_: pd.Series, verbose: bool | int = False) -> None:
        X = X_.to_numpy()
        y = y_.to_numpy()

        X = np.insert(X, 0, 1, axis=1)
        features_count = X.shape[1]
        self.weights = np.ones(features_count)

        for i in range(self.n_iter):
            y_pred = X @ self.weights
            grad = (2 / X.shape[0]) * (y_pred - y) @ X
            self.weights -= self.learning_rate * grad

            if verbose and (i == 0 or (i + 1) % verbose == 0):
                mse = RegressionMetrics.mean_squared_error(y, y_pred)
                log_text = f'[{i + 1}/{self.n_iter}] | loss = {mse:.2f}'
                if self.metric_function is not None:
                    metric = self.metric_function(y, y_pred)
                    log_text += f' | {self.metric} = {metric:.2f}'
                print(log_text)

    def predict(self, X_: pd.DataFrame) -> np.ndarray:
        X = X_.to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def get_best_score(self) -> float:
        return self.best_score
