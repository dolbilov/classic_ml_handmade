from typing import Callable
import numpy as np
import pandas as pd
from regression_metrics import RegressionMetrics


class MyLineReg:
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: str | None = None,
        reg: str | None = None,
        l1_coef: float = 0,
        l2_coef: float = 0
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.best_score: float | None = None
        self.metric = metric
        self.metric_function = RegressionMetrics.get_metric_by_name(metric) if metric else None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __get_learning_rate(self, iteration_number: int) -> float:
        return self.learning_rate if isinstance(self.learning_rate, int | float) else self.learning_rate(iteration_number)

    def __get_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        loss = np.square(y_true - y_pred).mean()

        if self.reg in ('l1', 'elasticnet'):
            loss += self.l1_coef * np.abs(self.weights).sum()

        if self.reg in ('l2', 'elasticnet'):
            loss += self.l2_coef * np.square(self.weights).sum()

        return loss

    def __get_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        grad = (2 / X.shape[0]) * (y_pred - y_true) @ X

        if self.reg in ('l1', 'elasticnet'):
            grad += self.l1_coef * np.sign(self.weights)

        if self.reg in ('l2', 'elasticnet'):
            grad += self.l2_coef * 2 * self.weights

        return grad

    def fit(self, X_: pd.DataFrame, y_: pd.Series, verbose: bool | int = False) -> None:
        X = X_.to_numpy()
        y = y_.to_numpy()

        X = np.insert(X, 0, 1, axis=1)
        features_count = X.shape[1]
        self.weights = np.ones(features_count)

        for i in range(1, self.n_iter + 1):
            y_pred = X @ self.weights
            grad = self.__get_grad(X, y, y_pred)
            lr = self.__get_learning_rate(i)
            self.weights -= lr * grad

            if verbose and (i == 1 or i % verbose == 0):
                loss = self.__get_loss(y, y_pred)
                log_text = f'[{i}/{self.n_iter}] | loss = {loss:.2f}'
                if self.metric_function is not None:
                    metric = self.metric_function(y, y_pred)
                    log_text += f' | {self.metric} = {metric:.2f}'
                print(log_text)

        # getting best score in fitted model
        y_pred = X @ self.weights
        self.best_score = self.metric_function(y, y_pred) if self.metric_function is not None else None

    def predict(self, X_: pd.DataFrame) -> np.ndarray:
        X = X_.to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def get_best_score(self) -> float | None:
        return self.best_score
