from typing import Literal, Callable
import numpy as np
import pandas as pd
from NotFittedError import NotFittedError
from classification_metrics import ClassificationMetrics


class MyLogReg:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float = 0.1,
        metric: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] | None = None,
        reg: Literal['l1', 'l2', 'elasticnet'] | None = None,
        l1_coef: float = 0,
        l2_coef: float = 0
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.metric = metric
        self.metric_function: Callable[[np.ndarray, np.ndarray], float] | None = ClassificationMetrics.get_metric_by_name(metric) if metric else None
        self.final_score: float | None = None
        self.reg =reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __get_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

    def fit(self, X_: pd.DataFrame, y_: pd.Series, verbose: int = 0) -> None:
        X = X_.to_numpy()
        y = y_.to_numpy()

        X = np.insert(X, 0, 1, axis=1)
        features_count = X.shape[1]
        self.weights = np.ones(features_count)

        for i in range(1, self.n_iter + 1):
            y_pred = 1 / (1 + np.exp(-X @ self.weights))
            log_loss = self.__get_log_loss(y, y_pred)
            grad = (y_pred - y) @ X / X.shape[0]
            self.weights -= self.learning_rate * grad

            if verbose and (i == 1 or i % verbose == 0):
                log_text = f'[{i}/{self.n_iter}] | loss = {log_loss:.2f}'
                if self.metric_function:
                    preds = y_pred if self.metric == 'roc_auc' else (y_pred > 0.5).astype(int)
                    metric = self.metric_function(y, preds)
                    log_text += f' | {self.metric} = {metric:.2f}'
                print(log_text)

        if self.metric_function:
            y_pred = 1 / (1 + np.exp(-X @ self.weights))
            preds = y_pred if self.metric == 'roc_auc' else (y_pred > 0.5).astype(int)
            self.final_score = self.metric_function(y, preds)

    def predict(self, X_: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X_: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise NotFittedError

        X = X_.to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        return 1 / (1 + np.exp(-X @ self.weights))

    def get_coef(self) -> np.ndarray:
        """
        Returns the learned model coefficients (excluding the bias term).

        Returns:
            np.ndarray: Model coefficients of shape (n_features,).

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if self.weights is None:
            raise NotFittedError

        return self.weights[1:]

    def get_best_score(self) -> float:
        """
        Returns the final evaluation score based on the selected metric after training.

        Returns:
            float: Final score computed using the selected metric.

        Raises:
            NotFittedError: If the model has not been fitted yet.
            RuntimeError: If no evaluation metric was selected during initialization.
        """
        if self.weights is None:
            raise NotFittedError

        if self.final_score is None:
            raise RuntimeError('Metric is not selected')

        return self.final_score
