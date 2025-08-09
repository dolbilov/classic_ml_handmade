import random
from typing import Callable, Literal
import numpy as np
import pandas as pd
from regression_metrics import RegressionMetrics
from exceptions import NotFittedError


class LinearRegression:
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: str | None = None,
        reg: Literal['l1', 'l2', 'elasticnet'] | None = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: int | float | None = None,
        random_state: int = 42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.final_score: float | None = None
        self.metric = metric
        self.metric_function: Callable[[np.ndarray, np.ndarray], float] | None = RegressionMetrics.get_metric_by_name(metric) if metric else None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        random.seed(random_state)

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __get_learning_rate(self, iteration_number: int) -> float:
        return self.learning_rate if isinstance(self.learning_rate, int | float) else self.learning_rate(iteration_number)

    def __get_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError('weights is not initialized')

        loss = np.square(y_true - y_pred).mean()

        if self.reg in ('l1', 'elasticnet'):
            loss += self.l1_coef * np.abs(self.weights).sum()

        if self.reg in ('l2', 'elasticnet'):
            loss += self.l2_coef * np.square(self.weights).sum()

        return loss

    def __get_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError('weights is not initialized')

        grad = (2 / X.shape[0]) * (y_pred - y_true) @ X

        if self.reg in ('l1', 'elasticnet'):
            grad += self.l1_coef * np.sign(self.weights)

        if self.reg in ('l2', 'elasticnet'):
            grad += self.l2_coef * 2 * self.weights

        return grad

    def __get_sgd_sample_k(self, X: np.ndarray) -> int:
        n = X.shape[0]

        if self.sgd_sample is None:
            return n

        if isinstance(self.sgd_sample, int):
            return min(n, self.sgd_sample)

        if isinstance(self.sgd_sample, float):
            return int(self.sgd_sample * n)

        raise TypeError

    def fit(self, X_: pd.DataFrame, y_: pd.Series, verbose: int = 0) -> None:
        """
        Trains the linear regression model using gradient descent (optionally with SGD).

        Parameters:
            X_ (pd.DataFrame): Feature matrix of shape (n_samples, n_features).
            y_ (pd.Series): Target vector of shape (n_samples,).
            verbose (int): If greater than 0, prints loss and metric every `verbose` iterations.

        Raises:
            TypeError: If `sgd_sample` is neither int nor float when specified.
        """
        X = X_.to_numpy()
        y = y_.to_numpy()

        sgd_sample_k = self.__get_sgd_sample_k(X)

        X = np.insert(X, 0, 1, axis=1)
        features_count = X.shape[1]
        self.weights = np.ones(features_count)

        for i in range(1, self.n_iter + 1):
            sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample_k)
            X_batch = X[sample_rows_idx]

            y_pred_batch = X_batch @ self.weights
            grad = self.__get_grad(X_batch, y[sample_rows_idx], y_pred_batch)
            lr = self.__get_learning_rate(i)
            self.weights -= lr * grad

            if verbose and (i == 1 or i % verbose == 0):
                y_pred = X @ self.weights
                loss = self.__get_loss(y, y_pred)
                log_text = f'[{i}/{self.n_iter}] | loss = {loss:.2f}'
                if self.metric_function is not None:
                    metric = self.metric_function(y, y_pred)
                    log_text += f' | {self.metric} = {metric:.2f}'
                print(log_text)

        # getting final score in fitted model
        y_pred = X @ self.weights
        if self.metric_function is not None:
            self.final_score = self.metric_function(y, y_pred)

    def predict(self, X_: pd.DataFrame) -> np.ndarray:
        """
        Predicts target values for given input features using the trained model.

        Parameters:
            X_ (pd.DataFrame): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if self.weights is None:
            raise NotFittedError

        X = X_.to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights

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

    def get_final_score(self) -> float:
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
