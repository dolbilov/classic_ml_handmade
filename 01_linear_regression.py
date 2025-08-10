import random
from typing import Callable, Literal
import numpy as np
import pandas as pd
from regression_metrics import RegressionMetrics
from exceptions import NotFittedError


class LinearRegression:
    """Custom implementation of Linear Regression with optional regularization and SGD mini-batching.

    This implementation supports L1, L2, and ElasticNet regularization, different learning
    rate strategies, stochastic gradient descent, and tracking various regression metrics.
    """

    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: Literal['mae', 'mse', 'rmse', 'mape', 'r2'] | None = None,
        reg: Literal['l1', 'l2', 'elasticnet'] | None = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: int | float | None = None,
        random_state: int = 42,
        verbose: int = 0
    ) -> None:
        """
        Args:
        n_iter (int, optional): Number of gradient descent iterations.
            Defaults to 100.
        learning_rate (float | Callable[[int], float], optional): Learning rate
            or a callable that returns the learning rate for a given iteration.
            Defaults to 0.1.
        metric (Literal['mae', 'mse', 'rmse', 'mape', 'r2'] | None, optional):
            Name of the evaluation metric to compute during training.
            Defaults to None.
        reg (Literal['l1', 'l2', 'elasticnet'] | None, optional): Type of
            regularization to apply. Defaults to None.
        l1_coef (float, optional): L1 regularization coefficient. Defaults to 0.
        l2_coef (float, optional): L2 regularization coefficient. Defaults to 0.
        sgd_sample (int | float | None, optional): Number or fraction of samples
            to use for each SGD step. If None, uses full batch. Defaults to None.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
        verbose (int, optional): If greater than 0, prints loss and metric every
            `verbose` iterations. Defaults to 0.
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.metric_function: Callable[[np.ndarray, np.ndarray], float] | None = (
            RegressionMetrics.get_metric_by_name(metric) if metric else None
        )
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.verbose = verbose

        self.weights: np.ndarray | None = None
        self.final_score: float | None = None

        random.seed(random_state)
        np.random.randint(random_state)

    def __repr__(self) -> str:
        return (
            f'LinearRegression(n_iter={self.n_iter}, '
            f'learning_rate={self.learning_rate}, '
            f'metric={self.metric}, '
            f'reg={self.reg}, '
            f'l1_coef={self.l1_coef}, '
            f'l2_coef={self.l2_coef}, '
            f'sgd_sample={self.sgd_sample}, '
            f'random_state={self.random_state}, '
            f'verbose={self.verbose})'
        )

    def _get_learning_rate(self, iteration_number: int) -> float:
        return (
            self.learning_rate
            if isinstance(self.learning_rate, (int, float))
            else self.learning_rate(iteration_number)
        )

    def _get_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError('weights is not initialized')

        loss = np.square(y_true - y_pred).mean()

        if self.reg in ('l1', 'elasticnet'):
            loss += self.l1_coef * np.abs(self.weights).sum()

        if self.reg in ('l2', 'elasticnet'):
            loss += self.l2_coef * np.square(self.weights).sum()

        return loss

    def _get_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError('weights is not initialized')

        grad = (2 / X.shape[0]) * (y_pred - y_true) @ X

        if self.reg in ('l1', 'elasticnet'):
            grad += self.l1_coef * np.sign(self.weights)

        if self.reg in ('l2', 'elasticnet'):
            grad += self.l2_coef * 2 * self.weights

        return grad

    def _get_sgd_sample_k(self, n_samples: int) -> int:
        if self.sgd_sample is None:
            return n_samples
        if isinstance(self.sgd_sample, int):
            return min(n_samples, self.sgd_sample)
        if isinstance(self.sgd_sample, float):
            return int(self.sgd_sample * n_samples)

        raise TypeError('sgd_sample must be int, float, or None.')


    def fit(self, X_: pd.DataFrame | np.ndarray, y_: pd.Series | np.ndarray) -> None:
        """Fit the linear regression model.

        Args:
            X_ (pd.DataFrame | np.ndarray): Feature matrix of shape
                (n_samples, n_features).
            y_ (pd.Series | np.ndarray): Target vector of shape (n_samples,).

        Raises:
            TypeError: If `sgd_sample` is neither int nor float when specified.
        """
        X = np.asarray(X_, dtype=float)
        y = np.asarray(y_, dtype=float)

        sgd_sample_k = self._get_sgd_sample_k(X.shape[0])

        X = np.insert(X, 0, 1, axis=1)
        features_count = X.shape[1]
        self.weights = np.ones(features_count)

        for i in range(1, self.n_iter + 1):
            sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample_k)
            X_batch = X[sample_rows_idx]

            y_pred_batch = X_batch @ self.weights
            grad = self._get_grad(X_batch, y[sample_rows_idx], y_pred_batch)
            lr = self._get_learning_rate(i)
            self.weights -= lr * grad

            if self.verbose and (i == 1 or i % self.verbose == 0):
                y_pred = X @ self.weights
                loss = self._get_loss(y, y_pred)
                log_text = f'[{i}/{self.n_iter}] | loss = {loss:.2f}'
                if self.metric_function is not None:
                    metric = self.metric_function(y, y_pred)
                    log_text += f' | {self.metric} = {metric:.2f}'
                print(log_text)

        if self.metric_function is not None:
            y_pred = X @ self.weights
            self.final_score = self.metric_function(y, y_pred)

    def predict(self, X_: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict target values using the trained model.

        Args:
            X_ (pd.DataFrame | np.ndarray): Feature matrix of shape
                (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if self.weights is None:
            raise NotFittedError()

        X = np.asarray(X_, dtype=float)

        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights

    def get_coef(self) -> np.ndarray:
        """Get the learned model coefficients (excluding bias).

        Returns:
            np.ndarray: Model coefficients of shape (n_features,).

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')

        return self.weights[1:]

    def get_final_score(self) -> float:
        """Get the final evaluation score after training.

        Returns:
            float: Final score computed using the selected metric.

        Raises:
            NotFittedError: If the model has not been fitted yet.
            RuntimeError: If no metric was selected during initialization.
        """
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')

        if self.final_score is None:
            raise RuntimeError('Metric was not selected during initialization.')

        return self.final_score


if __name__ == '__main__':
    X_demo = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
    y_demo = pd.Series(np.random.rand(100))

    model = LinearRegression(n_iter=50, learning_rate=0.1, metric='mse', verbose=10)
    model.fit(X_demo, y_demo)

    print('Coefficients:', model.get_coef())
    print('Final score:', model.get_final_score())
