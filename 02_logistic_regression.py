import random
from typing import Literal, Callable
import numpy as np
import pandas as pd
from classification_metrics import ClassificationMetrics
from NotFittedError import NotFittedError


class LogisticRegression:
    """Custom implementation of Logistic Regression with optional regularization and SGD mini-batching.

    This implementation supports L1, L2, and ElasticNet regularization, different learning
    rate strategies, stochastic gradient descent, and tracking various classification metrics.
    """

    EPSILON = 1e-15  # Small constant for numerical stability

    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float | Callable[[int], float] = 0.1,
        metric: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] | None = None,
        reg: Literal['l1', 'l2', 'elasticnet'] | None = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        sgd_sample: int | float | None = None,
        random_state: int = 42,
        threshold: float = 0.5,
        verbose: int = 0
    ) -> None:
        """Initialize the logistic regression model.

        Args:
            n_iter: Number of training iterations.
            learning_rate: Constant learning rate or callable mapping iteration -> learning rate.
            metric: Name of the evaluation metric to track during training.
            reg: Regularization type ('l1', 'l2', 'elasticnet', or None).
            l1_coef: L1 regularization strength.
            l2_coef: L2 regularization strength.
            sgd_sample: Mini-batch size (int for number of samples, float for fraction, None for full batch).
            random_state: Random seed for reproducibility.
            threshold: Probability threshold for binary classification.
            verbose: Frequency (in iterations) of printing training progress. 0 to disable.
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.metric_function = (
            ClassificationMetrics.get_metric_by_name(metric) if metric else None
        )
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.threshold = threshold
        self.verbose = verbose

        self.weights: np.ndarray | None = None
        self.final_score: float | None = None

        random.seed(random_state)
        np.random.seed(random_state)

    def __repr__(self) -> str:
        return (f'LogisticRegression(n_iter={self.n_iter}, learning_rate={self.learning_rate}, '
                f'metric={self.metric}, reg={self.reg}, '
                f'l1_coef={self.l1_coef}, l2_coef={self.l2_coef}, '
                f'verbose={self.verbose})')

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _get_learning_rate(self, iteration_number: int) -> float:
        return (
            self.learning_rate if isinstance(self.learning_rate, (int, float)) else self.learning_rate(iteration_number)
        )

    def _get_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')

        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

        if self.reg in ('l1', 'elasticnet'):
            loss += self.l1_coef * np.abs(self.weights).sum()

        if self.reg in ('l2', 'elasticnet'):
            loss += self.l2_coef * np.square(self.weights).sum()

        return loss

    def _get_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')

        grad = (y_pred - y_true) @ X / X.shape[0]

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
        """Train the logistic regression model.

        Args:
            X_: Feature matrix of shape (n_samples, n_features).
            y_: Target vector of shape (n_samples,).

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        X = np.asarray(X_, dtype=float)
        y = np.asarray(y_, dtype=float)

        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in X and y must match.')

        sgd_sample_k = self._get_sgd_sample_k(X.shape[0])
        X = np.insert(X, 0, 1, axis=1)  # bias term
        self.weights = np.zeros(X.shape[1])

        for i in range(1, self.n_iter + 1):
            batch_idx = np.random.choice(X.shape[0], sgd_sample_k, replace=False)
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            y_pred_batch = self._sigmoid(X_batch @ self.weights)
            grad = self._get_grad(X_batch, y_batch, y_pred_batch)
            lr = self._get_learning_rate(i)

            self.weights -= lr * grad

            if self.verbose and (i == 1 or i % self.verbose == 0):
                y_pred_full = self._sigmoid(X @ self.weights)
                loss = self._get_log_loss(y, y_pred_full)
                log_text = f'[{i}/{self.n_iter}] | loss = {loss:.4f}'
                if self.metric_function:
                    preds = y_pred_full if self.metric == 'roc_auc' else (y_pred_full > self.threshold).astype(int)
                    metric_val = self.metric_function(y, preds)
                    log_text += f' | {self.metric} = {metric_val:.4f}'
                print(log_text)

        if self.metric_function:
            y_pred_full = self._sigmoid(X @ self.weights)
            preds = y_pred_full if self.metric == 'roc_auc' else (y_pred_full > self.threshold).astype(int)
            self.final_score = self.metric_function(y, preds)

    def predict_proba(self, X_: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities for input samples.

        Args:
            X_: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of probabilities for the positive class.

        Raises:
            NotFittedError: If model is not fitted yet.
        """
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')

        X = np.asarray(X_, dtype=float)
        X = np.insert(X, 0, 1, axis=1)
        return self._sigmoid(X @ self.weights)

    def predict(self, X_: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary class labels for input samples.

        Args:
            X_: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        return (self.predict_proba(X_) > self.threshold).astype(int)

    def get_coef(self) -> np.ndarray:
        """Get the model coefficients (excluding bias term).

        Returns:
            np.ndarray: Coefficients of shape (n_features,).

        Raises:
            NotFittedError: If model is not fitted yet.
        """
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')
        return self.weights[1:]

    def get_final_score(self) -> float:
        """Get the final evaluation score after training.

        Returns:
            float: Final score based on the chosen metric.

        Raises:
            NotFittedError: If model is not fitted yet.
            RuntimeError: If no metric was specified during initialization.
        """
        if self.weights is None:
            raise NotFittedError('Model is not fitted yet.')
        if self.final_score is None:
            raise RuntimeError('Metric was not selected during initialization.')
        return self.final_score


if __name__ == '__main__':
    X_demo = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
    y_demo = pd.Series(np.random.randint(0, 2, size=100))

    model = LogisticRegression(n_iter=50, learning_rate=0.1, metric='accuracy', verbose=10)
    model.fit(X_demo, y_demo)

    print('Coefficients:', model.get_coef())
    print('Final score:', model.get_final_score())
