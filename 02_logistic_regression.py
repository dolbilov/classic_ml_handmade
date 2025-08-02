import numpy as np
import pandas as pd
from NotFittedError import NotFittedError


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __get_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
            grad = 1 / X.shape[0] * (y_pred - y) @ X
            self.weights -= self.learning_rate * grad

            if verbose and (i == 1 or i % verbose == 0):
                log_text = f'[{i}/{self.n_iter}] | loss = {log_loss:.2f}'
                print(log_text)

    def predict(self, X_: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_)
        return probs > 0.5

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
