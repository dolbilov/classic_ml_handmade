import numpy as np


class RegressionMetrics:
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.abs(y_true - y_pred).mean()

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.square(y_true - y_pred).mean()

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return RegressionMetrics.mean_squared_error(y_true, y_pred) ** 0.5

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = y_true.mean()
        return 1 - np.square(y_true - y_pred).mean() / np.square(y_true - y_mean).mean()

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        err = np.abs((y_true - y_pred) / y_true).mean()
        return 100 * err

    @staticmethod
    def get_metric_by_name(name: str):
        metric_function_map = {
            'mae': RegressionMetrics.mean_absolute_error,
            'mse': RegressionMetrics.mean_squared_error,
            'rmse': RegressionMetrics.root_mean_squared_error,
            'mape': RegressionMetrics.mean_absolute_percentage_error,
            'r2': RegressionMetrics.r2
        }

        return metric_function_map[name]
