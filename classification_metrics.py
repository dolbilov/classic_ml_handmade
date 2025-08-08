import numpy as np


class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-10)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-10)

    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        prec = ClassificationMetrics.precision(y_true, y_pred)
        rec = ClassificationMetrics.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 1e-10)

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred_proba = y_pred_proba.round(10)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred_proba[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        sum_positive_ranks = 0

        for i, temp_y_true in enumerate(y_true_sorted):
            if temp_y_true == 1:
                continue

            # Число положительных с бОльшим score (левее в отсортированном списке)
            higher_pos = np.sum(y_true_sorted[:i] == 1)

            # Число положительных с ТАКИМ ЖЕ score (включая текущий, но он negative)
            same_score_mask = y_pred_sorted == y_pred_sorted[i]
            same_pos = np.sum(y_true_sorted[same_score_mask] == 1)

            sum_positive_ranks += higher_pos + 0.5 * same_pos

        auc = (sum_positive_ranks) / (n_pos * n_neg)
        return auc

    @staticmethod
    def get_metric_by_name(name: str):
        metric_function_map = {
            'accuracy': ClassificationMetrics.accuracy,
            'precision': ClassificationMetrics.precision,
            'recall': ClassificationMetrics.recall,
            'f1': ClassificationMetrics.f1,
            'roc_auc': ClassificationMetrics.roc_auc
        }

        return metric_function_map[name]
