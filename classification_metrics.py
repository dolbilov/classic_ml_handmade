import numpy as np

class ClassificationMetrics:
    """
    Static utility class for evaluating binary classification models.
    Provides methods for calculating common performance metrics.
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the accuracy score.

        Args:
            y_true (np.ndarray): Array of true binary labels (0 or 1).
            y_pred (np.ndarray): Array of predicted binary labels (0 or 1).

        Returns:
            float: Proportion of correct predictions.
        """
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the precision score.

        Args:
            y_true (np.ndarray): Array of true binary labels.
            y_pred (np.ndarray): Array of predicted binary labels.

        Returns:
            float: Ratio of true positives to all predicted positives.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-10)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the recall score.

        Args:
            y_true (np.ndarray): Array of true binary labels.
            y_pred (np.ndarray): Array of predicted binary labels.

        Returns:
            float: Ratio of true positives to all actual positives.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-10)

    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the F1 score.

        Args:
            y_true (np.ndarray): Array of true binary labels.
            y_pred (np.ndarray): Array of predicted binary labels.

        Returns:
            float: Harmonic mean of precision and recall.
        """
        prec = ClassificationMetrics.precision(y_true, y_pred)
        rec = ClassificationMetrics.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 1e-10)

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate the ROC AUC score using a ranking-based approach.

        Args:
            y_true (np.ndarray): Array of true binary labels.
            y_pred_proba (np.ndarray): Array of predicted probabilities for the positive class.

        Returns:
            float: Area Under the ROC Curve (AUC).
        """
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

            # Number of positive samples with a higher score (to the left in the sorted list)
            higher_pos = np.sum(y_true_sorted[:i] == 1)

            # Number of positive samples with the SAME score (including current one, but it is negative)
            same_score_mask = y_pred_sorted == y_pred_sorted[i]
            same_pos = np.sum(y_true_sorted[same_score_mask] == 1)

            sum_positive_ranks += higher_pos + 0.5 * same_pos

        auc = (sum_positive_ranks) / (n_pos * n_neg)
        return auc

    @staticmethod
    def get_metric_by_name(name: str):
        """Retrieve a metric calculation function by its name.

        Args:
            name (str): Name of the metric ('accuracy', 'precision', 'recall', 'f1', 'roc_auc').

        Returns:
            Callable: Corresponding static method for metric calculation.

        Raises:
            KeyError: If the metric name is not recognized.
        """
        metric_function_map = {
            'accuracy': ClassificationMetrics.accuracy,
            'precision': ClassificationMetrics.precision,
            'recall': ClassificationMetrics.recall,
            'f1': ClassificationMetrics.f1,
            'roc_auc': ClassificationMetrics.roc_auc
        }

        return metric_function_map[name]
