class NotFittedError(Exception):
    """Exception raised when an estimator is used before fitting.

    This error is raised when calling methods that require a fitted model
    (e.g., `predict`, `transform`, `score`) before `fit` has been called.

    Parameters
    ----------
    message : str, optional
        Custom error message. If not provided, a default message is used.
    """
    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or 'Model not fitted. Use "fit" method first.')
