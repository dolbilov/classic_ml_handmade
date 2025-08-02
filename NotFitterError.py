class NotFittedError(Exception):
    def __init__(self, message: str | None) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message or 'Model not fitted. Use "fit" method first.'
