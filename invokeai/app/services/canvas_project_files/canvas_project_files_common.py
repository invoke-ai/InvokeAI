class CanvasProjectFileNotFoundException(Exception):
    """Raised when a canvas project file is not found in storage."""

    def __init__(self, message="Canvas project file not found"):
        super().__init__(message)


class CanvasProjectFileSaveException(Exception):
    """Raised when a canvas project file cannot be saved."""

    def __init__(self, message="Canvas project file not saved"):
        super().__init__(message)


class CanvasProjectFileDeleteException(Exception):
    """Raised when a canvas project file cannot be deleted."""

    def __init__(self, message="Canvas project file not deleted"):
        super().__init__(message)
