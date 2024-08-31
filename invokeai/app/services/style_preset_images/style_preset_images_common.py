class StylePresetImageFileNotFoundException(Exception):
    """Raised when an image file is not found in storage."""

    def __init__(self, message: str = "Style preset image file not found"):
        super().__init__(message)


class StylePresetImageFileSaveException(Exception):
    """Raised when an image cannot be saved."""

    def __init__(self, message: str = "Style preset image file not saved"):
        super().__init__(message)


class StylePresetImageFileDeleteException(Exception):
    """Raised when an image cannot be deleted."""

    def __init__(self, message: str = "Style preset image file not deleted"):
        super().__init__(message)
