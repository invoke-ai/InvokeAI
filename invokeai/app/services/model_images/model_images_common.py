# TODO: Should these excpetions subclass existing python exceptions?
class ModelImageFileNotFoundException(Exception):
    """Raised when an image file is not found in storage."""

    def __init__(self, message="Image file not found"):
        super().__init__(message)


class ModelImageFileSaveException(Exception):
    """Raised when an image cannot be saved."""

    def __init__(self, message="Image file not saved"):
        super().__init__(message)


class ModelImageFileDeleteException(Exception):
    """Raised when an image cannot be deleted."""

    def __init__(self, message="Image file not deleted"):
        super().__init__(message)
