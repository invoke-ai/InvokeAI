# TODO: Should these excpetions subclass existing python exceptions?
class ImageFileNotFoundException(Exception):
    """Raised when an image file is not found in storage."""

    def __init__(self, message="Image file not found"):
        super().__init__(message)


class ImageFileSaveException(Exception):
    """Raised when an image cannot be saved."""

    def __init__(self, message="Image file not saved"):
        super().__init__(message)


class ImageFileDeleteException(Exception):
    """Raised when an image cannot be deleted."""

    def __init__(self, message="Image file not deleted"):
        super().__init__(message)
