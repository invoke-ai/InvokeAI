class VideoFileNotFoundException(Exception):
    """Raised when a video file is not found in storage."""

    def __init__(self, message="Video file not found"):
        super().__init__(message)


class VideoFileSaveException(Exception):
    """Raised when a video file cannot be saved."""

    def __init__(self, message="Video file not saved"):
        super().__init__(message)


class VideoFileDeleteException(Exception):
    """Raised when a video file cannot be deleted."""

    def __init__(self, message="Video file not deleted"):
        super().__init__(message)
