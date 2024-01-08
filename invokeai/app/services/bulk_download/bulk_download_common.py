
class BulkDownloadException(Exception):
    """Exception raised when a bulk download fails."""

    def __init__(self, message="Bulk download failed"):
        super().__init__(message)
        self.message = message

class BulkDownloadTargetException(BulkDownloadException):
    """Exception raised when a bulk download target is not found."""

    def __init__(self, message="The bulk download target was not found"):
        super().__init__(message)
        self.message = message

class BulkDownloadParametersException(BulkDownloadException):
    """Exception raised when a bulk download parameter is invalid."""

    def __init__(self, message="The bulk download parameters are invalid, either an array of image names or a board id must be provided"):
        super().__init__(message)
        self.message = message