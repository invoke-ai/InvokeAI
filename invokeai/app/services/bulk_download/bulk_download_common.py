from enum import Enum

DEFAULT_BULK_DOWNLOAD_ID = "default"


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

    def __init__(self, message="No image names or board ID provided"):
        super().__init__(message)
        self.message = message


class BulkDownloadBusyError(BulkDownloadException):
    """Exception raised when a 16-bit download is requested while another one is being built."""

    def __init__(self, message="Another 16-bit bulk download is being prepared"):
        super().__init__(message)
        self.message = message


class BulkDownloadStatus(str, Enum):
    """Where a download stands, as the fetch endpoint reports it."""

    READY = "ready"
    BUILDING = "building"
    FAILED = "failed"
    UNKNOWN = "unknown"
