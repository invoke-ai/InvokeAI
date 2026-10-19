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


# The three keys a media file carries its generation record under — PNG text chunks, MP4 keyed
# metadata and the legacy JSON sidecar all use the same names.
METADATA_KEY = "invokeai_metadata"
WORKFLOW_KEY = "invokeai_workflow"
GRAPH_KEY = "invokeai_graph"
EMBEDDED_METADATA_KEYS = (METADATA_KEY, WORKFLOW_KEY, GRAPH_KEY)
