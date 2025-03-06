class WorkflowThumbnailFileNotFoundException(Exception):
    """Raised when a workflow thumbnail file is not found"""

    def __init__(self, message: str = "Workflow thumbnail file not found"):
        self.message = message
        super().__init__(self.message)


class WorkflowThumbnailFileSaveException(Exception):
    """Raised when a workflow thumbnail file cannot be saved"""

    def __init__(self, message: str = "Workflow thumbnail file cannot be saved"):
        self.message = message
        super().__init__(self.message)


class WorkflowThumbnailFileDeleteException(Exception):
    """Raised when a workflow thumbnail file cannot be deleted"""

    def __init__(self, message: str = "Workflow thumbnail file cannot be deleted"):
        self.message = message
        super().__init__(self.message)
