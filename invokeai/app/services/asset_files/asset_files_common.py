class AssetFileNotFoundException(Exception):
    """Raised when a 3D asset file is not found"""

    def __init__(self, message: str = "Asset file not found"):
        self.message = message
        super().__init__(self.message)


class AssetFileSaveException(Exception):
    """Raised when a 3D asset file cannot be saved"""

    def __init__(self, message: str = "Asset file cannot be saved"):
        self.message = message
        super().__init__(self.message)


class AssetFileDeleteException(Exception):
    """Raised when a 3D asset file cannot be deleted"""

    def __init__(self, message: str = "Asset file cannot be deleted"):
        self.message = message
        super().__init__(self.message)
