class ObjectNotFoundError(KeyError):
    """Raised when an object is not found while loading"""

    def __init__(self, name: str) -> None:
        super().__init__(f"Object with name {name} not found")


class InvalidObjectNameError(ValueError):
    """Raised when an object name is not a plain filename (e.g. it contains path separators)"""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid object name {name!r}, potential directory traversal detected")
