class ObjectNotFoundError(KeyError):
    """Raised when an object is not found while loading"""

    def __init__(self, name: str) -> None:
        super().__init__(f"Object with name {name} not found")
