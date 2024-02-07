class ItemNotFoundError(KeyError):
    """Raised when an item is not found in storage"""

    def __init__(self, item_id: str) -> None:
        super().__init__(f"Item with id {item_id} not found")
