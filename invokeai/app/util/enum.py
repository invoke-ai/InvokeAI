from enum import EnumMeta


class MetaEnum(EnumMeta):
    """Metaclass to support `in` syntax value checking in String Enums"""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
