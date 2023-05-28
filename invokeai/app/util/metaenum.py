from enum import EnumMeta


class MetaEnum(EnumMeta):
    """Metaclass to support additional features in Enums.

    - `in` operator support: `'value' in MyEnum -> bool`
    """

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
