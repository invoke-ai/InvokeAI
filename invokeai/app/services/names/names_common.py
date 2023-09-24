from enum import Enum, EnumMeta


class ResourceType(str, Enum, metaclass=EnumMeta):
    """Enum for resource types."""

    IMAGE = "image"
    LATENT = "latent"
