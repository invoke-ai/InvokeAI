# TODO: Make a new model for this
from enum import Enum

from invokeai.app.util.enum import MetaEnum


class ResourceType(str, Enum, metaclass=MetaEnum):
    """The type of a resource."""

    IMAGES = "images"
    TENSORS = "tensors"


# class ResourceOrigin(str, Enum, metaclass=MetaEnum):
#     """The origin of a resource (eg image or tensor)."""

#     RESULTS = "results"
#     UPLOADS = "uploads"
#     INTERMEDIATES = "intermediates"



class TensorKind(str, Enum, metaclass=MetaEnum):
    """The kind of a tensor. Use TensorKind.OTHER for non-default kinds."""

    IMAGE_LATENTS = "image_latents"
    CONDITIONING = "conditioning"
    OTHER = "other"
