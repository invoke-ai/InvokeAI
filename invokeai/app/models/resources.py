# TODO: Make a new model for this
from enum import Enum

from invokeai.app.util.enum import MetaEnum


class ResourceOrigin(str, Enum, metaclass=MetaEnum):
    """The origin of a resource (eg image or tensor)."""

    RESULTS = "results"
    UPLOADS = "uploads"
    INTERMEDIATES = "intermediates"


class ImageKind(str, Enum, metaclass=MetaEnum):
    """The kind of an image."""

    IMAGE = "image"
    CONTROL_IMAGE = "control_image"


class TensorKind(str, Enum, metaclass=MetaEnum):
    """The kind of a tensor."""

    IMAGE_TENSOR = "tensor"
    CONDITIONING = "conditioning"
