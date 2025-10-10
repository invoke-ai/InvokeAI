from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel


class Spandrel_Checkpoint_Config(Config_Base):
    """Model config for Spandrel Image to Image models."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.SpandrelImageToImage] = Field(default=ModelType.SpandrelImageToImage)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_spandrel_loads_model(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_spandrel_loads_model(cls, mod: ModelOnDisk) -> None:
        try:
            # It would be nice to avoid having to load the Spandrel model from disk here. A couple of options were
            # explored to avoid this:
            # 1. Call `SpandrelImageToImageModel.load_from_state_dict(ckpt)`, where `ckpt` is a state_dict on the meta
            #    device. Unfortunately, some Spandrel models perform operations during initialization that are not
            #    supported on meta tensors.
            # 2. Spandrel has internal logic to determine a model's type from its state_dict before loading the model.
            #    This logic is not exposed in spandrel's public API. We could copy the logic here, but then we have to
            #    maintain it, and the risk of false positive detections is higher.
            SpandrelImageToImageModel.load_from_file(mod.path)
        except Exception as e:
            raise NotAMatchError("model does not match SpandrelImageToImage heuristics") from e
