from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.ltx2.duration_head_state_dict_utils import is_state_dict_likely_ltx2_duration_head
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


class LTX2DurationHead_Checkpoint_Config(Config_Base):
    """Model config for the LTX-2 duration head, which predicts a shot's natural length from a prompt."""

    type: Literal[ModelType.LTX2DurationHead] = Field(default=ModelType.LTX2DurationHead)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.LTX2] = Field(default=BaseModelType.LTX2)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if not is_state_dict_likely_ltx2_duration_head(mod.load_state_dict()):
            raise NotAMatchError("model does not match LTX-2 duration head heuristics")

        return cls(**override_fields)
