from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.flux.redux.flux_redux_state_dict_utils import is_state_dict_likely_flux_redux
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


class FLUXRedux_Checkpoint_Config(Config_Base):
    """Model config for FLUX Tools Redux model."""

    type: Literal[ModelType.FluxRedux] = Field(default=ModelType.FluxRedux)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if not is_state_dict_likely_flux_redux(mod.load_state_dict()):
            raise NotAMatchError("model does not match FLUX Tools Redux heuristics")

        return cls(**override_fields)
