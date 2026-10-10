"""`ModelLoaderRegistry.register` refuses anything but a `ModelLoader` subclass.

Insert a helper function between the decorator and the class it was meant to decorate — a plausible
outcome of extracting a loop into a named function — and a decorator without a type check binds the
helper instead. The module imports and the helper keeps working where it is called directly; the only
symptom is that `ModelLoadService.load_model` calls `implementation(app_config=..., ...)` and gets a
`TypeError`, i.e. that model type can no longer be loaded at all. This happened to
`Qwen3EncoderCheckpointLoader`.

A class built directly on `ModelLoaderBase` is refused too: `ModelLoader` is where a cold load is checked
against edits of its record, so such a loader could cache a model built from a superseded record.
"""

from typing import Optional

import pytest

from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


class _DirectLoader(ModelLoaderBase):
    def __init__(self, app_config, logger, ram_cache) -> None:
        pass

    @property
    def ram_cache(self):
        raise NotImplementedError

    def get_size_fs(self, config, model_path, submodel_type: Optional[SubModelType] = None) -> int:
        return 0

    def load_model(self, model_config, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        raise NotImplementedError


def _helper() -> None:
    pass


@pytest.mark.parametrize("implementation", [_helper, _DirectLoader], ids=["function", "direct-base-subclass"])
def test_register_refuses_anything_but_a_model_loader(implementation) -> None:
    before = dict(ModelLoaderRegistry._registry)

    with pytest.raises(TypeError, match="must subclass ModelLoader"):
        ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Main, format=ModelFormat.Diffusers)(
            implementation
        )

    assert ModelLoaderRegistry._registry == before
