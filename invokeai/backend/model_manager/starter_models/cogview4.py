"""CogView 4 starter models."""

from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)

# region CogView4
cogview4 = StarterModel(
    source_urls=["https://huggingface.co/THUDM/CogView4-6B"],
    name="CogView4",
    base=BaseModelType.CogView4,
    source="THUDM/CogView4-6B",
    description="The base CogView4 model (~31GB).",
    type=ModelType.Main,
)
