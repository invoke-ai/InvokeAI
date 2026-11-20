"""Records that serve as another model type (`Config_Base.roles`), and a context that resolves them like the real one."""

from types import SimpleNamespace
from typing import Mapping
from unittest.mock import MagicMock

from invokeai.app.services.shared.invocation_context import ModelsInterface
from invokeai.backend.model_manager.configs.base import ModelRole
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config, TextLLM_Diffusers_Config
from invokeai.backend.model_manager.taxonomy import (
    ModelFormat,
    ModelSourceType,
    ModelType,
    Qwen3VariantType,
    Qwen35VariantType,
)


def _common(key: str, name: str) -> dict:
    return {
        "key": key,
        "hash": f"blake3:{key}",
        "path": f"llm/{key}",
        "file_size": 1024,
        "name": name,
        "source": "test",
        "source_type": ModelSourceType.Path,
    }


def text_llm_file(key: str, serves_encoder: Qwen35VariantType | None = Qwen35VariantType.Qwen35_9B):
    """A whole Qwen3.5 VLM file filed as a Text LLM; it serves as the Qwen3.5 encoder of `serves_encoder`, if any."""
    roles = [] if serves_encoder is None else [ModelRole(type=ModelType.Qwen35Encoder, variant=serves_encoder)]
    return TextLLM_Checkpoint_Config(**_common(key, f"{key} llm"), roles=roles)


def encoder_file(key: str, variant: Qwen35VariantType = Qwen35VariantType.Qwen35_9B, serves_text_llm: bool = True):
    """A Qwen3.5 encoder record; it serves as a Text LLM when the file is a whole model."""
    roles = [ModelRole(type=ModelType.TextLLM)] if serves_text_llm else []
    return Qwen35Encoder_Checkpoint_Config(**_common(key, f"{key} encoder"), variant=variant, roles=roles)


def text_llm_qwen3_folder(key: str, variant: Qwen3VariantType | None = Qwen3VariantType.Qwen3_4B):
    """A complete Qwen3 causal-LM folder filed as a Text LLM; it serves as the Qwen3 encoder of `variant`, if any."""
    roles = (
        []
        if variant is None
        else [ModelRole(type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder, variant=variant)]
    )
    return TextLLM_Diffusers_Config(**_common(key, f"{key} folder"), roles=roles)


def models_interface(configs: Mapping[str, AnyModelConfig]) -> ModelsInterface:
    """The real `ModelsInterface` over a store holding `configs`, so identifiers resolve as in a running graph."""
    store = SimpleNamespace(get_model=lambda key: configs[key])
    services = SimpleNamespace(model_manager=SimpleNamespace(store=store))
    return ModelsInterface(services, MagicMock(), MagicMock())  # type: ignore[arg-type]


def node_context(configs: Mapping[str, AnyModelConfig]) -> SimpleNamespace:
    return SimpleNamespace(models=models_interface(configs), logger=MagicMock())
