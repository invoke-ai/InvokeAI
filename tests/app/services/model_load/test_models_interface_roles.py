"""`ModelsInterface` resolves a `ModelIdentifierField` to the record, or to its view in the role the identifier's type
names; `get_config` and `load` agree."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import ModelsInterface
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from tests.fixtures.model_roles import text_llm_file


def _identifier(model_type: ModelType, submodel_type: SubModelType | None = None) -> ModelIdentifierField:
    return ModelIdentifierField(
        key="llm", hash="h", name="llm", base=BaseModelType.Any, type=model_type, submodel_type=submodel_type
    )


@pytest.fixture
def models() -> ModelsInterface:
    records = {"llm": text_llm_file("llm")}
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=SimpleNamespace(get_model=lambda key: records[key]),
            load=SimpleNamespace(load_model=lambda config, submodel_type=None, user_id=None: (config, submodel_type)),
        )
    )
    data = SimpleNamespace(queue_item=SimpleNamespace(user_id="user"))
    return ModelsInterface(services, data, MagicMock())  # type: ignore[arg-type]


def test_an_identifier_of_the_records_own_type_or_a_key_resolves_to_the_stored_record(models: ModelsInterface) -> None:
    assert isinstance(models.get_config(_identifier(ModelType.TextLLM)), TextLLM_Checkpoint_Config)
    assert isinstance(models.get_config("llm"), TextLLM_Checkpoint_Config)


def test_an_identifier_of_a_role_the_record_serves_resolves_to_its_view(models: ModelsInterface) -> None:
    config = models.get_config(_identifier(ModelType.Qwen35Encoder))

    assert isinstance(config, Qwen35Encoder_Checkpoint_Config)
    assert config.role_of is ModelType.TextLLM


def test_an_identifier_of_a_type_the_record_does_not_serve_reaches_the_node_with_the_stored_record(
    models: ModelsInterface,
) -> None:
    config = models.get_config(_identifier(ModelType.Qwen3Encoder))

    assert isinstance(config, TextLLM_Checkpoint_Config)
    assert config.role_of is None


def test_load_loads_the_view_and_keeps_the_identifiers_submodel(models: ModelsInterface) -> None:
    config, submodel = models.load(_identifier(ModelType.Qwen35Encoder, SubModelType.Tokenizer))  # type: ignore[misc]

    assert isinstance(config, Qwen35Encoder_Checkpoint_Config) and config.role_of is ModelType.TextLLM
    assert submodel is SubModelType.Tokenizer


def test_load_by_key_loads_the_stored_record(models: ModelsInterface) -> None:
    config, _ = models.load("llm")  # type: ignore[misc]

    assert isinstance(config, TextLLM_Checkpoint_Config)
