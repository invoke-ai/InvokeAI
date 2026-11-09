"""Model roles: `view_as` turns a record into the config of another type its files serve as."""

import json

import pytest

from invokeai.backend.model_manager.configs.base import ModelRole
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
)
from invokeai.backend.model_manager.configs.roles import ModelRoleError, role_variant, serves_as, view_as
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config, TextLLM_Diffusers_Config
from invokeai.backend.model_manager.taxonomy import (
    ModelFormat,
    ModelSourceType,
    ModelType,
    Qwen3VariantType,
    Qwen35VariantType,
)

_COMMON = {
    "key": "key1",
    "hash": "blake3:abc",
    "path": "llm/model",
    "file_size": 1024,
    "name": "A model",
    "source": "test",
    "source_type": ModelSourceType.Path,
}


def _text_llm_file() -> TextLLM_Checkpoint_Config:
    return TextLLM_Checkpoint_Config(
        **_COMMON,
        cpu_only=True,
        roles=[ModelRole(type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_9B)],
    )


def _text_llm_folder() -> TextLLM_Diffusers_Config:
    return TextLLM_Diffusers_Config(
        **_COMMON,
        roles=[
            ModelRole(type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder, variant=Qwen3VariantType.Qwen3_4B)
        ],
    )


def _dumped(roles: list[ModelRole]) -> list[dict]:
    return [role.model_dump() for role in roles]


def test_a_text_llm_file_viewed_as_the_encoder_is_an_encoder_config_of_the_same_files() -> None:
    record = _text_llm_file()

    view = view_as(record, ModelType.Qwen35Encoder)

    assert isinstance(view, Qwen35Encoder_Checkpoint_Config)
    assert (view.key, view.path, view.hash, view.name, view.cpu_only) == (
        "key1",
        "llm/model",
        "blake3:abc",
        "A model",
        True,
    )
    assert view.variant is Qwen35VariantType.Qwen35_9B
    assert view.role_of is ModelType.TextLLM
    assert record.role_of is None
    assert _dumped(view.roles) == _dumped([ModelRole(type=ModelType.TextLLM)])


def test_a_folder_viewed_as_the_qwen3_encoder_switches_format_and_remembers_its_own() -> None:
    view = view_as(_text_llm_folder(), ModelType.Qwen3Encoder)

    assert isinstance(view, Qwen3Encoder_Qwen3Encoder_Config)
    assert view.format is ModelFormat.Qwen3Encoder
    assert view.variant is Qwen3VariantType.Qwen3_4B
    assert _dumped(view.roles) == _dumped([ModelRole(type=ModelType.TextLLM, format=ModelFormat.Diffusers)])
    back = view_as(view, ModelType.TextLLM)
    assert isinstance(back, TextLLM_Diffusers_Config)
    assert back.format is ModelFormat.Diffusers


def test_a_roles_variant_wins_over_the_primary_one() -> None:
    record = Qwen3Encoder_Checkpoint_Config(
        **_COMMON,
        variant=Qwen3VariantType.Qwen3_4B,
        roles=[ModelRole(type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_9B)],
    )

    view = view_as(record, ModelType.Qwen35Encoder)

    assert isinstance(view, Qwen35Encoder_Checkpoint_Config)
    assert view.variant is Qwen35VariantType.Qwen35_9B


def test_the_primary_variant_is_not_carried_into_a_role_that_names_none(monkeypatch: pytest.MonkeyPatch) -> None:
    record = Qwen3Encoder_Checkpoint_Config(
        **_COMMON, variant=Qwen3VariantType.Qwen3_4B, roles=[ModelRole(type=ModelType.TextLLM)]
    )
    built: list[dict] = []
    from_dict = ModelConfigFactory.from_dict
    monkeypatch.setattr(ModelConfigFactory, "from_dict", lambda fields: built.append(dict(fields)) or from_dict(fields))

    view_as(record, ModelType.TextLLM)

    assert "variant" not in built[0]


def test_a_role_without_the_variant_its_class_needs_is_refused() -> None:
    record = TextLLM_Checkpoint_Config(**_COMMON, roles=[ModelRole(type=ModelType.Qwen35Encoder)])

    with pytest.raises(ModelRoleError):
        view_as(record, ModelType.Qwen35Encoder)


def test_a_record_viewed_as_its_own_type_is_itself() -> None:
    record = _text_llm_file()

    assert view_as(record, ModelType.TextLLM) is record


def test_a_role_the_model_does_not_serve_is_refused_naming_both_types() -> None:
    with pytest.raises(ModelRoleError, match="A model.*text_llm.*qwen3_encoder"):
        view_as(_text_llm_file(), ModelType.Qwen3Encoder)


def test_serves_as_and_role_variant_cover_the_primary_type_a_role_and_neither() -> None:
    record = _text_llm_file()

    assert serves_as(record, ModelType.TextLLM) and serves_as(record, ModelType.Qwen35Encoder)
    assert not serves_as(record, ModelType.Qwen3Encoder)
    assert role_variant(record, ModelType.Qwen35Encoder) is Qwen35VariantType.Qwen35_9B
    assert role_variant(record, ModelType.TextLLM) is None
    assert role_variant(record, ModelType.Qwen3Encoder) is None
    encoder = view_as(record, ModelType.Qwen35Encoder)
    assert role_variant(encoder, ModelType.Qwen35Encoder) is Qwen35VariantType.Qwen35_9B


def test_roles_survive_the_json_round_trip_of_a_stored_record() -> None:
    record = _text_llm_folder()

    restored = ModelConfigFactory.from_dict(json.loads(record.model_dump_json()))

    assert isinstance(restored, TextLLM_Diffusers_Config)
    assert _dumped(restored.roles) == _dumped(record.roles)


def test_a_role_view_never_dumps_its_origin() -> None:
    view = view_as(_text_llm_file(), ModelType.Qwen35Encoder)

    assert "_role_of" not in view.model_dump()
    assert "role_of" not in json.loads(view.model_dump_json())


def test_a_role_this_build_cannot_read_is_dropped_and_the_record_survives() -> None:
    fields = _text_llm_file().model_dump(mode="json")
    fields["roles"] = [
        {"type": "a_future_type"},
        {"type": "qwen3_5_encoder", "variant": "a_future_size"},
        {"type": "qwen3_5_encoder", "variant": "qwen3_5_9b", "a_future_field": 1},
    ]

    restored = ModelConfigFactory.from_dict(fields)

    assert _dumped(restored.roles) == _dumped(
        [ModelRole(type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_9B)]
    )
