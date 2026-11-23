import pytest
from pydantic import ValidationError

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings


def test_accepts_a_map_of_component_slots_to_model_keys() -> None:
    components = {"t5_encoder": "key-a", "qwen3_encoder": "key-b", "pid_decoder2": "key-c"}
    assert MainModelDefaultSettings(components=components).components == components


def test_components_are_optional() -> None:
    assert MainModelDefaultSettings().components is None
    assert MainModelDefaultSettings(components={}).components is None


@pytest.mark.parametrize("slot", ["", "T5Encoder", "t5-encoder", "t5 encoder", "t5_encoder\n", "köder"])
def test_rejects_a_slot_that_is_not_snake_case(slot: str) -> None:
    with pytest.raises(ValidationError, match="snake_case"):
        MainModelDefaultSettings(components={slot: "key"})


def test_rejects_the_vae_slot_because_the_vae_field_owns_it() -> None:
    with pytest.raises(ValidationError, match="`vae` field"):
        MainModelDefaultSettings(components={"vae": "key"})


@pytest.mark.parametrize("model_key", ["", "   "])
def test_rejects_an_empty_model_key(model_key: str) -> None:
    with pytest.raises(ValidationError, match="needs a model key"):
        MainModelDefaultSettings(components={"t5_encoder": model_key})


def test_allows_sixteen_components_and_rejects_seventeen() -> None:
    assert len(MainModelDefaultSettings(components={f"slot_{i}": "key" for i in range(16)}).components or {}) == 16
    with pytest.raises(ValidationError, match="at most 16"):
        MainModelDefaultSettings(components={f"slot_{i}": "key" for i in range(17)})


def test_unknown_top_level_fields_stay_forbidden() -> None:
    with pytest.raises(ValidationError):
        MainModelDefaultSettings(component={"t5_encoder": "key"})  # type: ignore[call-arg]


def test_a_record_without_components_dumps_without_the_key() -> None:
    """An older server forbids unknown fields, so even a null `components` would get the record skipped."""
    assert "components" not in MainModelDefaultSettings(steps=20).model_dump()
    assert "components" not in MainModelDefaultSettings(steps=20).model_dump_json()
    assert MainModelDefaultSettings(components={"t5_encoder": "k"}).model_dump()["components"] == {"t5_encoder": "k"}
