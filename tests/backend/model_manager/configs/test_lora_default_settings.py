import pytest
from pydantic import ValidationError

from invokeai.backend.model_manager.configs.lora import LoraModelDefaultSettings


def test_accepts_none_for_all_fields() -> None:
    settings = LoraModelDefaultSettings()
    assert settings.weight is None
    assert settings.weight_min is None
    assert settings.weight_max is None


def test_accepts_both_bounds_with_min_less_than_max() -> None:
    settings = LoraModelDefaultSettings(weight_min=-2.0, weight_max=3.0)
    assert settings.weight_min == -2.0
    assert settings.weight_max == 3.0


def test_rejects_both_bounds_with_min_greater_than_or_equal_to_max() -> None:
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_min=2.0, weight_max=2.0)

    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_min=3.0, weight_max=1.0)


def test_accepts_only_weight_min_within_default_range() -> None:
    # Default max is 2.0; a weight_min of 1.0 leaves a valid effective range [1.0, 2.0].
    settings = LoraModelDefaultSettings(weight_min=1.0)
    assert settings.weight_min == 1.0
    assert settings.weight_max is None


def test_rejects_only_weight_min_above_default_max() -> None:
    # Reproduces the partial-bound bug: saving only weight_min=3 used to be accepted but
    # rendered as a min>max slider in LoRACard. The effective max defaults to 2.0.
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_min=3.0)


def test_rejects_only_weight_min_equal_to_default_max() -> None:
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_min=2.0)


def test_accepts_only_weight_max_within_default_range() -> None:
    # Default min is -1.0; a weight_max of 0.5 leaves a valid effective range [-1.0, 0.5].
    settings = LoraModelDefaultSettings(weight_max=0.5)
    assert settings.weight_max == 0.5
    assert settings.weight_min is None


def test_rejects_only_weight_max_below_default_min() -> None:
    # Reproduces the symmetric partial-bound bug for weight_max <= -1.
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_max=-2.0)


def test_rejects_only_weight_max_equal_to_default_min() -> None:
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight_max=-1.0)


def test_accepts_weight_within_effective_range_with_explicit_bounds() -> None:
    settings = LoraModelDefaultSettings(weight=0.5, weight_min=-1.0, weight_max=2.0)
    assert settings.weight == 0.5


def test_accepts_weight_within_default_effective_range() -> None:
    # With no bounds set the effective range is [-1.0, 2.0].
    settings = LoraModelDefaultSettings(weight=1.5)
    assert settings.weight == 1.5


def test_rejects_weight_above_explicit_effective_range() -> None:
    # Reproduces the out-of-range-weight bug: weight=10 with bounds [-1, 2] used to be
    # accepted but the slider in LoRACard could only show [-1, 2].
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight=10.0, weight_min=-1.0, weight_max=2.0)


def test_rejects_weight_below_explicit_effective_range() -> None:
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight=-5.0, weight_min=-1.0, weight_max=2.0)


def test_rejects_weight_above_default_effective_range() -> None:
    # No bounds set; default max is 2.0.
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight=2.5)


def test_rejects_weight_outside_partial_bound() -> None:
    # Only weight_min set; weight must respect effective range [weight_min, default_max].
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight=0.0, weight_min=0.5)

    # Only weight_max set; weight must respect effective range [default_min, weight_max].
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(weight=1.0, weight_max=0.5)


def test_accepts_weight_at_effective_bounds() -> None:
    # Inclusive bounds.
    LoraModelDefaultSettings(weight=-1.0, weight_min=-1.0, weight_max=2.0)
    LoraModelDefaultSettings(weight=2.0, weight_min=-1.0, weight_max=2.0)


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LoraModelDefaultSettings(extra_field=True)  # type: ignore[call-arg]
