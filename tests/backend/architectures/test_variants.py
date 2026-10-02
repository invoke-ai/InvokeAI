"""CI guards for the variant plumbing.

One list of variant enums is written by hand four times -- `AnyVariant`, the `variant_type_adapter`
subscript, the adapter's runtime argument (all three in `taxonomy.py`) and
`ModelRecordChanges.variant`. Nothing checked that they agreed, and nothing checked the invariant
they all rest on: that variant *values* are globally unique, because `configs/factory.py` resolves a
bare variant string without knowing the base.

That invariant is real and has already cost something -- it is why `Krea2VariantType.Turbo` is
``krea2_turbo`` rather than ``turbo``. Until now it lived only in a docstring.
"""

import typing
from enum import Enum

import pytest

from invokeai.backend.architectures import declared_variant_enums, get_variant_enum
from invokeai.backend.architectures.registry import _NOT_ARCHITECTURES
from invokeai.backend.model_manager import taxonomy
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig  # noqa: F401  (registers every config class)
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    ClipVariantType,
    MistralVariantType,
    ModelType,
    Qwen3VariantType,
    Qwen3VLVariantType,
    variant_type_adapter,
)

BASE_AGNOSTIC_VARIANT_ENUMS = frozenset({ClipVariantType, Qwen3VariantType, Qwen3VLVariantType, MistralVariantType})
"""Variant enums that cannot be declared by any architecture.

All four sit on `base=Any` configs -- CLIP embedders and the Qwen3, Qwen3-VL and Mistral text
encoders are components shared across architectures, not architectures. `Any` is a sentinel the
registry refuses to register, so these are named here to keep the completeness check against
`AnyVariant` total rather than quietly partial.
"""


KNOWN_VALUE_COLLISIONS = ["'dev': Flux2VariantType.Dev vs FluxVariantType.Dev"]
"""Value collisions that exist and are tolerated, listed so a *new* one still fails.

FLUX.1 and FLUX.2 both call their full model `dev`, and both enums are `str` mixins, so the members
compare equal and every `variant != Flux2VariantType.Dev` guard still answers correctly. What the
collision does cost is the enum *class*: `variant_type_adapter` resolves a bare ``'dev'`` to
`FluxVariantType`, so an `isinstance(variant, Flux2VariantType)` check would be False for a FLUX.2
dev model. No such check exists today (the three in the tree name Qwen3, MiniMax H3 and Wan), which
is why this is tolerated rather than fixed -- renaming the value would need a migration of persisted
model records.
"""


def _variant_enums_in_taxonomy() -> set[type[Enum]]:
    """Every `*VariantType` enum defined in taxonomy.py.

    Discovered by name rather than listed, so that adding one and forgetting `AnyVariant` fails
    here. `ModelRepoVariant` is deliberately not matched: it is the fp16/fp32 repo flavour, an
    unrelated concept that is not part of `AnyVariant`.
    """
    return {
        obj
        for name, obj in vars(taxonomy).items()
        if name.endswith("VariantType") and isinstance(obj, type) and issubclass(obj, Enum)
    }


def _enums_in_annotation(annotation: object) -> set[type[Enum]]:
    """Every Enum class reachable from a type annotation, through Optional/Union/Literal."""
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return {type(arg) for arg in typing.get_args(annotation) if isinstance(arg, Enum)}
    if origin is not None:
        return set().union(*(_enums_in_annotation(arg) for arg in typing.get_args(annotation)), set())
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {annotation}
    return set()


def _config_classes_with_a_variant() -> dict[tuple[BaseModelType, ModelType], set[type[Enum]]]:
    """The (base, type) -> variant enum map as the model config classes actually declare it."""
    rows: dict[tuple[BaseModelType, ModelType], set[type[Enum]]] = {}
    for config_class in Config_Base.CONFIG_CLASSES:
        field = config_class.model_fields.get("variant")
        if field is None:
            continue
        enums = _enums_in_annotation(field.annotation)
        if not enums:
            continue
        key = (config_class.model_fields["base"].default, config_class.model_fields["type"].default)
        rows.setdefault(key, set()).update(enums)
    return rows


# --- the invariant configs/factory.py rests on ----------------------------------------------------


def test_variant_values_are_globally_unique() -> None:
    """`build_common_fields` validates a bare variant string against the union of every variant enum
    without passing the base (configs/factory.py), so two enums sharing a value would silently
    resolve to whichever the union tried first.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for enum_class in sorted(_variant_enums_in_taxonomy(), key=lambda e: e.__name__):
        for member in enum_class:
            owner = seen.setdefault(member.value, f"{enum_class.__name__}.{member.name}")
            if owner != f"{enum_class.__name__}.{member.name}":
                collisions.append(f"{member.value!r}: {owner} vs {enum_class.__name__}.{member.name}")

    assert sorted(collisions) == sorted(KNOWN_VALUE_COLLISIONS), (
        "variant values must be globally unique -- see the note on Krea2VariantType.Turbo, which is "
        f"'krea2_turbo' for exactly this reason. Collisions: {collisions}"
    )


def test_the_type_adapter_resolves_every_value_to_its_own_enum() -> None:
    """The behavioural form of the check above, straight through the call factory.py makes."""
    wrong: list[str] = []
    for enum_class in sorted(_variant_enums_in_taxonomy(), key=lambda e: e.__name__):
        for member in enum_class:
            resolved = variant_type_adapter.validate_strings(member.value)
            if type(resolved) is not enum_class:
                wrong.append(f"{member.value!r} -> {type(resolved).__name__}, expected {enum_class.__name__}")

    # The values in KNOWN_VALUE_COLLISIONS are exactly the ones that can land on the wrong class.
    tolerated = {c.split(":", 1)[0].strip("'") for c in KNOWN_VALUE_COLLISIONS}
    assert [w for w in wrong if w.split(" ->", 1)[0].strip("'") not in tolerated] == []


# --- the four hand-maintained copies of one list --------------------------------------------------


def test_any_variant_covers_every_variant_enum_in_the_taxonomy() -> None:
    assert set(typing.get_args(AnyVariant)) == _variant_enums_in_taxonomy()


def test_model_record_changes_covers_every_variant_enum() -> None:
    from invokeai.app.services.model_records.model_records_base import ModelRecordChanges

    annotation = ModelRecordChanges.model_fields["variant"].annotation

    assert _enums_in_annotation(annotation) == _variant_enums_in_taxonomy()


def test_the_registry_and_the_allowlist_together_are_any_variant() -> None:
    """Every variant enum is either declared by an architecture or explicitly base-agnostic.

    This is what makes the facet load-bearing rather than decorative: a new variant enum has to be
    accounted for on one side or the other.
    """
    assert declared_variant_enums() | BASE_AGNOSTIC_VARIANT_ENUMS == set(typing.get_args(AnyVariant))


def test_the_allowlist_holds_only_genuinely_base_agnostic_enums() -> None:
    """Guards against parking an architecture's enum in the allowlist to make the check pass."""
    for enum_class in BASE_AGNOSTIC_VARIANT_ENUMS:
        bases = {base for (base, _type), enums in _config_classes_with_a_variant().items() if enum_class in enums}
        assert bases <= set(_NOT_ARCHITECTURES), f"{enum_class.__name__} is used by real architectures: {bases}"


# --- the facet against reality --------------------------------------------------------------------


def test_every_facet_declaration_matches_the_config_classes() -> None:
    """Derived from `Config_Base.CONFIG_CLASSES`, so a wrong declaration cannot pass unnoticed.

    Both directions: a (base, type) the configs give a variant must be declared, and a declaration
    the configs do not back must not exist.
    """
    from_configs = {
        (base, model_type): enums
        for (base, model_type), enums in _config_classes_with_a_variant().items()
        if base not in _NOT_ARCHITECTURES
    }

    problems: list[str] = []
    for (base, model_type), enums in sorted(
        from_configs.items(), key=lambda item: (item[0][0].value, item[0][1].value)
    ):
        assert len(enums) == 1, f"{base.value} x {model_type.value} declares several variant enums: {enums}"
        expected = next(iter(enums))
        declared = get_variant_enum(base, model_type)
        if declared is not expected:
            name = declared.__name__ if declared else "nothing"
            problems.append(f"{base.value} x {model_type.value}: registry says {name}, configs say {expected.__name__}")

    for base in BaseModelType:
        if base in _NOT_ARCHITECTURES:
            continue
        for model_type in ModelType:
            declared = get_variant_enum(base, model_type)
            if declared is not None and (base, model_type) not in from_configs:
                problems.append(
                    f"{base.value} x {model_type.value}: registry declares {declared.__name__}, "
                    f"but no config class for that combination has a variant field"
                )

    assert not problems, "VariantFacet declarations disagree with the model config classes:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize(
    ("base", "model_type", "expected"),
    [
        # The two rows that a base-keyed facet could not express, spelled out.
        (BaseModelType.Wan, ModelType.Main, "WanVariantType"),
        (BaseModelType.Wan, ModelType.LoRA, "WanLoRAVariantType"),
        # One enum, five bases.
        (BaseModelType.Flux, ModelType.PiDDecoder, "PiDDecoderVariantType"),
        (BaseModelType.StableDiffusion3, ModelType.PiDDecoder, "PiDDecoderVariantType"),
    ],
)
def test_the_cases_that_motivate_the_model_type_dimension(
    base: BaseModelType, model_type: ModelType, expected: str
) -> None:
    declared = get_variant_enum(base, model_type)

    assert declared is not None
    assert declared.__name__ == expected


def test_an_architecture_without_variants_declares_nothing() -> None:
    # CogView4, ERNIE-Image, Ideogram 4 and Anima model no variants at all. An empty facet would be
    # indistinguishable from an absent one, so they omit it.
    for base in (BaseModelType.CogView4, BaseModelType.ErnieImage, BaseModelType.Ideogram4, BaseModelType.Anima):
        assert get_variant_enum(base, ModelType.Main) is None
