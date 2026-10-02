"""Which variant enum labels an architecture's models, keyed by model type.

Base alone does not determine it, which is why this facet carries a mapping rather than a single
enum:

- Wan mains carry `WanVariantType`, Wan LoRAs carry `WanLoRAVariantType`. Same base, different
  enum, and they are not interchangeable -- applying an A14B LoRA to a TI2V-5B main crashes in the
  layer patcher.
- The PiD decoder's resolution presets are one enum shared across five bases.

Three variant enums cannot live here at all: `ClipVariantType`, `Qwen3VariantType` and
`MistralVariantType` sit on `base=Any` configs, and `Any` is a sentinel the registry refuses to
register. They are named explicitly in `tests/backend/architectures/test_variants.py` so the
completeness check against `AnyVariant` stays total rather than quietly partial.

This facet is deliberately *not* wired into `configs/factory.py`. That module validates a bare
variant string against `variant_type_adapter` without passing the base (`build_common_fields`), so
variant values have to be globally unique -- which is why `Krea2VariantType.Turbo` is
``krea2_turbo`` rather than ``turbo``. Making resolution base-aware would still need a fallback for
the `base=Any` models, so it would move the trap rather than close it, at the cost of touching the
identification path. The invariant is pinned by a test instead.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import generative_bases, get
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@dataclass(frozen=True)
class VariantFacet(Facet):
    """The variant enums an architecture's models are labelled with, by model type.

    Optional: four registered architectures (CogView4, ERNIE-Image, Ideogram 4, Anima) model no
    variants at all, and declaring an empty facet would be indistinguishable from declaring nothing.
    """

    by_model_type: Mapping[ModelType, type[Enum]]

    def __post_init__(self) -> None:
        if not self.by_model_type:
            raise ValueError(
                "VariantFacet declares no model types. An architecture without variants omits the facet entirely."
            )


def get_variant_enum(base: BaseModelType, model_type: ModelType) -> type[Enum] | None:
    """The variant enum for this base and model type, or None if that combination has no variants."""
    facet = get(base, VariantFacet)
    if facet is None:
        return None
    return facet.by_model_type.get(model_type)


def declared_variant_enums() -> frozenset[type[Enum]]:
    """Every variant enum declared by any registered architecture.

    The registry side of the CI guards that keep `AnyVariant`, `variant_type_adapter` and
    `ModelRecordChanges.variant` -- four hand-maintained copies of one list -- from drifting apart.
    """
    enums: set[type[Enum]] = set()
    for base in generative_bases():
        facet = get(base, VariantFacet)
        if facet is not None:
            enums.update(facet.by_model_type.values())
    return frozenset(enums)
