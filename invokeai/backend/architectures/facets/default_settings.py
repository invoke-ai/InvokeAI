"""The generation parameters an architecture recommends.

These are product decisions, not derivable facts: someone decided Wan TI2V-5B wants 30 steps and
Krea-2 Turbo wants CFG disabled. They are read once, when a model is identified, stored on its
config, and used by the UI to prefill the generation sliders.

They were a `match base:` block of twelve cases in `configs/main.py`, four of which sub-dispatched on
variant, ending in `case _: return None`. Four architectures fell into that fallback and had no
defaults at all, under a standing `TODO(psyche)` asking whether they should; all four have since
been given the values their model cards recommend, which answers it.

With no architecture left without defaults, this is `REQUIRED`. The failure it guards is milder than
the other facets' — a forgotten declaration would leave the sliders wherever the last model put them
rather than crash — but that is a reason to catch it at boot, not a reason to tolerate it: nothing
about a missing prefill is visible enough to be noticed any other way.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import AnyVariant, BaseModelType


@dataclass(frozen=True)
class DefaultSettingsFacet(Facet):
    """What the sliders should say when a model of this architecture is selected."""

    REQUIRED: ClassVar[bool] = True

    by_variant: Mapping[Any, MainModelDefaultSettings]
    """Keyed by variant, with `None` as the fallback for every variant not named.

    Keys are variant enum members. Those are `str`-mixin enums, so they hash by value — a lookup of
    `FluxVariantType.Dev` would find a `Flux2VariantType.Dev` key. Harmless, because a mapping is
    only ever consulted for the architecture that declared it, but it does mean two variants from
    different enums sharing a value would shadow each other within one mapping.
    """

    by_name_hint: Mapping[str, MainModelDefaultSettings] = field(default_factory=dict)
    """Settings selected by a substring of the model's name, checked before `by_variant`.

    For architectures whose sub-models are indistinguishable on disk. ERNIE-Image is the only one:
    Turbo and the base model share an architecture and a config, so there is nothing to probe and no
    variant is modeled — the name is the only signal there is.
    """

    def __post_init__(self) -> None:
        if not self.by_variant:
            raise ValueError(
                "DefaultSettingsFacet declares no settings. `validate()` only checks that the facet is "
                "present, so an empty mapping would boot and then resolve to None at generation time."
            )

    def resolve(
        self,
        variant: AnyVariant | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> MainModelDefaultSettings | None:
        """The settings for one concrete model.

        The install directory's own name is searched as well as the model name, so that renaming a
        model in the install dialog does not lose its defaults. Only the leaf directory: an in-place
        install records an absolute path, and an unrelated ancestor (`/mnt/turbo-nvme/models/`) must
        not hand the base model Turbo's settings.
        """
        if self.by_name_hint:
            haystack = " ".join(part for part in (name, Path(path).name if path else None) if part).lower()
            for hint, settings in self.by_name_hint.items():
                if hint in haystack:
                    return settings

        if variant is not None and variant in self.by_variant:
            return self.by_variant[variant]
        return self.by_variant.get(None)


def resolve_default_settings(
    base: BaseModelType,
    variant: AnyVariant | None = None,
    name: str | None = None,
    path: str | None = None,
) -> MainModelDefaultSettings | None:
    """The generation defaults for a model, or None if its architecture declares none."""
    facet = get(base, DefaultSettingsFacet)
    return facet.resolve(variant, name, path) if facet is not None else None
