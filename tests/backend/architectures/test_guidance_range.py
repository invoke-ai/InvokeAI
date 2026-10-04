"""The guidance range each architecture declares, against the node field that enforces it.

There is one guidance slider in the UI and its bounds were hardcoded: minimum 0, numeric input
maximum 100, for every architecture alike. Both ends were wrong somewhere. `flux2_denoise.guidance`
is `le=20`, so 30 -- a value FLUX Fill genuinely wants, on a *different* architecture -- persisted
on a FLUX.2 model and failed at enqueue. `ernie_image_denoise.guidance_scale` is `ge=1.0`, so 0 and
0.5 were offered, forwarded unchanged by `graph.ts`, and failed the same way.

`FeaturesFacet.guidance_min` / `guidance_max` move those bounds to the architecture. The ceiling and
any floor above `GUIDANCE_FLOOR` are read off the invocation schema rather than transcribed, the
same way `test_features.py::test_the_dimension_grid_matches_the_node_that_enforces_it` pins the
dimension grid to the node's `multipleOf`; the floor itself is a product rule (see `GUIDANCE_FLOOR`)
and is the one number asserted directly.
"""

from typing import Any

import pytest

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.services.shared.graph import *  # noqa: F401 F403 -- imports all invocations, populating the registry
from invokeai.backend.architectures import generative_bases
from invokeai.backend.architectures.capabilities import architecture_capabilities
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.registry import require
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType

GUIDANCE_FIELD: dict[BaseModelType, tuple[str, str]] = {
    BaseModelType.StableDiffusion1: ("denoise_latents", "cfg_scale"),
    BaseModelType.StableDiffusion2: ("denoise_latents", "cfg_scale"),
    BaseModelType.StableDiffusionXL: ("denoise_latents", "cfg_scale"),
    BaseModelType.StableDiffusion3: ("sd3_denoise", "cfg_scale"),
    BaseModelType.CogView4: ("cogview4_denoise", "cfg_scale"),
    BaseModelType.Krea2: ("krea2_denoise", "cfg_scale"),
    BaseModelType.QwenImage: ("qwen_image_denoise", "cfg_scale"),
    BaseModelType.Flux: ("flux_denoise", "guidance"),
    BaseModelType.Flux2: ("flux2_denoise", "guidance"),
    BaseModelType.Anima: ("anima_denoise", "guidance_scale"),
    BaseModelType.ErnieImage: ("ernie_image_denoise", "guidance_scale"),
    BaseModelType.Wan: ("wan_denoise", "guidance_scale"),
    BaseModelType.LTX2: ("ltx2_denoise", "cfg_scale"),
    BaseModelType.ZImage: ("z_image_denoise", "guidance_scale"),
}
"""Where the one guidance slider's value ends up, per architecture.

Not derivable from the node: `flux2_denoise` takes both a `guidance` and a `cfg_scale`, and only
`guidance` carries the slider -- `FeaturesFacet.guidance_label` is what says so. Recorded in
`generateGraphNodeTypes.json` under each node type's `literalInputs`, which is where these thirteen
were read off, and checked against the graph builders by
`tests/app/invocations/test_frontend_graph_node_types.py`.
"""

NO_GUIDANCE_SLIDER = frozenset(
    {
        # `ideogram4_denoise.guidance_scale` exists and is bounded, but it is preset-derived and has
        # its own optional control (`ideogram4GuidanceScale`); the shared slider never reaches it.
        BaseModelType.Ideogram4,
        # No generation path in webv2, and no guidance field on its denoise node: video-first, with
        # two hardcoded flow schedules.
        BaseModelType.MiniMaxH3,
        # A second pass over someone else's latents rather than a model you generate with.
        BaseModelType.StableDiffusionXLRefiner,
    }
)

GUIDANCE_FLOOR = 1.0
"""The lowest guidance the UI offers anywhere. 1.0 is "no guidance" on every sampler and nothing
below it is a value a person picks on purpose, so an architecture declares this floor unless its
node enforces a higher one. SD's `cfg_scale` enforces the same floor through a `field_validator`
the JSON schema cannot express, which is why the floor is not read off the node. A node whose own
floor is lower (FLUX.2's `guidance` is `ge=0`) is absorbed by this rule on purpose: the UI declares
1.0 for it too. A sampler that someday wants values below 1.0 offered is the moment to revisit the
rule.
"""


def _numeric_bounds(node_type: str, field_name: str) -> tuple[float | None, float | None]:
    """The `ge`/`le` a node applies to one numeric field, as a client sees them.

    Read through the JSON schema rather than off the annotation, because that is the artifact the
    frontend is generated from. A field that is optional or accepts a list of values puts its
    constraints on the numeric branch of an `anyOf` -- `denoise_latents.cfg_scale` is
    `float | list[float]` -- so the top level is not always where they are.
    """
    cls = InvocationRegistry.get_invocations_map().get(node_type)
    assert cls is not None, f"no invocation is registered as '{node_type}'"
    schema: dict[str, Any] = cls.model_json_schema()["properties"][field_name]
    numeric = [branch for branch in schema.get("anyOf", [schema]) if branch.get("type") == "number"]
    assert len(numeric) == 1, f"{node_type}.{field_name} has {len(numeric)} numeric branches, not one"
    return numeric[0].get("minimum"), numeric[0].get("maximum")


def test_every_architecture_is_accounted_for() -> None:
    """A new architecture cannot be added without saying where its guidance value goes.

    Without this the pin below is only as complete as the map above it, and an architecture omitted
    from both would go silently unchecked -- which is how the UI came to offer 0 for ERNIE-Image.
    """
    covered = set(GUIDANCE_FIELD) | NO_GUIDANCE_SLIDER
    missing = sorted(b.value for b in generative_bases() if b not in covered)
    assert missing == [], (
        f"{missing} declare a guidance range that nothing checks. Add the denoise node and the "
        f"field its slider value is sent as to GUIDANCE_FIELD, or say why there is none in "
        f"NO_GUIDANCE_SLIDER."
    )
    assert not (set(GUIDANCE_FIELD) & NO_GUIDANCE_SLIDER)


@pytest.mark.parametrize("base", sorted(GUIDANCE_FIELD, key=lambda b: b.value), ids=lambda b: b.value)
def test_the_guidance_range_matches_the_node_that_enforces_it(base: BaseModelType) -> None:
    node_type, field_name = GUIDANCE_FIELD[base]
    node_min, node_max = _numeric_bounds(node_type, field_name)
    facet = require(base, FeaturesFacet)

    expected_min = GUIDANCE_FLOOR if node_min is None else max(GUIDANCE_FLOOR, node_min)
    assert (facet.guidance_min, facet.guidance_max) == (expected_min, node_max), (
        f"'{base.value}' declares guidance {facet.guidance_min}..{facet.guidance_max} but "
        f"{node_type}.{field_name} constrains it to {node_min}..{node_max} and the UI floor is "
        f"{GUIDANCE_FLOOR}, so it should declare {expected_min}..{node_max}. Whichever is wrong, a UI "
        f"reading the declaration offers a value the graph rejects or hides one it takes -- see "
        f"invokeai/backend/architectures/defs/{base.value.replace('-', '_')}.py."
    )


@pytest.mark.parametrize("base", sorted(NO_GUIDANCE_SLIDER, key=lambda b: b.value), ids=lambda b: b.value)
def test_an_architecture_with_no_slider_declares_no_range(base: BaseModelType) -> None:
    """Nothing sends these a guidance value, so a narrowed range here would be a claim about a
    control that does not exist."""
    facet = require(base, FeaturesFacet)
    assert (facet.guidance_min, facet.guidance_max) == (GUIDANCE_FLOOR, None), base.value


def test_a_range_that_holds_no_values_is_refused() -> None:
    """Declaration-time, like `dimension_grid_by_variant`'s None-key guard. The two bounds are
    separate fields and nothing about the types stops them being entered the wrong way round; a
    ceiling below the floor leaves the UI a slider with nothing on it."""
    with pytest.raises(ValueError, match="holds no values"):
        FeaturesFacet(
            negative_prompt=NegativePrompt(visible=True, usage="always"),
            dimension_grid=16,
            guidance_min=1.0,
            guidance_max=0.5,
        )


def _slider_value(settings: MainModelDefaultSettings) -> float | None:
    """The number the slider is restored to: `guidance` where the declaration sets it, `cfg_scale`
    otherwise -- the rule `FeaturesFacet.guidance_label` documents, and the reason FLUX can carry
    both (`cfg_scale=1.0` meaning off, `guidance=3.5` meaning the distilled embedding)."""
    return settings.guidance if settings.guidance is not None else settings.cfg_scale


@pytest.mark.parametrize("base", sorted(GUIDANCE_FIELD, key=lambda b: b.value), ids=lambda b: b.value)
def test_every_recommended_guidance_value_is_inside_the_declared_range(base: BaseModelType) -> None:
    """A recommended default outside its own architecture's range is a one-click failed enqueue.

    Reaches the per-variant and per-name-hint settings too: FLUX Fill's `guidance=30.0` is a variant
    row, and it is exactly the kind of value a ceiling copied from the wrong architecture forbids.
    """
    facet = require(base, FeaturesFacet)
    defaults = require(base, DefaultSettingsFacet)

    labelled = [(str(variant), s) for variant, s in defaults.by_variant.items()]
    labelled += [(f"name hint '{hint}'", s) for hint, s in defaults.by_name_hint.items()]

    outside = []
    for label, settings in labelled:
        value = _slider_value(settings)
        if value is None:
            continue
        if value < facet.guidance_min or (facet.guidance_max is not None and value > facet.guidance_max):
            outside.append(f"{label}: {value}")
    assert outside == [], (
        f"'{base.value}' recommends {outside}, outside its declared guidance range "
        f"{facet.guidance_min}..{facet.guidance_max}."
    )


def test_the_served_table_carries_the_range_on_every_row() -> None:
    """The declaration is only useful if a client can read it, and it has to be on the variant rows
    as well: a client joins on `(base, variant)` and does not fall back to the base row."""
    declared = {base.value: require(base, FeaturesFacet) for base in generative_bases()}
    served = {
        (row.base.value, row.variant): (row.features.guidance_min, row.features.guidance_max)
        for row in architecture_capabilities()
    }
    assert served, "no rows served"
    mismatched = {
        key: value
        for key, value in served.items()
        if value != (declared[key[0]].guidance_min, declared[key[0]].guidance_max)
    }
    assert mismatched == {}
