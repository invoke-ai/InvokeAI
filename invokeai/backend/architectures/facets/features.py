"""Which generation features an architecture supports.

The UI has to know whether to show a negative prompt box, whether a ControlNet layer can attach,
how many reference images to accept, whether clip-skip means anything. None of it is derivable from
a model file — it follows from what the architecture is — and all of it was living in the frontend.

This is where those answers live now. `capabilities.py` serves them, and webv2 reads the served table
instead of keeping its own copy; a new architecture cannot be registered without declaring them.
Where a number is enforced by a node -- a `multiple_of`, a `ge`/`le` -- a test pins the declaration to
that node, so the table cannot offer what the graph will reject.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.model_manager.taxonomy import AnyVariant

NegativePromptUsage = Literal["always", "cfg-gated", "never"]
"""`cfg-gated` means the field exists but only does anything above CFG 1 — the distilled models."""

ControlKind = Literal["controlnet", "t2i_adapter", "control_lora", "z_image_control"]

SchedulerSet = Literal["standard", "flow", "flow-no-lcm", "anima"]
"""Which family of schedulers to offer. `None` means the architecture drives its own and offers no
choice — MiniMax H3 steps video and audio down two hardcoded flow schedules. `flow-no-lcm` is the flow
set without LCM, for a variant whose schedule LCM does not fit."""


@dataclass(frozen=True)
class NegativePrompt:
    """Whether to show the box, and whether what is typed in it is used."""

    visible: bool
    usage: NegativePromptUsage


@dataclass(frozen=True)
class FeaturesFacet(Facet):
    """What the UI may offer for this architecture."""

    REQUIRED: ClassVar[bool] = True

    negative_prompt: NegativePrompt
    dimension_grid: int
    """Width and height must be a multiple of this, for every variant `dimension_grid_by_variant`
    does not override.

    Pinned to the `multiple_of` on the architecture's denoise node, where it has one: a test asserts
    the two agree, so the base grid cannot drift into a UI that offers dimensions that node's field
    validation will reject. It says nothing about constraints a node applies inside `invoke()` --
    those are per-variant, and `dimension_grid_by_variant` is where they are declared."""

    dimension_grid_by_variant: Mapping[AnyVariant, int] = field(default_factory=dict)
    """Variants whose grid is coarser (or finer) than the base one, keyed by variant enum member.

    Wan is the reason this exists: A14B denoises in the 16-channel space at 8x and takes multiples
    of 16, TI2V-5B in the 48-channel Wan2.2 space at 16x and takes multiples of 32. The node fields
    carry the A14B number, so 1280x720 -- both multiples of 16 -- was offered by the UI and then
    rejected inside `wan_denoise.invoke()`, after enqueue and after the model had loaded.

    Keys are variant enum members, with the same `str`-mixin hashing caveat as
    `DefaultSettingsFacet.by_variant`. A variant that is not named, or a model whose variant is not
    known, gets `dimension_grid`; `None` is not a key, because the base value is already the
    fallback."""

    guidance_label: Literal["CFG", "Guidance"] = "CFG"
    """What to call the slider. FLUX-family models expose a distilled guidance embedding rather than
    classifier-free guidance, and calling it CFG has confused users into expecting CFG behaviour.

    There is only ever *one* slider. No architecture offers both knobs, so the UI shows a single
    control and this label is the whole difference between them -- see `GenerateModelFields.tsx`,
    where the field is labelled from here and valued from `cfgScale`. `MainModelDefaultSettings` has
    both a `cfg_scale` and a `guidance` field, so a declaration must not fill in both and expect the
    UI to distinguish: the value the slider takes is `guidance` where it is set and `cfg_scale`
    otherwise. Today only FLUX sets both (cfg_scale 1.0 meaning "off", guidance 3.5 meaning the
    distilled embedding)."""

    guidance_min: float = 1.0
    """Lowest value the guidance slider may offer.

    1.0 means "no guidance" on every sampler, and nothing below it is a value a person picks on
    purpose -- so 1.0 is the UI's floor everywhere, raised to the node field's `ge` where that is
    higher. The node field is the one the slider's value is actually sent as: `guidance` for the
    models whose `guidance_label` says so, `cfg_scale` or `guidance_scale` for the rest. A test
    asserts the two agree, so the UI cannot offer a number the graph will refuse.

    Anima, ERNIE-Image, Wan and Z-Image enforce `ge=1.0` on the node itself; SD's `cfg_scale`
    enforces the same floor through a validator the JSON schema cannot see. The UI used to offer 0
    and 0.5 for all of them, and `graph.ts` forwarded the value unchanged into an enqueue that
    failed validation."""

    guidance_max: float | None = None
    """Highest value the guidance slider may offer, or None where the node enforces no ceiling.

    FLUX.2 is the only architecture with one: `flux2_denoise.guidance` is `le=20`, while the UI's
    numeric input accepted 100 for every architecture alike -- so 30, a value FLUX Fill really does
    want, persisted on a FLUX.2 model and was submitted and rejected.

    None is the node's answer, not the UI's: a client still needs its own slider ceiling, and None
    only says the graph will not reject a large value."""

    scheduler_set: SchedulerSet | None = None
    scheduler_set_by_variant: Mapping[AnyVariant, SchedulerSet] = field(default_factory=dict)
    """A variant whose scheduler choice differs from `scheduler_set`; the declaring architecture says why."""

    scheduler_applies_to_graph: bool = False
    """Whether the chosen scheduler reaches the graph, or is only a UI affordance."""

    control_kinds: frozenset[ControlKind] = frozenset()
    max_reference_images: int = 0
    reference_images_require_variant: str | None = None
    """Qwen-Image accepts reference images only as the `edit` variant — the one feature in this
    table that a base alone cannot answer."""

    supports_regional_guidance: bool = False
    regional_negative: bool = False
    """Regional *negative* prompts, which only the SD family has."""

    clip_skip_max: int | None = None
    supports_seamless: bool = False
    supports_cfg_rescale: bool = False
    sd_vae_override: bool = False
    color_compensation: bool = False
    vae_precision: bool = False

    def __post_init__(self) -> None:
        if None in self.dimension_grid_by_variant:
            raise ValueError(
                "FeaturesFacet.dimension_grid_by_variant has a None key. Unlike "
                "DefaultSettingsFacet.by_variant, it has no fallback entry: `dimension_grid` is the "
                "answer for every variant not named here, and a None key would be dead weight."
            )

        if None in self.scheduler_set_by_variant:
            raise ValueError(
                "FeaturesFacet.scheduler_set_by_variant has a None key; `scheduler_set` is already the "
                "answer for every variant not named here."
            )

        if self.guidance_max is not None and self.guidance_max < self.guidance_min:
            raise ValueError(
                f"FeaturesFacet declares a guidance range that holds no values: guidance_min="
                f"{self.guidance_min} is above guidance_max={self.guidance_max}."
            )

    def resolve_dimension_grid(self, variant: AnyVariant | None = None) -> int:
        """The grid one concrete model's width and height must be a multiple of."""
        if variant is None:
            return self.dimension_grid
        return self.dimension_grid_by_variant.get(variant, self.dimension_grid)

    def resolve_scheduler_set(self, variant: AnyVariant | None = None) -> SchedulerSet | None:
        """The scheduler family one concrete model is offered."""
        if variant is None:
            return self.scheduler_set
        return self.scheduler_set_by_variant.get(variant, self.scheduler_set)

    @property
    def supports_reference_images(self) -> bool:
        return self.max_reference_images > 0
