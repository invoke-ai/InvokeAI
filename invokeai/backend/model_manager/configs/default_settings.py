"""`MainModelDefaultSettings` — the recommended generation parameters for a model.

Set once, when a model is identified, and stored on its config. The UI reads them to prefill the
generation sliders: pick a Wan TI2V-5B and the steps go to 30, pick Krea-2 Turbo and CFG drops to 1.

Split out of `configs/main.py` so it can stay a leaf. The values themselves are declared per
architecture under `invokeai/backend/architectures/defs/`, and a facet that holds instances of this
class cannot import the module that also does the *looking up* — `configs/main.py` would then import
the registry which imports the defs which import the facet, and the cycle closes on a
half-initialized module.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES

DEFAULTS_PRECISION = Literal["fp16", "fp32"]

_COMPONENT_SLOT_PATTERN = re.compile(r"^[a-z0-9_]+$")
MAX_DEFAULT_COMPONENTS = 16


class MainModelDefaultSettings(BaseModel):
    vae: str | None = Field(default=None, description="Default VAE for this model (model key)")
    vae_precision: DEFAULTS_PRECISION | None = Field(default=None, description="Default VAE precision for this model")
    scheduler: SCHEDULER_NAME_VALUES | None = Field(default=None, description="Default scheduler for this model")
    steps: int | None = Field(default=None, gt=0, description="Default number of steps for this model")
    cfg_scale: float | None = Field(default=None, ge=1, description="Default CFG Scale for this model")
    cfg_rescale_multiplier: float | None = Field(
        default=None, ge=0, lt=1, description="Default CFG Rescale Multiplier for this model"
    )
    width: int | None = Field(default=None, multiple_of=8, ge=64, description="Default width for this model")
    height: int | None = Field(default=None, multiple_of=8, ge=64, description="Default height for this model")
    guidance: float | None = Field(default=None, ge=1, description="Default Guidance for this model")
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    fp8_storage: bool | None = Field(
        default=None,
        description="Store weights in FP8 to reduce VRAM usage (~50% savings). Weights are cast to compute dtype during inference.",
    )

    components: dict[str, str] | None = Field(
        default=None,
        # Left out of a dump while unset: an older server forbids unknown fields, so a record that carries the key
        # even as null would be skipped by it.
        exclude_if=lambda value: value is None,
        description=(
            "Default component models for this main model, keyed by component slot (the generation metadata name, "
            "e.g. `t5_encoder`, `qwen3_encoder`); values are model keys. The default VAE lives in `vae`."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("components")
    @classmethod
    def _validate_components(cls, components: dict[str, str] | None) -> dict[str, str] | None:
        # Keys are not checked against installed models, like `vae`: a model removed later leaves a dangling key that
        # the client ignores. Only the shape is bounded, since the map is stored with every record.
        if not components:
            # An empty map stores nothing, like null, so clients see one representation of "no defaults".
            return None
        if len(components) > MAX_DEFAULT_COMPONENTS:
            raise ValueError(f"at most {MAX_DEFAULT_COMPONENTS} default components are allowed, got {len(components)}")
        for slot, model_key in components.items():
            if slot == "vae":
                raise ValueError("the default VAE is set with the `vae` field, not as a component")
            if not _COMPONENT_SLOT_PATTERN.fullmatch(slot):
                raise ValueError(f"component slot {slot!r} must be snake_case (lowercase letters, digits, underscores)")
            if not model_key.strip():
                raise ValueError(f"component {slot!r} needs a model key")
        return components
