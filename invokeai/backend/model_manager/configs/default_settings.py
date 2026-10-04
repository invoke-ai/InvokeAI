"""`MainModelDefaultSettings` — the recommended generation parameters for a model.

Set once, when a model is identified, and stored on its config. The UI reads them to prefill the
generation sliders: pick a Wan TI2V-5B and the steps go to 30, pick Krea-2 Turbo and CFG drops to 1.

Split out of `configs/main.py` so it can stay a leaf. The values themselves are declared per
architecture under `invokeai/backend/architectures/defs/`, and a facet that holds instances of this
class cannot import the module that also does the *looking up* — `configs/main.py` would then import
the registry which imports the defs which import the facet, and the cycle closes on a
half-initialized module.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES

DEFAULTS_PRECISION = Literal["fp16", "fp32"]


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

    model_config = ConfigDict(extra="forbid")
