"""LTX-2 guidance: which transformer passes a step runs, and how their predictions combine.

LTX-2 steers with three terms, each an extra forward of the same step:

- **CFG** against the negative prompt,
- **STG** (spatio-temporal guidance) against a pass whose self-attention is perturbed -- skipped
  outright -- in a small set of blocks,
- **modality isolation** against a pass with the audio<->video cross-attention disabled.

The terms are summed as deltas from the conditional prediction, in x0 space, and optionally
rescaled toward the conditional prediction's standard deviation. Video and audio are combined
separately: they share every scale except CFG, which the release sets far higher for audio.

The passes are the *union* over the two modalities, because one forward produces both predictions.
A modality that does not want a term simply has a scale that makes its delta zero, so the combine
needs no per-modality pass bookkeeping -- only the planning does, so that a pass no modality wants
is never paid for.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import torch

from invokeai.backend.ltx2.constants import (
    LTX2_AUDIO_CFG_SCALE,
    LTX2_CFG_SCALE,
    LTX2_GUIDANCE_RESCALE,
    LTX2_MODALITY_SCALE,
    LTX2_STG_BLOCKS,
    LTX2_STG_SCALE,
)

LTX2GuidancePass = Literal["cond", "uncond", "stg", "modality"]

PASS_COND: Final[LTX2GuidancePass] = "cond"
PASS_UNCOND: Final[LTX2GuidancePass] = "uncond"
PASS_STG: Final[LTX2GuidancePass] = "stg"
PASS_MODALITY: Final[LTX2GuidancePass] = "modality"


@dataclass(frozen=True)
class LTX2Guidance:
    """The guidance scales of one denoise run, and the passes they imply."""

    cfg_scale: float = LTX2_CFG_SCALE
    audio_cfg_scale: float = LTX2_AUDIO_CFG_SCALE
    stg_scale: float = LTX2_STG_SCALE
    modality_scale: float = LTX2_MODALITY_SCALE
    rescale: float = LTX2_GUIDANCE_RESCALE
    stg_blocks: tuple[int, ...] = LTX2_STG_BLOCKS

    @property
    def passes(self) -> tuple[LTX2GuidancePass, ...]:
        """The forwards one step runs, conditional first."""
        passes: list[LTX2GuidancePass] = [PASS_COND]
        if not (math.isclose(self.cfg_scale, 1.0) and math.isclose(self.audio_cfg_scale, 1.0)):
            passes.append(PASS_UNCOND)
        if not math.isclose(self.stg_scale, 0.0) and self.stg_blocks:
            passes.append(PASS_STG)
        if not math.isclose(self.modality_scale, 1.0):
            passes.append(PASS_MODALITY)
        return tuple(passes)

    @property
    def needs_negative_conditioning(self) -> bool:
        return PASS_UNCOND in self.passes

    def combine_video(self, predictions: Mapping[LTX2GuidancePass, torch.Tensor]) -> torch.Tensor:
        return self._combine(predictions, self.cfg_scale)

    def combine_audio(self, predictions: Mapping[LTX2GuidancePass, torch.Tensor]) -> torch.Tensor:
        return self._combine(predictions, self.audio_cfg_scale)

    def _combine(self, predictions: Mapping[LTX2GuidancePass, torch.Tensor], cfg_scale: float) -> torch.Tensor:
        cond = predictions[PASS_COND].float()
        pred = cond
        if PASS_UNCOND in predictions:
            pred = pred + (cfg_scale - 1.0) * (cond - predictions[PASS_UNCOND].float())
        if PASS_STG in predictions:
            pred = pred + self.stg_scale * (cond - predictions[PASS_STG].float())
        if PASS_MODALITY in predictions:
            pred = pred + (self.modality_scale - 1.0) * (cond - predictions[PASS_MODALITY].float())

        if self.rescale != 0.0 and pred is not cond:
            # Pull the guided prediction's overall scale back toward the conditional one's, which
            # summed deltas inflate. The reference implementations spell this two ways -- a whole
            # tensor's standard deviation upstream, per batch item in diffusers' `rescale_noise_cfg`
            # -- and this runs one video per request, where they are the same number.
            factor = self.rescale * (cond.std() / pred.std()) + (1.0 - self.rescale)
            pred = pred * factor
        return pred
