"""Sub-tower config resolution for full-CLIP checkpoints.

``CLIPVisionModelWithProjection`` / ``CLIPTextModelWithProjection`` size their
projection head from the *nested* ``vision_config`` / ``text_config``
``projection_dim``. ``CLIPModel`` itself sizes both projections from the
*top-level* ``projection_dim`` and never reads the nested values — so a
published full-CLIP repo can carry nested defaults that disagree with its own
weights (apple/DFN2B-CLIP-ViT-L-14-39B: top-level 768, nested 512) and still
load fine as ``CLIPModel``, while the WithProjection classes build a 512-wide
head and fail on the 768-wide checkpoint tensor. The top-level value is the
authoritative one; these helpers copy it into the sub-config so a tower can be
loaded on its own.
"""

from pathlib import Path
from typing import Literal, Optional

from transformers import AutoConfig, CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def clip_tower_config_override(
    model_path: str | Path, tower: Literal["vision", "text"]
) -> Optional[CLIPVisionConfig | CLIPTextConfig]:
    """The config to pass when loading one tower of the checkpoint at ``model_path``.

    Returns ``None`` when the checkpoint is not a full ``CLIPModel`` (e.g. a
    vision-only IP-Adapter image-encoder directory, or a SigLIP model) — the
    tower class's own config handling is already correct there, and the caller
    should not pass an override.
    """
    try:
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    except Exception:
        # AutoConfig is stricter than both the install probe and the tower
        # classes' own config resolution: it refuses a config.json with no
        # model_type, which the pre-override load path accepted. Returning
        # None restores that path exactly; a genuinely broken directory then
        # fails in from_pretrained with its usual error.
        return None
    if not isinstance(config, CLIPConfig):
        return None
    sub_config = config.vision_config if tower == "vision" else config.text_config
    sub_config.projection_dim = config.projection_dim
    return sub_config
