"""Reading which backbone a model is for out of its name.

Some checkpoints are the same network for several backbones, so their weights cannot say which one
a file belongs to: a PiD decoder for FLUX.1, SD3 or Qwen-Image, or a 16-channel `AutoencoderKL` for
FLUX.1 or SD3. There the weights narrow the answer to a family and the name has to pick within it.
Identification uses these helpers only for that last step, never to choose a backbone the weights
have not already allowed.
"""

import re
from typing import Any

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelSourceType


def name_components(mod: ModelOnDisk, override_fields: dict[str, Any]) -> tuple[str, ...]:
    """The name evidence for a model, most specific first: its file name, its folder's name, and the
    install source.

    Publishers often put the backbone in the *directory* name rather than the weights filename --
    NVIDIA ships PiD as ``PiD_res2k_sr4x_official_<backbone>_distill_4step/model_ema_bf16.pth``, and
    a Hugging Face VAE lives at ``<repo>::vae/diffusion_pytorch_model.safetensors``. A direct
    single-file install stores the checkpoint as ``<uuid>/<file>`` and drops that directory, which is
    why the install source is consulted at all: for an HF or URL install it still carries the name.

    These used to be concatenated into one string and substring-matched, which let a fixed backbone
    precedence decide cases the name had already answered — `/flux/model_sd3.pth` matched `flux`
    first and was registered as FLUX although the file itself says sd3. Matching component by
    component and taking the first that names exactly one backbone lets the more specific name win.

    A local install contributes no source: the model manager sets `source` to the file's own path
    when there is no remote one (`ModelConfigFactory.build_common_fields`), so trusting it would mean
    matching against arbitrary ancestor directories of wherever the user keeps their models. Nothing
    is lost by dropping it — `install_path` identifies a local file *before* it moves it, so the
    filename and parent directory are still the originals.
    """
    components = [mod.path.name, mod.path.parent.name]
    if override_fields.get("source_type") != ModelSourceType.Path:
        components.append(str(override_fields.get("source") or ""))
    return tuple(c for c in components if c)


# Ordered so that a more specific spelling is consumed before a more general one that it contains:
# `flux2` before `flux`. That is precedence between two spellings of one answer, not between two
# answers — see `backbone_named_in`.
_BACKBONE_NAME_PATTERNS: tuple[tuple[BaseModelType, re.Pattern[str]], ...] = (
    (BaseModelType.Flux2, re.compile(r"(?<![a-z0-9])flux[_\-.]?2(?![a-z0-9])")),
    (BaseModelType.StableDiffusionXL, re.compile(r"(?<![a-z0-9])sdxl(?![a-z0-9])")),
    (BaseModelType.QwenImage, re.compile(r"(?<![a-z0-9])qwen[_\-.]?image(?![a-z0-9])")),
    # `sd3`, `sd3.5`, `sd35`, Stability's own `stable-diffusion-3.5-large`, and the size run into the
    # version, as in `sd35l` or `sd3medium`.
    (
        BaseModelType.StableDiffusion3,
        re.compile(r"(?<![a-z0-9])(?:sd|stable[_\-.]?diffusion)[_\-.]?3(?:[_\-.]?5)?(?:l|m|large|medium)?(?![a-z0-9])"),
    ),
    # `flux`, and `flux1` / `flux.1` — how FLUX.1 is spelled wherever FLUX.2 also exists.
    (BaseModelType.Flux, re.compile(r"(?<![a-z0-9])flux(?:[_\-.]?1)?(?![a-z0-9])")),
)


def backbone_named_in(text: str) -> BaseModelType | None:
    """The single backbone *text* names, or None if it names none — or more than one.

    Two different backbones in one string is not a precedence question, it is a text that decides
    nothing; resolving it by a fixed order is how a directory named `flux` came to outrank a file
    named `model_sd3`. Abstaining leaves the decision to the explicit `base` override, or to the
    caller's default for the family.
    """
    remaining, found = text.lower(), set()
    for base, pattern in _BACKBONE_NAME_PATTERNS:
        if pattern.search(remaining):
            found.add(base)
            # Consumed so the general spelling cannot match the specific one's leftovers.
            remaining = pattern.sub(" ", remaining)
    return next(iter(found)) if len(found) == 1 else None


def backbone_from_components(components: tuple[str, ...]) -> BaseModelType | None:
    """The backbone named by the most specific component that names exactly one."""
    for component in components:
        if (named := backbone_named_in(component)) is not None:
            return named
    return None
