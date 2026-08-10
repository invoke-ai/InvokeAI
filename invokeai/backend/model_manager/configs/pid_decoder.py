"""Model configs for PiD (Pixel Diffusion Decoder) checkpoints.

PiD decoders are released by NVIDIA at https://huggingface.co/nvidia/PiD and
ship per supported backbone (FLUX.1, FLUX.2, SD3, SDXL, Qwen-Image). Most
backbones offer two resolution presets (`res2k_sr4x_*` and `res2kto4k_sr4x_*`),
while SDXL and Qwen-Image ship only the `res2kto4k_sr4x_*` preset. See
`LICENSE-PiD.txt` at the repo root — code is Apache-2.0, weights are NSCLv1
(non-commercial / research).
"""

import re
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelSourceType,
    ModelType,
    PiDDecoderVariantType,
)
from invokeai.backend.pid.state_dict_utils import pid_net_shapes

# Marker substring produced by `PidNet.lq_proj` (see
# invokeai/backend/pid/_src/networks/pid_net.py). The pretrained PixDiT_T2I
# weights do not contain `lq_proj`, so its presence in any key is diagnostic
# of a PiD-style checkpoint. We match by substring (not prefix) because the
# official `.pth` files keep PidDistillModel's `net.` prefix, so keys look
# like `net.lq_proj.layers.0.weight`.
_PID_MARKER_SUBSTRING = "lq_proj"


def _looks_like_pid_decoder(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(k, str) and _PID_MARKER_SUBSTRING in k for k in state_dict)


# PidNet's latent input projection: a Conv2d of shape (lq_hidden_dim, lq_latent_channels, 3, 3).
# Identification reads three separate facts off this one weight — the architecture version (dim 0),
# the backbone (dim 1) and, via the contract, its kernel — which is why it is worth naming.
_LATENT_PROJ_KEY = "lq_proj.latent_proj.0.weight"

# dim 1 of the latent projection is the backbone's latent channel count. It is the only architectural
# dimension that varies between backbones, and therefore the only name-independent discriminator
# available. FLUX.1, SD3 and Qwen-Image are architecturally identical and share 16 channels; nothing
# in the weights can separate them.
_LATENT_CHANNELS_TO_BASES: dict[int, set[BaseModelType]] = {
    4: {BaseModelType.StableDiffusionXL},
    16: {BaseModelType.Flux, BaseModelType.StableDiffusion3, BaseModelType.QwenImage},
    128: {BaseModelType.Flux2},
}

# dim 0 is PidNet's `lq_hidden_dim`. `build_pid_net` constructs the legacy 512-dim network; NVIDIA's
# v1.5 checkpoints use 1024 (plus PiT injection, scalar gates, etc.) and cannot be loaded into it.
_SUPPORTED_LQ_HIDDEN_DIM = 512

_Shapes = Mapping[str, tuple[int, ...] | None]


def _raise_if_discriminator_malformed(shapes: _Shapes, contract: Mapping[str, tuple[int, ...]]) -> None:
    """Reject a checkpoint whose latent projection is present but is not a conv weight.

    Every read identification makes off this weight requires it to be a 4D conv, and each used to
    answer None when it was not — so a malformed tensor made the architecture check, the backbone
    check and the channel check all abstain at once, and the file fell through to name-only matching,
    which happily accepted it. Loading then failed on a size mismatch.

    Only reached when the weight is present: its *absence* is a truncation, which
    `_raise_if_pid_net_contract_unmet` diagnoses far better than a guess about the architecture.
    """
    shape = shapes[_LATENT_PROJ_KEY]
    expected = contract[_LATENT_PROJ_KEY]
    if shape is None or len(shape) != len(expected) or shape[2:] != expected[2:]:
        raise InvalidMatchError(
            f"PiD checkpoint has a malformed {_LATENT_PROJ_KEY}: expected a "
            f"{len(expected)}D conv weight with a {'x'.join(str(d) for d in expected[2:])} kernel, got "
            f"{shape if shape is not None else 'a value with no shape'}"
        )


def _raise_if_architecture_unsupported(shapes: _Shapes) -> None:
    """Reject a PiD decoder whose network shape `build_pid_net` cannot construct.

    Runs before the contract check so the diagnosis is the accurate one: a v1.5 checkpoint is intact,
    and judging it against the legacy contract would report it as a pile of missing and unexpected
    keys rather than as the newer architecture it is.
    """
    lq_hidden_dim = shapes[_LATENT_PROJ_KEY][0]  # type: ignore[index]  # rank checked above
    if lq_hidden_dim != _SUPPORTED_LQ_HIDDEN_DIM:
        raise InvalidMatchError(
            f"PiD decoder has lq_proj hidden dim {lq_hidden_dim}, but InvokeAI only supports the legacy "
            f"{_SUPPORTED_LQ_HIDDEN_DIM}-dim architecture (NVIDIA's v1.5 checkpoints are not yet supported)."
        )


def _raise_if_no_backbone_can_accept(shapes: _Shapes) -> None:
    """Reject a PiD decoder that none of the five backbone configs could ever claim.

    The counterpart to `_validate_base`, and the reason the two are separate. `_validate_base` decides
    *which* backbone a checkpoint belongs to and says "not this one" with `NotAMatchError` — four of
    the five classes are meant to say exactly that about every valid checkpoint. A rejection here is
    backbone-independent, so all five would raise it for the same reason, leaving the file with no
    match at all and letting the factory register it through the `Unknown_Config` fallback: a PiD
    decoder on record as a model nothing can load. Hence `InvalidMatchError`.

    Runs before the contract check because a decoder for an unsupported backbone would otherwise be
    reported as a shape mismatch on one weight, which is true and useless.
    """
    channels = shapes[_LATENT_PROJ_KEY][1]  # type: ignore[index]  # rank checked above
    if channels not in _LATENT_CHANNELS_TO_BASES:
        raise InvalidMatchError(
            f"PiD checkpoint has {channels} latent channels; no supported backbone uses this "
            "(supported: 4 for SDXL, 16 for FLUX.1/SD3/Qwen-Image, 128 for FLUX.2)"
        )


def _and_more(items: list[Any]) -> str:
    return f" (+ {len(items) - 5} more)" if len(items) > 5 else ""


def _raise_if_pid_net_contract_unmet(shapes: _Shapes, contract: Mapping[str, tuple[int, ...]]) -> None:
    """Hold the checkpoint to exactly the contract `load_pid_decoder` enforces.

    Checking only the LQ projection accepted a file that carried every LQ weight and none of the 385
    backbone weights; the loader then refused it. A subset check is not a milder version of the same
    guarantee — loaders run under `skip_torch_weight_init()`, so a weight the checkpoint does not
    supply is uninitialised memory rather than a default.

    Missing *and* unexpected keys are fatal here because both are fatal there, which is what makes
    installation and loading accept the same set of files. A stricter installer cannot reject a file
    that would have loaded: the loader already refuses everything rejected here.

    `_LATENT_PROJ_KEY` is excluded from the shape comparison, and only from that: it is the one
    parameter whose shape legitimately varies by backbone, and its variable dimensions each have a
    dedicated check above with a dedicated message.
    """
    # No "this is a base PixDiT_T2I checkpoint" special case, unlike `load_pid_decoder`: those weights
    # carry no `lq_proj` key at all, so such a file never reaches here — `_looks_like_pid_decoder`
    # has already turned it away, and with a better message.
    if missing := sorted(contract.keys() - shapes.keys()):
        raise InvalidMatchError(
            f"PiD checkpoint is missing {len(missing)} of the weights required by PidNet; the file is "
            f"incomplete and cannot be used as a PiD decoder: {missing[:5]}{_and_more(missing)}"
        )

    if unexpected := sorted(shapes.keys() - contract.keys()):
        raise InvalidMatchError(
            f"PiD checkpoint has {len(unexpected)} keys PidNet does not expect, which `load_pid_decoder` "
            f"rejects too: {unexpected[:5]}{_and_more(unexpected)}"
        )

    mismatched = [(k, shapes[k], want) for k, want in contract.items() if k != _LATENT_PROJ_KEY and shapes[k] != want]
    if mismatched:
        k, got, want = mismatched[0]
        raise InvalidMatchError(
            f"PiD checkpoint has {len(mismatched)} weights whose shape PidNet cannot accept "
            f"(e.g. {k}: {got}, expected {want}); loading it would fail with a size mismatch"
        )


def _name_components(mod: ModelOnDisk, override_fields: dict[str, Any]) -> tuple[str, ...]:
    """The name evidence for backbone and variant, most specific first.

    NVIDIA distributes PiD checkpoints as
    ``PiD_res2k_sr4x_official_<backbone>_distill_4step/model_ema_bf16.pth``, so the backbone and the
    preset usually live in the *directory* name rather than the weights filename. A direct
    single-file install stores the checkpoint as ``<uuid>/model_ema_bf16.pth`` and drops that
    directory, which is why the install source is consulted at all: for an HF or URL install it still
    carries NVIDIA's name.

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
# answers — see `_backbone_named_in`.
_BACKBONE_NAME_PATTERNS: tuple[tuple[BaseModelType, re.Pattern[str]], ...] = (
    (BaseModelType.Flux2, re.compile(r"flux[_\-.]?2")),
    (BaseModelType.StableDiffusionXL, re.compile(r"sdxl")),
    (BaseModelType.QwenImage, re.compile(r"qwen[_\-.]?image")),
    (BaseModelType.StableDiffusion3, re.compile(r"sd[_\-.]?3")),
    (BaseModelType.Flux, re.compile(r"flux")),
)


def _backbone_named_in(text: str) -> BaseModelType | None:
    """The single backbone *text* names, or None if it names none — or more than one.

    Two different backbones in one string is not a precedence question, it is a text that decides
    nothing; resolving it by a fixed order is how a directory named `flux` came to outrank a file
    named `model_sd3`. Abstaining leaves the decision to the explicit `base` override, or to the
    FLUX.1 default for the 16-channel family.
    """
    remaining, found = text.lower(), set()
    for base, pattern in _BACKBONE_NAME_PATTERNS:
        if pattern.search(remaining):
            found.add(base)
            # Consumed so the general spelling cannot match the specific one's leftovers.
            remaining = pattern.sub(" ", remaining)
    return next(iter(found)) if len(found) == 1 else None


def _backbone_from_components(components: tuple[str, ...]) -> BaseModelType | None:
    """The backbone named by the most specific component that names exactly one."""
    for component in components:
        if (named := _backbone_named_in(component)) is not None:
            return named
    return None


# Backbones for which NVIDIA ships exactly one preset — for these the variant is known even when the
# name gives nothing away. FLUX.1 / FLUX.2 / SD3 ship both presets and fall back to `Res2k_Sr4x`.
_SINGLE_VARIANT_BACKBONES: dict[BaseModelType, PiDDecoderVariantType] = {
    BaseModelType.StableDiffusionXL: PiDDecoderVariantType.Res2kTo4k_Sr4x,
    BaseModelType.QwenImage: PiDDecoderVariantType.Res2kTo4k_Sr4x,
}


def _variant_from_components(components: tuple[str, ...], base: BaseModelType) -> PiDDecoderVariantType:
    """Map NVIDIA's `res2k_sr4x` / `res2kto4k_sr4x` name slice to a variant.

    Same specificity ordering as the backbone match. If no component names a preset, fall back to the
    backbone's only published one where there is one, and to ``Res2k_Sr4x`` for those shipping both.
    """
    for component in components:
        n = component.lower()
        # `res2kto4k` contains `res2k`, so the 2K-to-4K spellings are tested first.
        if "res2kto4k" in n or "res2k_to_4k" in n or "res2k_to4k" in n:
            return PiDDecoderVariantType.Res2kTo4k_Sr4x
        if "res2k" in n:
            return PiDDecoderVariantType.Res2k_Sr4x
    return _SINGLE_VARIANT_BACKBONES.get(base, PiDDecoderVariantType.Res2k_Sr4x)


class PiDDecoder_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Shared logic for PiD decoder checkpoint configs.

    Concrete subclasses pin `base` to a specific backbone. A checkpoint is first held to the full
    `PidNet` contract — the same keys and shapes `load_pid_decoder` demands — and the backbone then
    comes from the latent channel count in the weights, with an explicit override or the name as the
    tie-breaker for the architecturally identical FLUX.1 / SD3 / Qwen-Image family. `variant` is
    carried as data without participating in the discriminator tag (one config class per backbone).
    """

    type: Literal[ModelType.PiDDecoder] = Field(default=ModelType.PiDDecoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        # An explicit `base` is validated against this class's Literal here, so it already narrows
        # identification to exactly one of the five PiD config classes.
        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _looks_like_pid_decoder(state_dict):
            raise NotAMatchError("state dict does not look like a PiD decoder (no 'lq_proj.*' keys)")

        # Imported lazily: it pulls in the vendored PiD network stack, which model identification has
        # no reason to load for the overwhelming majority of files.
        from invokeai.backend.pid.decode import required_pid_net_shapes

        contract = required_pid_net_shapes()
        shapes = pid_net_shapes(state_dict)

        # Everything from here to `_validate_base` is backbone-independent: each of these rejects a file
        # *every* PiD config class would reject for the same reason, which is exactly the case the plain
        # no-match signal cannot carry — no class matches, and the factory registers the file through its
        # `Unknown_Config` fallback. See `_raise_if_no_backbone_can_accept`.
        #
        # The latent projection carries both the architecture version and the backbone, so the checks
        # that read it can only speak when it is there. When it is not, the file is truncated, and the
        # contract check diagnoses that far better than a guess about the architecture would.
        if _LATENT_PROJ_KEY in shapes:
            _raise_if_discriminator_malformed(shapes, contract)
            _raise_if_architecture_unsupported(shapes)
            _raise_if_no_backbone_can_accept(shapes)
        _raise_if_pid_net_contract_unmet(shapes, contract)

        # Guaranteed by the checks above: the contract proved the weight is present and the malformed
        # check proved it is a conv. The backbone therefore always comes from the weights — the name
        # can only break the FLUX.1 / SD3 / Qwen-Image tie, never pick a backbone on its own.
        latent_channels = shapes[_LATENT_PROJ_KEY][1]  # type: ignore[index]
        components = _name_components(mod, override_fields)

        cls._validate_base(
            latent_channels=latent_channels,
            named_base=_backbone_from_components(components),
            had_base_override=override_fields.get("base") is not None,
        )

        base: BaseModelType = cls.model_fields["base"].default
        # Read, not popped: `override_fields` is built once by the factory and passed to every
        # candidate class, so consuming `variant` here would take it away from whichever PiD class
        # actually matches (which, without a `base` override, need not be this one).
        variant = override_fields.get("variant") or _variant_from_components(components, base)
        return cls(**{k: v for k, v in override_fields.items() if k != "variant"}, variant=variant)

    @classmethod
    def _validate_base(
        cls,
        *,
        latent_channels: int,
        named_base: BaseModelType | None,
        had_base_override: bool,
    ) -> None:
        """Confirm this checkpoint belongs to the config's pinned backbone.

        Every rejection here is a `NotAMatchError` and only ever means "not *this* backbone", which
        four of the five classes are supposed to say about every valid checkpoint. The reasons that
        would rule out all five are raised in ``from_model_on_disk`` before this runs.

        The latent channel count is authoritative and is the only thing separating SDXL (4ch) and
        FLUX.2 (128ch) from the 16ch family. FLUX.1, SD3 and Qwen-Image are architecturally
        identical, so within that family, in order of how much the evidence can be trusted:

        - an explicit ``base`` override wins outright. ``raise_for_override_fields`` has already
          validated it against this class's ``Literal``, so it names exactly one of the five, and
          whoever set it knows more than a filename anyone can write;
        - failing that, a name component naming exactly one of the three decides;
        - failing that, the family defaults to FLUX.1.
        """
        expected_base = cls.model_fields["base"].default
        # Guaranteed present: an unsupported channel count was rejected outright before this ran.
        candidate_bases = _LATENT_CHANNELS_TO_BASES[latent_channels]

        if expected_base not in candidate_bases:
            raise NotAMatchError(f"latent channels={latent_channels} do not match backbone {expected_base}")
        if len(candidate_bases) == 1 or had_base_override:
            return

        # A name pointing outside the family — a 16-channel file called "sdxl" — contradicts the
        # weights and is discarded rather than obeyed. Obeying it would have all three 16ch classes
        # reject the file, leaving a perfectly good decoder to the `Unknown_Config` fallback.
        if named_base not in candidate_bases:
            named_base = None

        if named_base is None:
            if expected_base is not BaseModelType.Flux:
                raise NotAMatchError("ambiguous 16-channel PiD checkpoint; defaulting to FLUX.1")
            return
        if named_base is not expected_base:
            raise NotAMatchError(f"name indicates {named_base}, not {expected_base}")


class PiDDecoder_Checkpoint_FLUX_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.1 backbone (16-channel latent)."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_Flux2_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.2 backbone (128-channel latent)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_SD3_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the Stable Diffusion 3 backbone (16-channel latent)."""

    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_SDXL_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the SDXL backbone (4-channel latent)."""

    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_QwenImage_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the Qwen-Image backbone (16-channel latent).

    Shares the 16-channel latent shape with FLUX.1 and SD3, so it relies on the same
    filename / directory-name disambiguation (or a trusted explicit ``base`` override)
    as SD3 - see ``_validate_base``.
    """

    base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")
