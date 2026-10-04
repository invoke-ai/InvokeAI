"""Model configs for PiD (Pixel Diffusion Decoder) checkpoints.

PiD decoders are released by NVIDIA at https://huggingface.co/nvidia/PiD and
ship per supported backbone (FLUX.1, FLUX.2, SD3, SDXL, Qwen-Image). Most
backbones offer two resolution presets (`res2k_sr4x_*` and `res2kto4k_sr4x_*`),
while SDXL and Qwen-Image ship only the `res2kto4k_sr4x_*` preset. The second
generation, PiD v1.5, exists for FLUX.1, FLUX.2 and Qwen-Image in the 2K-to-4K
preset; Comfy-Org repackages both generations as single safetensors files. See
`LICENSE-PiD.txt` at the repo root — code is Apache-2.0, weights are NSCLv1
(non-commercial / research).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.backbone_names import backbone_from_components, name_components
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
    ModelType,
    PiDDecoderVariantType,
)
from invokeai.backend.pid.state_dict_utils import (
    PID_VERSION_BY_LQ_HIDDEN_DIM,
    PiDVersion,
    pid_net_shapes,
    strip_net_prefix,
)

# Marker substring produced by `PidNet.lq_proj` (see
# invokeai/backend/pid/_src/networks/pid_net.py). The pretrained PixDiT_T2I
# weights do not contain `lq_proj`, so its presence in any key is diagnostic
# of a PiD-style checkpoint. We match by substring (not prefix) because the
# official `.pth` files keep PidDistillModel's `net.` prefix, so keys look
# like `net.lq_proj.layers.0.weight`.
_PID_MARKER_SUBSTRING = "lq_proj"


def _looks_like_pid_decoder(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(k, str) and _PID_MARKER_SUBSTRING in k for k in state_dict)


# PidNet's latent input projection: a Conv2d of shape (lq_hidden_dim, latent input channels, 3, 3).
# Identification reads three separate facts off this one weight — the architecture version (dim 0),
# the backbone (dim 1) and, via the contract, its kernel — which is why it is worth naming.
_LATENT_PROJ_KEY = "lq_proj.latent_proj.0.weight"

# dim 1 of the latent projection is the backbone's latent channel count as the projection takes it. It is the only
# architectural dimension that varies between backbones, and therefore the only name-independent discriminator
# available. FLUX.1, SD3 and Qwen-Image are architecturally identical and share 16 channels; nothing in the weights
# can separate them. v1.5 unpatchifies FLUX.2's 128 channels to 32 before the projection, and NVIDIA ships it for
# FLUX.1, FLUX.2 and Qwen-Image only.
_LATENT_CHANNELS_TO_BASES: dict[PiDVersion, dict[int, set[BaseModelType]]] = {
    PiDVersion.V1: {
        4: {BaseModelType.StableDiffusionXL},
        16: {BaseModelType.Flux, BaseModelType.StableDiffusion3, BaseModelType.QwenImage},
        128: {BaseModelType.Flux2},
    },
    PiDVersion.V1_5: {
        16: {BaseModelType.Flux, BaseModelType.QwenImage},
        32: {BaseModelType.Flux2},
    },
}

# Keyed by `Any`, not `str`: a bare checkpoint reaches identification with its keys untouched, so a
# `.pth` is free to supply keys that are not strings (see `strip_net_prefix`).
_Shapes = Mapping[Any, tuple[int, ...] | None]


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


def _pid_version(shapes: _Shapes) -> PiDVersion:
    """The decoder generation, read off the latent projection's width; reject a width no generation has.

    Runs before the contract check, which depends on it, so the diagnosis is the accurate one: judged against
    either generation's contract, an unknown architecture would be reported as a pile of missing and unexpected
    keys rather than as the architecture it is.
    """
    lq_hidden_dim = shapes[_LATENT_PROJ_KEY][0]  # type: ignore[index]  # rank checked above
    if (version := PID_VERSION_BY_LQ_HIDDEN_DIM.get(lq_hidden_dim)) is None:
        supported = ", ".join(f"{dim} ({v.value})" for dim, v in PID_VERSION_BY_LQ_HIDDEN_DIM.items())
        raise InvalidMatchError(f"PiD decoder has lq_proj hidden dim {lq_hidden_dim}; InvokeAI supports {supported}.")
    return version


def _raise_if_no_backbone_can_accept(shapes: _Shapes, version: PiDVersion) -> None:
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
    bases_by_channels = _LATENT_CHANNELS_TO_BASES[version]
    if channels not in bases_by_channels:
        supported = ", ".join(
            f"{count} for {'/'.join(sorted(base.value for base in bases))}"
            for count, bases in bases_by_channels.items()
        )
        raise InvalidMatchError(
            f"PiD {version.value} checkpoint has {channels} latent channels; no supported backbone uses this "
            f"(supported: {supported})"
        )


def _and_more(items: list[Any]) -> str:
    return f" (+ {len(items) - 5} more)" if len(items) > 5 else ""


def _raise_if_pid_net_contract_unmet(shapes: _Shapes, contract: Mapping[str, tuple[int, ...]]) -> None:
    """Hold the checkpoint to exactly the contract `load_pid_decoder` enforces.

    Checking only the LQ projection accepted a file that carried every LQ weight and none of the 385
    backbone weights; the loader then refused it. A subset check is not a milder version of the same
    guarantee — loaders run under `skip_torch_weight_init()`, so a weight the checkpoint does not
    supply is uninitialised memory rather than a default.

    Missing keys are fatal here because they are fatal there, which is what makes installation and
    loading accept the same set of files. A stricter installer cannot reject a file that would have
    loaded: the loader already refuses everything rejected here.

    Extra keys are *not* fatal — `load_pid_decoder` ignores them (issue #9437), so rejecting them
    here would refuse to install a file that loads fine. The one kind of extra key the loader still
    cannot survive is a non-string one, which makes `nn.Module.load_state_dict` raise from inside
    torch, so that is the extra this check keeps.

    `_LATENT_PROJ_KEY` is excluded from the shape comparison, and only from that: it is the one
    parameter whose shape legitimately varies by backbone, and its variable dimensions each have a
    dedicated check above with a dedicated message.
    """
    # No "this is a base PixDiT_T2I checkpoint" special case, unlike `load_pid_decoder`: those weights
    # carry no `lq_proj` key at all, so such a file never reaches here — `_looks_like_pid_decoder`
    # has already turned it away, and with a better message.
    # Both sorts take `key=str`: a bare checkpoint's keys need not all be strings (see
    # `strip_net_prefix`), and sorting a mixed set raises TypeError — which the factory answers with
    # the `Unknown_Config` registration these checks exist to prevent, so the crash fails as a silent
    # accept rather than loudly.
    if missing := sorted(contract.keys() - shapes.keys(), key=str):
        raise InvalidMatchError(
            f"PiD checkpoint is missing {len(missing)} of the weights required by PidNet; the file is "
            f"incomplete and cannot be used as a PiD decoder: {missing[:5]}{_and_more(missing)}"
        )

    if not_strings := sorted((k for k in shapes if not isinstance(k, str)), key=str):
        raise InvalidMatchError(
            f"PiD checkpoint has {len(not_strings)} keys that are not strings and so cannot name a "
            f"PidNet parameter, which `load_pid_decoder` rejects too: "
            f"{not_strings[:5]}{_and_more(not_strings)}"
        )

    mismatched = [(k, shapes[k], want) for k, want in contract.items() if k != _LATENT_PROJ_KEY and shapes[k] != want]
    if mismatched:
        k, got, want = mismatched[0]
        raise InvalidMatchError(
            f"PiD checkpoint has {len(mismatched)} weights whose shape PidNet cannot accept "
            f"(e.g. {k}: {got}, expected {want}); loading it would fail with a size mismatch"
        )


# Backbones for which NVIDIA ships exactly one preset — for these the variant is known even when the
# name gives nothing away. FLUX.1 / FLUX.2 / SD3 ship both presets and fall back to `Res2k_Sr4x`.
_SINGLE_VARIANT_BACKBONES: dict[BaseModelType, PiDDecoderVariantType] = {
    BaseModelType.StableDiffusionXL: PiDDecoderVariantType.Res2kTo4k_Sr4x,
    BaseModelType.QwenImage: PiDDecoderVariantType.Res2kTo4k_Sr4x,
}


def _variant_from_components(
    components: tuple[str, ...], base: BaseModelType, version: PiDVersion
) -> PiDDecoderVariantType:
    """Map NVIDIA's `res2k_sr4x` / `res2kto4k_sr4x` name slice to a variant.

    Same specificity ordering as the backbone match. If no component names a preset, fall back to the
    only published one where there is one — every v1.5 decoder, and SDXL's and Qwen-Image's v1 — and to
    ``Res2k_Sr4x`` for the v1 backbones shipping both.
    """
    for component in components:
        n = component.lower()
        # `res2kto4k` contains `res2k`, so the 2K-to-4K spellings are tested first. Comfy-Org names the preset by
        # its input and output sizes instead (`pid_1.5_flux1_1024_to_4096_4step_bf16`).
        if "res2kto4k" in n or "res2k_to_4k" in n or "res2k_to4k" in n or "1024_to_4096" in n:
            return PiDDecoderVariantType.Res2kTo4k_Sr4x
        if "res2k" in n:
            return PiDDecoderVariantType.Res2k_Sr4x
    if version is PiDVersion.V1_5:
        return PiDDecoderVariantType.Res2kTo4k_Sr4x
    return _SINGLE_VARIANT_BACKBONES.get(base, PiDDecoderVariantType.Res2k_Sr4x)


def _raise_if_named_undistilled(components: tuple[str, ...], folder_name: str) -> None:
    """Reject a v1.5 decoder whose name marks it as NVIDIA's undistilled teacher.

    NVIDIA published each v1.5 decoder twice: the 4-step distilled student the decode nodes sample, and the
    undistilled teacher (`PiD_v1pt5_*_undistilled`, Comfy-Org's `pid_1.5_*_bf16` without `_4step`), which is
    sampled with ~25 CFG steps. Both carry the same keys and shapes, so only the name tells them apart — and
    sampled with the student schedule a teacher decodes to a degraded image without any error. A name that says
    neither is accepted.

    Comfy-Org's spelling marks the teacher by what its file name lacks, so it is only read off file names — the
    last segment of the file's own name or of its install source — never off ``folder_name``, the folder a local
    install is identified in, which a user may well have named `pid_1.5_decoders`.
    """
    for component in components:
        n = component.lower()
        file_name = "" if component == folder_name else n.replace("\\", "/").rsplit("/", 1)[-1]
        if "undistilled" in n or (file_name.startswith("pid_1.5_") and "4step" not in file_name):
            raise InvalidMatchError(
                f"PiD v1.5 checkpoint {component!r} is named as an undistilled (teacher) decoder. InvokeAI samples "
                "PiD with the distilled 4-step schedule; install the `_4step` build instead."
            )


def _int8_sidecar_keys(state_dict: dict[Any, Any], path: Path) -> set[str]:
    """The keys of Comfy-Org's `int8_tensorwise` side channel, which the contract does not list; reject a quantization
    `load_pid_decoder` would refuse.

    Held to what the loader accepts, so an int8 build that registers also loads: every marked layer needs an int8
    weight, a per-row or per-tensor `weight_scale`, and a marker declaring `int8_tensorwise` with a rotation group the
    decode has a Hadamard for and that divides the layer's inputs; every int8 weight needs a marker. Anything else
    decodes garbage or fails at every decode, and every config class would turn the file away for the same reason.

    Dtypes and shapes come from the header identification reads. The markers' JSON does not — a header has no tensor
    data — so it is read from the safetensors file itself, once the dtypes have shown the file is worth reading.
    """
    import torch

    from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX
    from invokeai.backend.quantization.int8_convrot import (
        CONVROT_GROUP_SIZE,
        INT8_TENSORWISE_FORMAT,
        check_hadamard_size,
        check_int8_scale_layout,
        read_comfy_quant_markers,
    )

    stripped = strip_net_prefix(state_dict)
    marked = {k[: -len(COMFY_QUANT_SUFFIX)] for k in stripped if isinstance(k, str) and k.endswith(COMFY_QUANT_SUFFIX)}
    if not_int8 := sorted(
        layer for layer in marked if getattr(stripped.get(f"{layer}.weight"), "dtype", None) is not torch.int8
    ):
        raise InvalidMatchError(
            f"PiD checkpoint quantizes {len(not_int8)} layer(s) other than as int8, e.g. '{not_int8[0]}'; only "
            "int8_tensorwise builds are supported."
        )
    if unmarked := sorted(
        k
        for k, v in stripped.items()
        if getattr(v, "dtype", None) is torch.int8
        and not (isinstance(k, str) and k.endswith(".weight") and k[: -len(".weight")] in marked)
    ):
        raise InvalidMatchError(
            f"PiD checkpoint has {len(unmarked)} int8 weight(s) with no int8_tensorwise marker, e.g. {unmarked[:3]}."
        )
    if not marked:
        return set()

    try:
        markers = strip_net_prefix(read_comfy_quant_markers(path))
    except Exception as e:
        raise InvalidMatchError(
            f"PiD checkpoint marks {len(marked)} layer(s) int8, but its markers cannot be read from {path.name}: {e}"
        ) from e
    for layer in sorted(marked):
        weight, scale, marker = (
            stripped[f"{layer}.weight"],
            stripped.get(f"{layer}.weight_scale"),
            markers.get(layer, {}),
        )
        try:
            if marker.get("format") != INT8_TENSORWISE_FORMAT:
                raise ValueError(f"'{layer}' is marked {marker.get('format') or 'with an unreadable marker'}")
            if scale is None:
                raise ValueError(f"'{layer}' is missing its weight_scale")
            check_int8_scale_layout(layer, weight, scale)
            if marker.get("convrot", False):
                group = int(marker.get("convrot_groupsize", CONVROT_GROUP_SIZE))
                check_hadamard_size(group)
                if weight.shape[-1] % group:
                    raise ValueError(
                        f"'{layer}' rotates groups of {group}, which do not divide its {weight.shape[-1]} inputs"
                    )
        except ValueError as e:
            raise InvalidMatchError(f"PiD int8 checkpoint cannot be loaded: {e}") from e
    return {f"{layer}{suffix}" for layer in marked for suffix in (COMFY_QUANT_SUFFIX, ".weight_scale")}


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

        sidecars = _int8_sidecar_keys(state_dict, mod.path)
        shapes = {k: v for k, v in pid_net_shapes(state_dict).items() if k not in sidecars}

        # Everything from here to `_validate_base` is backbone-independent: each of these rejects a file
        # *every* PiD config class would reject for the same reason, which is exactly the case the plain
        # no-match signal cannot carry — no class matches, and the factory registers the file through its
        # `Unknown_Config` fallback. See `_raise_if_no_backbone_can_accept`.
        #
        # The latent projection carries both the architecture version and the backbone, so the checks
        # that read it can only speak when it is there. When it is not, the file is truncated, and the
        # contract check diagnoses that far better than a guess about the architecture would.
        version = PiDVersion.V1
        if _LATENT_PROJ_KEY in shapes:
            # Both generations give this conv the same rank and kernel, so either contract judges its form.
            _raise_if_discriminator_malformed(shapes, required_pid_net_shapes(version=PiDVersion.V1))
            version = _pid_version(shapes)
            _raise_if_no_backbone_can_accept(shapes, version)
        _raise_if_pid_net_contract_unmet(shapes, required_pid_net_shapes(version=version))

        # Guaranteed by the checks above: the contract proved the weight is present and the malformed
        # check proved it is a conv. The backbone therefore always comes from the weights — the name
        # can only break the FLUX.1 / SD3 / Qwen-Image tie, never pick a backbone on its own.
        latent_channels = shapes[_LATENT_PROJ_KEY][1]  # type: ignore[index]
        components = name_components(mod, override_fields)
        if version is PiDVersion.V1_5:
            _raise_if_named_undistilled(components, mod.path.parent.name)

        cls._validate_base(
            latent_channels=latent_channels,
            version=version,
            named_base=backbone_from_components(components),
            had_base_override=override_fields.get("base") is not None,
        )

        base: BaseModelType = cls.model_fields["base"].default
        # Read, not popped: `override_fields` is built once by the factory and passed to every
        # candidate class, so consuming `variant` here would take it away from whichever PiD class
        # actually matches (which, without a `base` override, need not be this one).
        variant = override_fields.get("variant") or _variant_from_components(components, base, version)
        return cls(**{k: v for k, v in override_fields.items() if k != "variant"}, variant=variant)

    @classmethod
    def _validate_base(
        cls,
        *,
        latent_channels: int,
        version: PiDVersion,
        named_base: BaseModelType | None,
        had_base_override: bool,
    ) -> None:
        """Confirm this checkpoint belongs to the config's pinned backbone.

        Every rejection here is a `NotAMatchError` and only ever means "not *this* backbone", which
        four of the five classes are supposed to say about every valid checkpoint. The reasons that
        would rule out all five are raised in ``from_model_on_disk`` before this runs.

        The latent channel count is authoritative and is the only thing separating SDXL (4ch) and
        FLUX.2 (128ch; 32ch in v1.5) from the 16ch family. FLUX.1, SD3 and Qwen-Image are
        architecturally identical, so within that family, in order of how much the evidence can be
        trusted:

        - an explicit ``base`` override wins outright. ``raise_for_override_fields`` has already
          validated it against this class's ``Literal``, so it names exactly one of the five, and
          whoever set it knows more than a filename anyone can write;
        - failing that, a name component naming exactly one of the family decides;
        - failing that, the family defaults to FLUX.1.
        """
        expected_base = cls.model_fields["base"].default
        # Guaranteed present: an unsupported channel count was rejected outright before this ran.
        candidate_bases = _LATENT_CHANNELS_TO_BASES[version][latent_channels]

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
    """PiD decoder for the FLUX.2 backbone (128-channel latent; PiD v1.5 projects it unpatchified to 32)."""

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
