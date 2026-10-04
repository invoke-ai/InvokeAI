"""Regression tests for PiD decoder checkpoint identification.

Covers what identification has to get right before a checkpoint reaches the decode:

- The contract: identification holds a file to exactly the key set and shapes `load_pid_decoder`
  enforces. Checking a subset is not a milder version of the same guarantee — loaders run under
  `skip_torch_weight_init()`, so a weight the checkpoint does not supply is uninitialised memory
  rather than a default, and a file accepted here but refused there decodes nothing.
- Architecture: the LQ projection's width names the decoder generation (512 for v1, 1024 for v1.5),
  and the file is held to that generation's contract. A width no generation has is rejected as an
  architecture rather than reported as a pile of missing and unexpected keys.
- Backbone and variant: read from the weights where the weights can say, and from name evidence only
  for the FLUX.1 / SD3 / Qwen-Image tie the weights cannot break.

Anything rejected for a backbone-independent reason must be rejected *outright* (`InvalidMatchError`)
rather than fall through to the factory's `Unknown_Config` fallback — see
`TestUnusableCheckpointIsNeverRegistered`.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.pid_decoder import (
    _LATENT_PROJ_KEY,
    PiDDecoder_Checkpoint_Flux2_Config,
    PiDDecoder_Checkpoint_FLUX_Config,
    PiDDecoder_Checkpoint_QwenImage_Config,
    PiDDecoder_Checkpoint_SD3_Config,
    PiDDecoder_Checkpoint_SDXL_Config,
)
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, PiDDecoderVariantType
from invokeai.backend.pid.decode import BACKBONE_DISCRIMINATOR_KEY, required_pid_net_shapes
from invokeai.backend.pid.state_dict_utils import PiDVersion

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/pid.pth",
    "file_size": 1000,
    "name": "pid",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
    "base": "flux",
}

# What the model manager records for a single-file install from Hugging Face: the source still
# carries NVIDIA's directory name, and is trusted evidence because it is not a local filesystem path.
_HF_SOURCE_TYPE: dict[str, object] = {"source_type": "hf_repo_id"}


class _FakeShapeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


# NVIDIA's `.pth` files keep PidDistillModel's `net.` prefix; identification has to see through it.
_NET_PREFIX = "net."


def test_the_config_and_the_network_agree_on_the_discriminator_weight() -> None:
    """`pid_decoder` names the weight it reads the architecture and backbone from; `decode` names the
    one whose shape varies per backbone. A drift between the two silently unhooks identification."""
    assert _LATENT_PROJ_KEY == BACKBONE_DISCRIMINATOR_KEY


def _pid_state_dict(
    lq_hidden_dim: int | None = None, latent_channels: int = 16, version: PiDVersion = PiDVersion.V1
) -> dict[str, object]:
    """A complete PiD-looking state dict: every weight a *version* PidNet expects, at the shape it expects, with
    the discriminator conv overridden to the given latent channel count (and hidden dim, if given)."""
    contract = required_pid_net_shapes(version=version)
    sd: dict[str, object] = {f"{_NET_PREFIX}{k}": _FakeShapeTensor(*shape) for k, shape in contract.items()}
    width = contract[_LATENT_PROJ_KEY][0] if lq_hidden_dim is None else lq_hidden_dim
    sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(width, latent_channels, 3, 3)
    return sd


def _mock_mod(
    root: Path,
    state_dict: dict[str, object],
    dir_name: str | None = None,
    file_name: str = "model_ema_bf16.pth",
) -> MagicMock:
    """A ModelOnDisk stand-in. `dir_name` mimics NVIDIA's checkpoint directory; omit it for a direct
    single-file install, where the file lands in a UUID directory that says nothing about the model."""
    parent = root / dir_name if dir_name else root
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / file_name
    path.write_bytes(b"x")
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    return mod


def test_legacy_512_checkpoint_is_accepted() -> None:
    """A legacy 512-dim FLUX checkpoint (16 latent channels) identifies successfully. The counterweight
    to everything below: none of the strictness may make a real decoder harder to install."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict())
        config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "pid_decoder"
        assert config.base.value == "flux"


@pytest.mark.parametrize(
    ("config_class", "latent_channels", "file_name", "base"),
    [
        (
            PiDDecoder_Checkpoint_FLUX_Config,
            16,
            "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors",
            BaseModelType.Flux,
        ),
        (
            PiDDecoder_Checkpoint_QwenImage_Config,
            16,
            "pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
            BaseModelType.QwenImage,
        ),
        (
            PiDDecoder_Checkpoint_Flux2_Config,
            32,
            "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
            BaseModelType.Flux2,
        ),
    ],
)
def test_a_v1_5_checkpoint_identifies_under_comfy_org_s_name(
    config_class: type, latent_channels: int, file_name: str, base: BaseModelType
) -> None:
    """Comfy-Org ships the v1.5 decoders as single files named by input and output size; FLUX.2's is unpatchified
    to 32 channels before its projection. Only the 2K-to-4K preset exists for v1.5, and the name says so."""
    fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
    state_dict = _pid_state_dict(latent_channels=latent_channels, version=PiDVersion.V1_5)
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), state_dict, file_name=file_name)
        config = config_class.from_model_on_disk(mod, dict(fields))
    assert config.base is base
    assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x


@pytest.mark.parametrize("latent_channels", [4, 128])
def test_v1_5_has_no_sdxl_or_unpatchified_flux2_decoder(latent_channels: int) -> None:
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(latent_channels=latent_channels, version=PiDVersion.V1_5))
        with pytest.raises(InvalidMatchError, match=f"PiD v1.5 checkpoint has {latent_channels} latent channels"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_a_v1_checkpoint_at_v1_5_width_lacks_the_pit_injection() -> None:
    """The width selects the v1.5 contract, and a v1 file widened to it is missing what v1.5 adds."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(lq_hidden_dim=1024))
        with pytest.raises(InvalidMatchError, match="missing 5 of the weights required by PidNet"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize(
    "file_name",
    ["pid_1.5_qwenimage_1024_to_4096_bf16.safetensors", "PiD_v1pt5_res2kto4k_sr4x_qwenimage_undistilled.pth"],
)
def test_a_v1_5_teacher_is_rejected_by_its_name(file_name: str) -> None:
    """Teacher and 4-step student share every key and shape, and sampled with the student schedule a teacher
    decodes to a degraded image without any error. Comfy-Org's teacher files lack `_4step`; NVIDIA's say
    `undistilled`. The name is the only evidence, so it has to be heeded."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(version=PiDVersion.V1_5), file_name=file_name)
        with pytest.raises(InvalidMatchError, match="undistilled"):
            PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS, base="qwen-image"))


def test_a_v1_5_student_in_a_folder_named_like_a_comfy_file_is_accepted() -> None:
    """Comfy-Org's teacher spelling is read off file names: a local install is identified before it moves, so its
    parent is whatever folder the user keeps decoders in."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(
            Path(tmpdir),
            _pid_state_dict(version=PiDVersion.V1_5),
            dir_name="pid_1.5_decoders",
            file_name="pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
        )
        config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(
            mod, dict(_OVERRIDE_FIELDS, base="qwen-image")
        )
    assert config.base is BaseModelType.QwenImage


def test_a_v1_5_file_whose_name_names_no_preset_is_2k_to_4k() -> None:
    """v1.5 exists only as 2K-to-4K, so a renamed FLUX.1 file must not take the v1 FLUX default of 2K."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(version=PiDVersion.V1_5), file_name="decoder.safetensors")
        config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
    assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x


_INT8_LAYER = "patch_blocks.0.attn.qkv_x"


def _int8_v1_5_state_dict(
    weight_dtype: torch.dtype = torch.int8, scale_shape: tuple[int, ...] | None = (4608, 1)
) -> dict[str, object]:
    """A v1.5 contract with one Linear in Comfy-Org's `int8_tensorwise` layout, as identification reads it off a
    safetensors header: meta tensors with shapes and dtypes, and no data — the marker's bytes are not in it."""
    sd = _pid_state_dict(version=PiDVersion.V1_5)
    layer = f"{_NET_PREFIX}{_INT8_LAYER}"
    sd[f"{layer}.weight"] = torch.empty(4608, 1536, dtype=weight_dtype, device="meta")
    if scale_shape is not None:
        sd[f"{layer}.weight_scale"] = torch.empty(*scale_shape, device="meta")
    sd[f"{layer}.comfy_quant"] = torch.empty(72, dtype=torch.uint8, device="meta")
    return sd


def _int8_mod(root: Path, state_dict: dict[str, object], marker_format: str = "int8_tensorwise", group: int = 256):
    """`_mock_mod` over a safetensors file holding the marker's bytes, which is where identification reads them."""
    from safetensors.torch import save_file

    mod = _mock_mod(root, state_dict, file_name="pid_1.5_qwenimage_1024_to_4096_4step_int8_convrot.safetensors")
    marker = f'{{"format": "{marker_format}", "convrot": true, "convrot_groupsize": {group}}}'.encode()
    save_file(
        {f"{_NET_PREFIX}{_INT8_LAYER}.comfy_quant": torch.frombuffer(bytearray(marker), dtype=torch.uint8)}, mod.path
    )
    return mod


def test_an_int8_tensorwise_v1_5_checkpoint_identifies() -> None:
    """Its scales and markers are not PidNet parameters, but `load_pid_decoder` consumes them."""
    with TemporaryDirectory() as tmpdir:
        mod = _int8_mod(Path(tmpdir), _int8_v1_5_state_dict())
        config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(
            mod, dict(_OVERRIDE_FIELDS, base="qwen-image")
        )
    assert config.base is BaseModelType.QwenImage


def test_a_checkpoint_quantized_in_another_format_is_rejected() -> None:
    """The decode applies no fp8 scale, so such a layer would load off by its scale."""
    with TemporaryDirectory() as tmpdir:
        mod = _int8_mod(Path(tmpdir), _int8_v1_5_state_dict(weight_dtype=torch.float8_e4m3fn))
        with pytest.raises(InvalidMatchError, match=f"1 layer\\(s\\) other than as int8, e.g. '{_INT8_LAYER}'"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize(
    ("marker_format", "group", "scale_shape", "reason"),
    [
        ("float8_e4m3fn", 256, (4608, 1), "is marked float8_e4m3fn"),
        ("int8_tensorwise", 256, None, "is missing its weight_scale"),
        ("int8_tensorwise", 256, (48, 48), "Blockwise scale grids are not implemented"),
        ("int8_tensorwise", 512, (4608, 1), "power of 4"),
        ("int8_tensorwise", 1024, (4608, 1), "groups of 1024, which do not divide its 1536 inputs"),
    ],
    ids=["foreign_marker", "missing_scale", "blockwise_scale", "no_hadamard", "group_does_not_divide"],
)
def test_an_int8_build_the_loader_would_refuse_does_not_register(
    marker_format: str, group: int, scale_shape: tuple[int, ...] | None, reason: str
) -> None:
    """Identification accepts exactly what `load_pid_decoder` accepts: a file registered here fails at every decode.
    Each case is one the loader refuses, read off the header and the marker the file carries."""
    with TemporaryDirectory() as tmpdir:
        mod = _int8_mod(Path(tmpdir), _int8_v1_5_state_dict(scale_shape=scale_shape), marker_format, group)
        with pytest.raises(InvalidMatchError, match=reason):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_an_int8_build_whose_markers_cannot_be_read_does_not_register() -> None:
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _int8_v1_5_state_dict())
        with pytest.raises(InvalidMatchError, match="markers cannot be read"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_an_int8_weight_no_marker_claims_is_rejected() -> None:
    """Loaded as the float parameter it replaces, the raw codes would decode noise without any error."""
    sd = _int8_v1_5_state_dict()
    del sd[f"{_NET_PREFIX}patch_blocks.0.attn.qkv_x.comfy_quant"]
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), sd)
        with pytest.raises(InvalidMatchError, match="1 int8 weight\\(s\\) with no int8_tensorwise marker"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_an_lq_width_no_generation_has_is_rejected_as_an_architecture() -> None:
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(lq_hidden_dim=768))
        with pytest.raises(InvalidMatchError, match=r"lq_proj hidden dim 768; InvokeAI supports 512 \(v1\), 1024"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_unsupported_latent_channel_count_is_rejected() -> None:
    """No backbone uses 32 latent channels, so all five configs reject it for the same reason — which
    is exactly the case a plain no-match cannot carry, since it leaves the file to `Unknown_Config`."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(latent_channels=32))
        with pytest.raises(InvalidMatchError, match="32 latent channels"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


class TestPidNetContract:
    """Identification accepts exactly what `load_pid_decoder` accepts: the same 456 keys, the same
    shapes, and no extras.

    Checking only the LQ projection let a file with all 71 LQ weights and none of the 385 backbone
    weights be registered, to be refused at load time. That is not a milder version of the same
    guarantee: loaders run under `skip_torch_weight_init()`, so a weight the checkpoint does not
    supply is uninitialised memory rather than a default.

    Every rejection here is `InvalidMatchError`, not `NotAMatchError`: the file has already
    identified itself as a PiD checkpoint, so it must not fall through to the factory's
    `Unknown_Config` fallback (see `TestUnusableCheckpointIsNeverRegistered`).
    """

    @pytest.mark.parametrize(
        "dropped",
        [
            "lq_proj.output_heads.3.weight",
            "lq_proj.gate_modules.0.log_alpha",
            "lq_proj.latent_proj.3.block.2.bias",
        ],
    )
    def test_a_missing_lq_weight_is_rejected(self, dropped: str) -> None:
        sd = _pid_state_dict()
        del sd[f"{_NET_PREFIX}{dropped}"]
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 1 of the weights required by PidNet"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    @pytest.mark.parametrize("dropped", ["y_pos_embedding", "s_embedder.proj.weight", "final_layer.linear.weight"])
    def test_a_missing_backbone_weight_is_rejected(self, dropped: str) -> None:
        """The gap the LQ-only check left: 385 of the 456 weights were never looked at, so a
        checkpoint truncated anywhere outside the LQ projection installed cleanly and then failed."""
        sd = _pid_state_dict()
        assert f"{_NET_PREFIX}{dropped}" in sd, "fixture drifted from the real contract"
        del sd[f"{_NET_PREFIX}{dropped}"]
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 1 of the weights required by PidNet"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_an_unexpected_key_is_accepted(self) -> None:
        """Since issue #9437 `load_pid_decoder` ignores extra keys, so rejecting them here would
        refuse to install a file that loads fine — the installer must track the loader in both
        directions, not just the strict one."""
        sd = _pid_state_dict()
        sd[f"{_NET_PREFIX}not_a_pid_key"] = _FakeShapeTensor(1)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.base is BaseModelType.Flux

    def test_a_non_string_key_is_rejected(self) -> None:
        """The one extra key the loader genuinely cannot survive: `nn.Module.load_state_dict` calls
        `.startswith()` on every key, so a non-string one raises from inside torch. Identification
        has to keep refusing these even though it now accepts ordinary extras.

        Built bare, without the `net.` prefix: `strip_net_prefix` drops non-string keys when it has a
        prefix to strip, so only a bare checkpoint can carry one this far."""
        sd: dict[Any, Any] = {k: _FakeShapeTensor(*shape) for k, shape in required_pid_net_shapes().items()}
        sd[1] = _FakeShapeTensor(1)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="1 keys that are not strings"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_a_wrong_shaped_weight_is_rejected(self) -> None:
        """Right names, wrong tensors. Loading would fail on a size mismatch deep in the decode."""
        sd = _pid_state_dict()
        sd[f"{_NET_PREFIX}final_layer.linear.weight"] = _FakeShapeTensor(3, 3)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="shape PidNet cannot accept"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_a_base_pixdit_checkpoint_is_turned_away_before_any_of_this(self) -> None:
        """The base text-to-image weights the decoder is distilled from carry no `lq_proj` key at
        all, so they never reach the contract check — and get a better message than a key count."""
        sd = {k: v for k, v in _pid_state_dict().items() if "lq_proj" not in k}
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(NotAMatchError, match="does not look like a PiD decoder"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_marker_key_alone_is_not_enough(self) -> None:
        """The old behaviour: one `lq_proj.*` key identified a decoder, and the rest were tolerated."""
        sd = {f"{_NET_PREFIX}{_LATENT_PROJ_KEY}": _FakeShapeTensor(512, 16, 3, 3)}
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 455 of the weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_truncation_is_reported_as_truncation_even_without_the_diagnostic_weight(self) -> None:
        """The architecture and the backbone are both read from `lq_proj.latent_proj.0.weight`, so a
        file truncated past *that* weight used to fail with "cannot determine backbone" — accurate,
        but not the reason, and not the message the install flow promises for a truncated checkpoint.
        Those reads are skipped when the weight is absent rather than made to guess at it."""
        sd = {f"{_NET_PREFIX}lq_proj.latent_proj.1.weight": _FakeShapeTensor(1)}
        fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
        with TemporaryDirectory() as tmpdir:
            # No directory name and no base override: nothing but the weights identifies this file.
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 456 of the weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(fields))

    def test_a_rank_1_latent_projection_is_rejected(self) -> None:
        """Identification reads the architecture, the backbone and the kernel off this one weight.
        A tensor of the wrong rank made all three reads abstain at once, and the file fell through to
        name-only matching — which, given a name that supplied a backbone, accepted it."""
        sd = _pid_state_dict()
        sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(512)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd, dir_name="PiD_res2k_sr4x_official_flux_distill_4step")
            with pytest.raises(InvalidMatchError, match="malformed lq_proj.latent_proj.0.weight"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_distill_only_submodules_do_not_count(self) -> None:
        """`net_ema.*` shadows PidNet's own parameter names. The loader drops those submodules, so
        identification has to as well — otherwise a checkpoint carrying only the EMA copy would look
        complete here and then fail in `load_pid_decoder`."""
        sd: dict[str, object] = {
            f"net_ema.{k}": _FakeShapeTensor(*shape) for k, shape in required_pid_net_shapes().items()
        }
        sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(512, 16, 3, 3)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 455 of the weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def _write_pid_checkpoint(root: Path, state_dict: dict[Any, object]) -> Path:
    """Write a state dict as a real `.pth`, the format NVIDIA ships, so the factory reaches it through
    the same pickle-scan-and-`torch.load` path a real download would."""
    path = root / "model_ema_bf16.pth"
    torch.save(state_dict, path)
    return path


def _real_pid_state_dict(
    lq_hidden_dim: int | None = None, latent_channels: int = 16, version: PiDVersion = PiDVersion.V1
) -> dict[str, object]:
    """`_pid_state_dict` with tensors that can actually be serialised.

    PidNet is ~5.5 GB in float32, so every weight is a zero-stride view onto one shared scalar: the
    shapes are the real ones, the file is ~50 KB, and `torch.save` deduplicates the storage.
    """
    scalar = torch.zeros(())
    contract = required_pid_net_shapes(version=version)
    sd: dict[str, object] = {f"{_NET_PREFIX}{k}": scalar.expand(shape) for k, shape in contract.items()}
    width = contract[_LATENT_PROJ_KEY][0] if lq_hidden_dim is None else lq_hidden_dim
    sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = scalar.expand(width, latent_channels, 3, 3)
    return sd


class TestUnusableCheckpointIsNeverRegistered:
    """Rejecting an unusable checkpoint in the config class is only half the job.

    Every config class signals "not mine" with `NotAMatchError`, which the factory collects and then,
    with `allow_unknown_models` (default: true), papers over by returning `Unknown_Config`. A file that
    identified itself as a PiD decoder and was then found unusable would therefore still be installed —
    as an unknown model, with a database record, failing only when something tried to load it.
    `InvalidMatchError` is what makes the rejection stick, and every reason that would rule out *all
    five* backbone configs has to raise it, not just the truncation case.
    """

    def _partial(self) -> dict[str, object]:
        """A single `lq_proj.*` weight: enough to be recognised, far from loadable."""
        return {f"{_NET_PREFIX}lq_proj.latent_proj.1.weight": torch.zeros(1)}

    def _missing_backbone_weight(self) -> dict[str, object]:
        """Every LQ weight present, one backbone weight gone — what the LQ-only check let through."""
        sd = _real_pid_state_dict()
        del sd[f"{_NET_PREFIX}final_layer.linear.weight"]
        return sd

    def _truncated_v1_5(self) -> dict[str, object]:
        """A v1.5 file is held to the v1.5 contract, down to the one weight only its PiT injection has."""
        sd = _real_pid_state_dict(version=PiDVersion.V1_5)
        del sd[f"{_NET_PREFIX}pit_lq_gate.log_alpha"]
        return sd

    def _unknown_architecture(self) -> dict[str, object]:
        """Intact but for an LQ width no generation has: every config class would reject it for that reason."""
        return _real_pid_state_dict(lq_hidden_dim=768)

    def _marker_beside_a_float_weight(self) -> dict[str, object]:
        """A quantization marker on a layer whose weight is not int8 names a scale the decode would never apply."""
        sd = _real_pid_state_dict(version=PiDVersion.V1_5)
        sd[f"{_NET_PREFIX}patch_blocks.0.attn.qkv_x.weight_scale"] = torch.ones(())
        sd[f"{_NET_PREFIX}patch_blocks.0.attn.qkv_x.comfy_quant"] = torch.zeros(28, dtype=torch.uint8)
        return sd

    def _unsupported_latent_channels(self) -> dict[str, object]:
        return _real_pid_state_dict(latent_channels=32)

    def _malformed_discriminator(self) -> dict[str, object]:
        sd = _real_pid_state_dict()
        sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = torch.zeros(512)
        return sd

    def _bare_with_a_non_string_key(self) -> dict[Any, object]:
        """A complete *bare* contract plus two keys PidNet does not expect, one of them not a string.

        A bare checkpoint is passed through `strip_net_prefix` untouched, so a `.pth` can hand
        identification whatever it was pickled with. Since issue #9437 the plain `not_a_pid_key`
        is accepted (the loader ignores it), but `1` is not: `load_state_dict` calls `.startswith()`
        on every key. Reporting it sorts the offenders, and sorting a mixed set raises TypeError —
        which the factory catches as a generic candidate failure and answers with the Unknown_Config
        registration this class is about. A crash in an unusability check therefore does not fail
        loudly; it fails as a silent accept, so the sort stays `key=str`.
        """
        scalar = torch.zeros(())
        sd: dict[Any, object] = {k: scalar.expand(shape) for k, shape in required_pid_net_shapes().items()}
        sd[1] = torch.zeros(1)
        sd["not_a_pid_key"] = torch.zeros(1)
        return sd

    @pytest.mark.parametrize(
        ("case", "expected_reason"),
        [
            ("_partial", "missing 456 of the weights"),
            ("_missing_backbone_weight", "missing 1 of the weights required by PidNet"),
            ("_truncated_v1_5", "missing 1 of the weights required by PidNet"),
            ("_unknown_architecture", "lq_proj hidden dim 768"),
            ("_marker_beside_a_float_weight", "quantizes 1 layer(s) other than as int8"),
            ("_unsupported_latent_channels", "32 latent channels"),
            ("_malformed_discriminator", "malformed lq_proj.latent_proj.0.weight"),
            ("_bare_with_a_non_string_key", "1 keys that are not strings"),
        ],
    )
    def test_factory_returns_no_config_even_with_allow_unknown(self, case: str, expected_reason: str) -> None:
        with TemporaryDirectory() as tmpdir:
            path = _write_pid_checkpoint(Path(tmpdir), getattr(self, case)())
            result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

        assert result.config is None, "a recognised-but-unusable checkpoint must not be registered"
        assert not any(isinstance(r, Unknown_Config) for r in result.details.values())
        # The reason survives for `_probe` to report, so the user is told what is actually wrong
        # instead of the misleading "could not identify model".
        assert result.invalid_matches
        assert expected_reason in str(result.invalid_matches[0])

    @pytest.mark.parametrize("version", list(PiDVersion))
    def test_a_valid_checkpoint_still_identifies(self, version: PiDVersion) -> None:
        """The counterweight: none of the above may make a real decoder of either generation harder to install."""
        with TemporaryDirectory() as tmpdir:
            path = _write_pid_checkpoint(Path(tmpdir), _real_pid_state_dict(version=version))
            result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

        assert result.config is not None
        assert not result.invalid_matches
        assert result.config.base is BaseModelType.Flux


class TestBackboneFromInstallSource:
    """FLUX.1, SD3 and Qwen-Image PiD decoders are architecturally identical (16 latent channels), so the
    backbone can only come from the name. A direct single-file install has none — but its source does."""

    _SD3_SOURCE = "nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth"
    # A neutral stand-in for the UUID directory a direct install lands in, so a fluke match in the
    # TemporaryDirectory name cannot shadow the source.
    _UUID_DIR = "checkpoints"

    def test_source_identifies_sd3_without_a_base_override(self) -> None:
        """Without this, a directly installed SD3 decoder is recorded as FLUX and then rejected by the
        SD3 decode node."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name=self._UUID_DIR)
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {
                "source": self._SD3_SOURCE,
                **_HF_SOURCE_TYPE,
            }
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(overrides))
            assert config.base is BaseModelType.StableDiffusion3

    def test_flux_config_rejects_a_checkpoint_the_source_names_as_sd3(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name=self._UUID_DIR)
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {
                "source": self._SD3_SOURCE,
                **_HF_SOURCE_TYPE,
            }
            with pytest.raises(NotAMatchError, match="name indicates"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(overrides))

    def test_qwen_image_source_is_recognised(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name=self._UUID_DIR)
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {
                "source": "nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
                **_HF_SOURCE_TYPE,
            }
            config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(overrides))
            assert config.base is BaseModelType.QwenImage

    def test_base_override_still_wins_when_nothing_names_the_backbone(self) -> None:
        """The starter installer's explicit base remains the fallback for a fully anonymous file."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict())
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, base="sd-3", source="local file")
            )
            assert config.base is BaseModelType.StableDiffusion3

    def test_base_override_beats_a_name_that_says_otherwise(self) -> None:
        """A filename is something anyone can write; an explicit base has already been validated
        against this class's Literal, so it names exactly one of the five and is trusted first."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name="PiD_res2k_sr4x_official_flux_distill_4step")
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS, base="sd-3"))
            assert config.base is BaseModelType.StableDiffusion3


class TestNameEvidence:
    """The name only ever breaks the 16-channel FLUX.1 / SD3 / Qwen-Image tie — the weights decide
    everything else. These pin how the name is read when it is consulted."""

    def test_incidental_backbone_substring_in_parent_directory_is_ignored(self) -> None:
        """An unrelated directory containing the letters ``sd3`` must not name the checkpoint's backbone."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "modelsd3cache"
            checkpoint_dir.mkdir()
            path = _write_pid_checkpoint(checkpoint_dir, _real_pid_state_dict())
            result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

        assert result.config is not None
        assert result.config.base is BaseModelType.Flux

    def test_the_filename_beats_the_parent_directory(self) -> None:
        """The reported case. Concatenating every name component into one string and substring-matching
        it let a fixed backbone precedence decide what the name had already answered: `/flux/model_sd3.pth`
        matched `flux` first and was registered as FLUX although the file itself says sd3."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name="flux", file_name="model_sd3.pth")
            fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(fields))
            assert config.base is BaseModelType.StableDiffusion3

    def test_one_component_naming_two_backbones_decides_nothing(self) -> None:
        """Two different backbones in one name is not a precedence question — it is a name that
        decides nothing, and the 16ch family falls back to its FLUX.1 default."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), file_name="pid_sd3_and_flux.pth")
            fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
            with pytest.raises(NotAMatchError, match="ambiguous 16-channel"):
                PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(fields))
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(fields))
            assert config.base is BaseModelType.Flux

    def test_flux2_is_a_spelling_of_flux2_not_an_ambiguity(self) -> None:
        """`flux2` contains `flux`; consuming the specific spelling first keeps that a precedence
        question between two spellings of one answer rather than a two-backbone tie."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), file_name="pid_flux2_sr4x.pth")
            fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
            with pytest.raises(NotAMatchError, match="latent channels=16 do not match"):
                PiDDecoder_Checkpoint_Flux2_Config.from_model_on_disk(mod, dict(fields))
            # 16 channels rule FLUX.2 out on the weights; the name still must not make this FLUX.1.
            with pytest.raises(NotAMatchError, match="ambiguous 16-channel"):
                PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(fields))

    def test_a_local_path_source_is_not_name_evidence(self) -> None:
        """The model manager sets `source` to the file's own path when there is no remote one, so
        trusting it would mean matching against arbitrary ancestor directories of the user's model
        library. Nothing is lost: a local install is identified before it is moved."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict())
            fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {
                "source": "D:/sd3-models/pid/model_ema_bf16.pth",
                "source_type": "path",
            }
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(fields))
            assert config.base is BaseModelType.Flux

    def test_a_name_pointing_outside_the_family_is_discarded_not_obeyed(self) -> None:
        """A 16-channel file called `sdxl` contradicts its own weights. Obeying the name would have
        all three 16ch classes reject it, leaving a perfectly good decoder to `Unknown_Config`."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), file_name="pid_sdxl_sr4x.pth")
            fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(fields))
            assert config.base is BaseModelType.Flux


class TestVariantIdentification:
    """The variant is read from NVIDIA's directory name where there is one, and falls back to the
    backbone's only published preset otherwise (SDXL and Qwen-Image ship 2K-to-4K only)."""

    def test_directory_name_wins(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name="PiD_res2kto4k_sr4x_official_flux_distill_4step")
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_install_source_is_used_when_the_stored_name_is_silent(self) -> None:
        """A direct single-file install lands in a UUID directory, but the HF source still names the preset."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name="checkpoints")
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(
                mod,
                dict(
                    _OVERRIDE_FIELDS,
                    source="nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
                    **_HF_SOURCE_TYPE,
                ),
            )
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_a_local_path_source_does_not_name_the_preset(self) -> None:
        """Same reasoning as for the backbone: `source` is the file's own path for a local install,
        so an ancestor directory must not decide the record's resolution preset."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict())
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, source="E:/res2kto4k-models/pid/model_ema_bf16.pth", source_type="path")
            )
            assert config.variant is PiDDecoderVariantType.Res2k_Sr4x

    def test_flux_defaults_to_2k_when_the_name_is_silent(self) -> None:
        """FLUX.1 ships both presets, so a nameless single-file install keeps the 2K default."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict())
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
            assert config.variant is PiDDecoderVariantType.Res2k_Sr4x

    def test_sdxl_single_file_install_gets_the_only_published_preset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(latent_channels=4))
            config = PiDDecoder_Checkpoint_SDXL_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS, base="sdxl"))
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_qwen_image_single_file_install_gets_the_only_published_preset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict())
            config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, base="qwen-image")
            )
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_explicit_variant_override_wins(self) -> None:
        """A starter-model install passes the variant it knows it is downloading."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(), dir_name="PiD_res2kto4k_sr4x_official_flux_distill_4step")
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, variant=PiDDecoderVariantType.Res2k_Sr4x)
            )
            assert config.variant is PiDDecoderVariantType.Res2k_Sr4x
