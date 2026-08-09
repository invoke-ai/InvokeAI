"""Regression tests for PiD decoder checkpoint identification.

Covers the three things identification has to get right before a checkpoint reaches the decode:

- Architecture: InvokeAI's build_pid_net constructs only the legacy 512-dim PidNet. NVIDIA's v1.5
  decoders use lq_hidden_dim=1024 (plus PiT injection, scalar gates, ...), which cannot be loaded into
  it, so such a checkpoint must be rejected here instead of crashing with a size mismatch mid-decode.
- Completeness: models are built under `skip_torch_weight_init()`, so a checkpoint that carries only
  part of the LQ projection would leave uninitialised Conv/Linear weights and decode to garbage. Such
  a file must be rejected outright rather than fall through to the factory's `Unknown_Config`.
- Variant: the record's resolution preset, including the direct single-file install case where the
  name carries no `res2k` / `res2kto4k` marker at all.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.pid_decoder import (
    PiDDecoder_Checkpoint_FLUX_Config,
    PiDDecoder_Checkpoint_QwenImage_Config,
    PiDDecoder_Checkpoint_SD3_Config,
    PiDDecoder_Checkpoint_SDXL_Config,
    _lq_hidden_dim_from_state_dict,
)
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, PiDDecoderVariantType
from invokeai.backend.pid.decode import required_lq_proj_keys

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


class _FakeShapeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


# NVIDIA's `.pth` files keep PidDistillModel's `net.` prefix; identification has to see through it.
_NET_PREFIX = "net."

# The diagnostic input Conv: dim-0 is the lq_hidden_dim, dim-1 the backbone's latent channel count.
_LATENT_PROJ_KEY = "lq_proj.latent_proj.0.weight"


def _pid_state_dict(lq_hidden_dim: int, latent_channels: int = 16) -> dict[str, object]:
    """A complete PiD-looking state dict: every LQ projection weight PidNet expects, plus the
    diagnostic input Conv shaped for the given hidden dim / latent channel count."""
    sd: dict[str, object] = {f"{_NET_PREFIX}{k}": _FakeShapeTensor(1) for k in required_lq_proj_keys()}
    sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(lq_hidden_dim, latent_channels, 3, 3)
    return sd


def test_lq_hidden_dim_read_from_state_dict() -> None:
    assert _lq_hidden_dim_from_state_dict(_pid_state_dict(512)) == 512
    assert _lq_hidden_dim_from_state_dict(_pid_state_dict(1024)) == 1024
    assert _lq_hidden_dim_from_state_dict({}) is None


def _mock_mod(root: Path, state_dict: dict[str, object], dir_name: str | None = None) -> MagicMock:
    """A ModelOnDisk stand-in. `dir_name` mimics NVIDIA's checkpoint directory; omit it for a direct
    single-file install, where the file lands in a UUID directory that says nothing about the model."""
    parent = root / dir_name if dir_name else root
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / "model_ema_bf16.pth"
    path.write_bytes(b"x")
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    return mod


def test_v1_5_checkpoint_is_rejected_at_identification() -> None:
    """A 1024-dim (v1.5) checkpoint must be rejected, not accepted and crashed on later.

    A plain `NotAMatchError`, deliberately: the file is intact, InvokeAI just cannot build its shape,
    so it should still be registrable as an unknown model. Only a *broken* file is fatal (see
    `TestTruncatedCheckpointIsNeverRegistered`). It is also why the hidden-dim check runs before the
    completeness check — judged against the legacy key set, a v1.5 file would be misreported as
    truncated, and would then be hard-rejected on top of it.
    """
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(1024))
        with pytest.raises(NotAMatchError, match="lq_proj hidden dim 1024"):
            PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_legacy_512_checkpoint_is_accepted() -> None:
    """A legacy 512-dim FLUX checkpoint (16 latent channels) still identifies successfully."""
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(512, latent_channels=16))
        config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "pid_decoder"
        assert config.base.value == "flux"


class TestIncompleteLqProjection:
    """A single `lq_proj` key is not evidence of a usable decoder — every LQ weight must be present.

    These rejections are `InvalidMatchError`, not `NotAMatchError`: the file has already identified
    itself as a PiD checkpoint, so it must not fall through to the factory's `Unknown_Config` fallback
    (see `TestTruncatedCheckpointIsNeverRegistered`).
    """

    @pytest.mark.parametrize(
        "dropped",
        [
            "lq_proj.output_heads.3.weight",
            "lq_proj.gate_modules.0.log_alpha",
            "lq_proj.latent_proj.3.block.2.bias",
        ],
    )
    def test_checkpoint_missing_a_required_lq_key_is_rejected(self, dropped: str) -> None:
        sd = _pid_state_dict(512)
        del sd[f"{_NET_PREFIX}{dropped}"]
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 1 of the LQ projection weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_marker_key_alone_is_not_enough(self) -> None:
        """The old behaviour: one `lq_proj.*` key identified a decoder, and the rest were tolerated."""
        sd = {f"{_NET_PREFIX}{_LATENT_PROJ_KEY}": _FakeShapeTensor(512, 16, 3, 3)}
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="LQ projection weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    def test_truncation_is_reported_as_truncation_even_without_the_diagnostic_weight(self) -> None:
        """The backbone is read from `lq_proj.latent_proj.0.weight`, so a file truncated past *that*
        weight used to fail with "cannot determine backbone" — accurate, but not the reason, and not
        the message the install flow promises for a truncated checkpoint. Completeness is therefore
        checked before the backbone, against the single backbone-independent key contract."""
        sd = {f"{_NET_PREFIX}lq_proj.latent_proj.1.weight": _FakeShapeTensor(1)}
        fields = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"}
        with TemporaryDirectory() as tmpdir:
            # No directory name and no base override: nothing but the weights identifies this file.
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="missing 71 of the LQ projection weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(fields))

    def test_distill_only_submodules_do_not_count_as_lq_weights(self) -> None:
        """`net_ema.*` shadows PidNet's own parameter names. The loader drops those submodules, so
        identification has to as well — otherwise a checkpoint carrying only the EMA copy would look
        complete here and then fail in `load_pid_decoder`."""
        sd = {f"net_ema.{k}": _FakeShapeTensor(1) for k in required_lq_proj_keys()}
        sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(512, 16, 3, 3)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="LQ projection weights"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


class TestTruncatedCheckpointIsNeverRegistered:
    """Rejecting a truncated checkpoint in the config class is only half the job.

    Every config class signals "not mine" with `NotAMatchError`, which the factory collects and then,
    with `allow_unknown_models` (default: true), papers over by returning `Unknown_Config`. A file that
    identified itself as a PiD decoder and was then found to be incomplete would therefore still be
    installed — as an unknown model, with a database record, failing only when something tried to load
    it. `InvalidMatchError` is what makes the rejection stick.
    """

    def _write_partial_pid_checkpoint(self, root: Path) -> Path:
        """A file carrying a single `lq_proj.*` weight: enough to be recognised, far from loadable.

        Written as a `.pth`, the format NVIDIA actually ships, so this goes through the same
        pickle-scan-and-`torch.load` path a real truncated download would.
        """
        import torch

        path = root / "model_ema_bf16.pth"
        torch.save({f"{_NET_PREFIX}lq_proj.latent_proj.1.weight": torch.zeros(1)}, path)
        return path

    def test_factory_returns_no_config_even_with_allow_unknown(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = self._write_partial_pid_checkpoint(Path(tmpdir))
            result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

        assert result.config is None, "a recognised-but-broken checkpoint must not be registered"
        assert not any(isinstance(r, Unknown_Config) for r in result.details.values())

    def test_the_rejection_reason_survives_for_the_installer_to_report(self) -> None:
        """`_probe` raises with this text, so the user is told the file is incomplete rather than the
        misleading "could not identify model"."""
        with TemporaryDirectory() as tmpdir:
            path = self._write_partial_pid_checkpoint(Path(tmpdir))
            result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

        assert result.invalid_matches
        assert "LQ projection weights" in str(result.invalid_matches[0])


class TestBackboneFromInstallSource:
    """FLUX.1, SD3 and Qwen-Image PiD decoders are architecturally identical (16 latent channels), so the
    backbone can only come from the name. A direct single-file install has none — but its source does."""

    _SD3_SOURCE = "nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth"

    def test_source_identifies_sd3_without_a_base_override(self) -> None:
        """Without this, a directly installed SD3 decoder is recorded as FLUX and then rejected by the
        SD3 decode node."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {"source": self._SD3_SOURCE}
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(mod, dict(overrides))
            assert config.base is BaseModelType.StableDiffusion3

    def test_flux_config_rejects_a_checkpoint_the_source_names_as_sd3(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {"source": self._SD3_SOURCE}
            with pytest.raises(NotAMatchError, match="name indicates"):
                PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(overrides))

    def test_qwen_image_source_is_recognised(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            overrides = {k: v for k, v in _OVERRIDE_FIELDS.items() if k != "base"} | {
                "source": "nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth"
            }
            config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(overrides))
            assert config.base is BaseModelType.QwenImage

    def test_base_override_still_wins_when_nothing_names_the_backbone(self) -> None:
        """The starter installer's explicit base remains the fallback for a fully anonymous file."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            config = PiDDecoder_Checkpoint_SD3_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, base="sd-3", source="local file")
            )
            assert config.base is BaseModelType.StableDiffusion3


class TestVariantIdentification:
    """The variant is read from NVIDIA's directory name where there is one, and falls back to the
    backbone's only published preset otherwise (SDXL and Qwen-Image ship 2K-to-4K only)."""

    def test_directory_name_wins(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(
                Path(tmpdir), _pid_state_dict(512), dir_name="PiD_res2kto4k_sr4x_official_flux_distill_4step"
            )
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_install_source_is_used_when_the_stored_name_is_silent(self) -> None:
        """A direct single-file install lands in a UUID directory, but the HF source still names the preset."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(
                mod,
                dict(
                    _OVERRIDE_FIELDS,
                    source="nvidia/PiD::checkpoints_deprecated/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
                ),
            )
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_flux_defaults_to_2k_when_the_name_is_silent(self) -> None:
        """FLUX.1 ships both presets, so a nameless single-file install keeps the 2K default."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512))
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
            assert config.variant is PiDDecoderVariantType.Res2k_Sr4x

    def test_sdxl_single_file_install_gets_the_only_published_preset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512, latent_channels=4))
            config = PiDDecoder_Checkpoint_SDXL_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS, base="sdxl"))
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_qwen_image_single_file_install_gets_the_only_published_preset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), _pid_state_dict(512, latent_channels=16))
            config = PiDDecoder_Checkpoint_QwenImage_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, base="qwen-image")
            )
            assert config.variant is PiDDecoderVariantType.Res2kTo4k_Sr4x

    def test_explicit_variant_override_wins(self) -> None:
        """A starter-model install passes the variant it knows it is downloading."""
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(
                Path(tmpdir), _pid_state_dict(512), dir_name="PiD_res2kto4k_sr4x_official_flux_distill_4step"
            )
            config = PiDDecoder_Checkpoint_FLUX_Config.from_model_on_disk(
                mod, dict(_OVERRIDE_FIELDS, variant=PiDDecoderVariantType.Res2k_Sr4x)
            )
            assert config.variant is PiDDecoderVariantType.Res2k_Sr4x
