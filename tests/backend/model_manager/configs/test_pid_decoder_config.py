"""Regression tests for PiD decoder checkpoint identification.

Covers what identification has to get right before a checkpoint reaches the decode:

- The contract: identification holds a file to exactly the key set and shapes `load_pid_decoder`
  enforces. Checking a subset is not a milder version of the same guarantee — loaders run under
  `skip_torch_weight_init()`, so a weight the checkpoint does not supply is uninitialised memory
  rather than a default, and a file accepted here but refused there decodes nothing.
- Architecture: InvokeAI's build_pid_net constructs only the legacy 512-dim PidNet. NVIDIA's v1.5
  decoders use lq_hidden_dim=1024 (plus PiT injection, scalar gates, ...), which cannot be loaded
  into it, so such a checkpoint must be rejected here instead of crashing mid-decode.
- Backbone and variant: read from the weights where the weights can say, and from name evidence only
  for the FLUX.1 / SD3 / Qwen-Image tie the weights cannot break.

Anything rejected for a backbone-independent reason must be rejected *outright* (`InvalidMatchError`)
rather than fall through to the factory's `Unknown_Config` fallback — see
`TestUnusableCheckpointIsNeverRegistered`.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
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


def _pid_state_dict(lq_hidden_dim: int = 512, latent_channels: int = 16) -> dict[str, object]:
    """A complete PiD-looking state dict: every weight PidNet expects, at the shape it expects, with
    the discriminator conv overridden to the given hidden dim / latent channel count."""
    sd: dict[str, object] = {
        f"{_NET_PREFIX}{k}": _FakeShapeTensor(*shape) for k, shape in required_pid_net_shapes().items()
    }
    sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = _FakeShapeTensor(lq_hidden_dim, latent_channels, 3, 3)
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


def test_v1_5_checkpoint_is_rejected_at_identification() -> None:
    """A 1024-dim (v1.5) checkpoint must be rejected, not accepted and crashed on later.

    The architecture check runs before the contract check so the diagnosis is the accurate one: a
    v1.5 file is intact, and judged against the legacy contract it would be reported as a pile of
    missing and unexpected keys rather than as the newer architecture it is.
    """
    with TemporaryDirectory() as tmpdir:
        mod = _mock_mod(Path(tmpdir), _pid_state_dict(lq_hidden_dim=1024))
        with pytest.raises(InvalidMatchError, match="lq_proj hidden dim 1024"):
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

    def test_an_unexpected_key_is_rejected(self) -> None:
        """`load_pid_decoder` refuses these too, so accepting them here would install a file that
        cannot load."""
        sd = _pid_state_dict()
        sd[f"{_NET_PREFIX}not_a_pid_key"] = _FakeShapeTensor(1)
        with TemporaryDirectory() as tmpdir:
            mod = _mock_mod(Path(tmpdir), sd)
            with pytest.raises(InvalidMatchError, match="1 keys PidNet does not expect"):
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


def _write_pid_checkpoint(root: Path, state_dict: dict[str, object]) -> Path:
    """Write a state dict as a real `.pth`, the format NVIDIA ships, so the factory reaches it through
    the same pickle-scan-and-`torch.load` path a real download would."""
    path = root / "model_ema_bf16.pth"
    torch.save(state_dict, path)
    return path


def _real_pid_state_dict(lq_hidden_dim: int = 512, latent_channels: int = 16) -> dict[str, object]:
    """`_pid_state_dict` with tensors that can actually be serialised.

    PidNet is ~5.5 GB in float32, so every weight is a zero-stride view onto one shared scalar: the
    shapes are the real ones, the file is ~50 KB, and `torch.save` deduplicates the storage.
    """
    scalar = torch.zeros(())
    sd: dict[str, object] = {
        f"{_NET_PREFIX}{k}": scalar.expand(shape) for k, shape in required_pid_net_shapes().items()
    }
    sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = scalar.expand(lq_hidden_dim, latent_channels, 3, 3)
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
        """Both wrong at once. The architecture check fires first, so this is the case where an
        unsupported-but-not-fatal verdict would let a *truncated* file through to Unknown_Config."""
        sd = _real_pid_state_dict(lq_hidden_dim=1024)
        del sd[f"{_NET_PREFIX}lq_proj.output_heads.3.weight"]
        return sd

    def _intact_v1_5(self) -> dict[str, object]:
        return _real_pid_state_dict(lq_hidden_dim=1024)

    def _unsupported_latent_channels(self) -> dict[str, object]:
        return _real_pid_state_dict(latent_channels=32)

    def _malformed_discriminator(self) -> dict[str, object]:
        sd = _real_pid_state_dict()
        sd[f"{_NET_PREFIX}{_LATENT_PROJ_KEY}"] = torch.zeros(512)
        return sd

    @pytest.mark.parametrize(
        ("case", "expected_reason"),
        [
            ("_partial", "missing 456 of the weights"),
            ("_missing_backbone_weight", "missing 1 of the weights required by PidNet"),
            ("_truncated_v1_5", "lq_proj hidden dim 1024"),
            ("_intact_v1_5", "lq_proj hidden dim 1024"),
            ("_unsupported_latent_channels", "32 latent channels"),
            ("_malformed_discriminator", "malformed lq_proj.latent_proj.0.weight"),
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

    def test_a_valid_checkpoint_still_identifies(self) -> None:
        """The counterweight: none of the above may make a real decoder harder to install."""
        with TemporaryDirectory() as tmpdir:
            path = _write_pid_checkpoint(Path(tmpdir), _real_pid_state_dict())
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
