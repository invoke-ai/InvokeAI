"""Tests for the GGUF Wan probe (Main_GGUF_Wan_Config)."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_GGUF_Wan_Config,
    _detect_wan_gguf_expert,
    _detect_wan_gguf_variant,
    _has_wan_keys,
    _is_native_wan_layout,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _ggml(shape: tuple[int, ...]) -> GGMLTensor:
    return GGMLTensor(
        data=torch.zeros((1,), dtype=torch.uint8),
        ggml_quantization_type=gguf.GGMLQuantizationType.Q4_0,
        tensor_shape=torch.Size(shape),
        compute_dtype=torch.float32,
    )


def _wan_a14b_state_dict(prefix: str = "") -> dict:
    """Synthetic Wan A14B GGUF state dict (16-channel patch embed)."""
    return {
        f"{prefix}patch_embedding.weight": _ggml((5120, 16, 1, 2, 2)),
        f"{prefix}condition_embedder.text_embedder.linear_1.weight": _ggml((5120, 4096)),
        f"{prefix}blocks.0.attn1.to_q.weight": _ggml((5120, 5120)),
        f"{prefix}blocks.0.ffn.net.0.proj.weight": _ggml((13824, 5120)),
    }


def _wan_ti2v_state_dict() -> dict:
    """Synthetic Wan TI2V-5B GGUF state dict (48-channel patch embed)."""
    return {
        "patch_embedding.weight": _ggml((3072, 48, 1, 2, 2)),
        "condition_embedder.text_embedder.linear_1.weight": _ggml((3072, 4096)),
        "blocks.0.attn1.to_q.weight": _ggml((3072, 3072)),
        "blocks.0.ffn.net.0.proj.weight": _ggml((14336, 3072)),
    }


def _wan_i2v_a14b_state_dict() -> dict:
    """Wan 2.2 I2V-A14B GGUF: same shape as T2V except patch_embedding has 36
    input channels (16 noise + 16 ref-image latents + 4 first-frame mask)."""
    return {
        "patch_embedding.weight": _ggml((5120, 36, 1, 2, 2)),
        "condition_embedder.text_embedder.linear_1.weight": _ggml((5120, 4096)),
        "blocks.0.attn1.to_q.weight": _ggml((5120, 5120)),
        "blocks.0.ffn.net.0.proj.weight": _ggml((13824, 5120)),
    }


def _wan_a14b_native_state_dict() -> dict:
    """Synthetic Wan A14B GGUF state dict using the native upstream key layout
    (text_embedding/self_attn/cross_attn/ffn.0 — what QuantStack and ComfyUI ship)."""
    return {
        "patch_embedding.weight": _ggml((5120, 16, 1, 2, 2)),
        "text_embedding.0.weight": _ggml((5120, 4096)),
        "text_embedding.2.weight": _ggml((5120, 5120)),
        "blocks.0.self_attn.q.weight": _ggml((5120, 5120)),
        "blocks.0.cross_attn.q.weight": _ggml((5120, 5120)),
        "blocks.0.ffn.0.weight": _ggml((13824, 5120)),
        "blocks.0.modulation": _ggml((1, 6, 5120)),
        "head.head.weight": _ggml((64, 5120)),
        "head.modulation": _ggml((1, 2, 5120)),
    }


def _build_overrides(model_path: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(model_path),
        "file_size": 0,
        "name": name,
        "source": str(model_path),
        "source_type": "path",
    }


def _make_mod(path: Path, sd: dict, metadata: dict[str, str] | None = None) -> MagicMock:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    mod.metadata.return_value = metadata or {}
    return mod


def test_model_on_disk_reads_gguf_name_metadata(tmp_path: Path) -> None:
    path = tmp_path / "renamed.gguf"
    writer = gguf.GGUFWriter(path, "wan")
    writer.add_name("Wan2.2 T2V A14B")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    assert ModelOnDisk(path).metadata()["general.name"] == "Wan2.2 T2V A14B"


class TestKeyFingerprint:
    def test_recognises_bare_keys(self):
        assert _has_wan_keys(_wan_ti2v_state_dict()) is True

    def test_recognises_comfyui_prefix(self):
        assert _has_wan_keys(_wan_a14b_state_dict(prefix="model.diffusion_model.")) is True

    def test_recognises_diffusion_model_prefix(self):
        assert _has_wan_keys(_wan_a14b_state_dict(prefix="diffusion_model.")) is True

    def test_recognises_native_upstream_layout(self):
        assert _has_wan_keys(_wan_a14b_native_state_dict()) is True

    def test_rejects_qwen_image(self):
        sd = {"txt_in.weight": _ggml((1, 1)), "img_in.weight": _ggml((1, 1))}
        assert _has_wan_keys(sd) is False

    def test_rejects_flux(self):
        sd = {"double_blocks.0.img_attn.proj.weight": _ggml((1, 1))}
        assert _has_wan_keys(sd) is False


class TestNativeLayoutDetection:
    def test_native_a14b(self):
        assert _is_native_wan_layout(_wan_a14b_native_state_dict()) is True

    def test_diffusers_a14b_is_not_native(self):
        assert _is_native_wan_layout(_wan_a14b_state_dict()) is False

    def test_diffusers_ti2v_is_not_native(self):
        assert _is_native_wan_layout(_wan_ti2v_state_dict()) is False


class TestVariantDetection:
    def test_a14b_from_16ch(self):
        sd = _wan_a14b_state_dict()
        assert _detect_wan_gguf_variant(sd) == WanVariantType.T2V_A14B

    def test_ti2v_from_48ch(self):
        sd = _wan_ti2v_state_dict()
        assert _detect_wan_gguf_variant(sd) == WanVariantType.TI2V_5B

    def test_i2v_a14b_from_36ch(self):
        """Wan 2.2 I2V has the same A14B architecture as T2V but with
        in_channels=36 because the ref-image latents and first-frame mask are
        concatenated to the noise along the channel dim before patch embedding."""
        sd = _wan_i2v_a14b_state_dict()
        assert _detect_wan_gguf_variant(sd) == WanVariantType.I2V_A14B

    def test_unknown_channel_count_returns_none(self):
        sd = {"patch_embedding.weight": _ggml((1, 32, 1, 2, 2))}
        assert _detect_wan_gguf_variant(sd) is None

    def test_missing_patch_embedding_returns_none(self):
        sd = {"blocks.0.attn1.to_q.weight": _ggml((1, 1))}
        assert _detect_wan_gguf_variant(sd) is None


class TestExpertFilenameHeuristic:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("wan2.2-t2v-a14b-high_noise-Q4_K_M", "high"),
            ("Wan2.2-T2V-A14B-High-Noise-Q4_K_M", "high"),
            ("wan_a14b_highnoise_q4", "high"),
            ("wan2.2-t2v-a14b-low_noise-Q4_K_M", "low"),
            ("Wan2.2-A14B-LowNoise-Q4", "low"),
            ("wan2.2-ti2v-5b-Q4_K_M", "none"),
            ("wan-A14B-flagship", "none"),
        ],
    )
    def test_filename_heuristic(self, name: str, expected: str):
        assert _detect_wan_gguf_expert(name) == expected


class TestProbe:
    @pytest.mark.parametrize(
        "filename,state_dict",
        [
            ("Wan2.1-T2V-14B-Q4_K_M.gguf", _wan_a14b_state_dict()),
            ("Wan2.1-I2V-14B-Q4_K_M.gguf", _wan_i2v_a14b_state_dict()),
        ],
    )
    def test_rejects_wan_2_1(self, filename: str, state_dict: dict) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            path.touch()

            with pytest.raises(NotAMatchError, match="Wan 2.1"):
                Main_GGUF_Wan_Config.from_model_on_disk(
                    _make_mod(path, state_dict),
                    _build_overrides(path, "unsupported Wan 2.1"),
                )

    def test_rejects_ambiguous_a14b_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "renamed-high_noise-Q4_K_M.gguf"
            path.touch()

            with pytest.raises(NotAMatchError, match="Wan 2.2"):
                Main_GGUF_Wan_Config.from_model_on_disk(
                    _make_mod(path, _wan_a14b_state_dict()),
                    _build_overrides(path, "ambiguous Wan A14B"),
                )

    def test_accepts_renamed_wan_2_2_from_gguf_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "renamed-high_noise-Q4_K_M.gguf"
            path.touch()

            config = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(path, _wan_a14b_state_dict(), {"general.name": "Wan2.2 T2V A14B"}),
                _build_overrides(path, "renamed Wan 2.2"),
            )

            assert config.variant == WanVariantType.T2V_A14B

    def test_rejects_wan_2_1_metadata_despite_wan_2_2_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "Wan2.2-T2V-A14B-high_noise-Q4_K_M.gguf"
            path.touch()

            with pytest.raises(NotAMatchError, match="Wan 2.1"):
                Main_GGUF_Wan_Config.from_model_on_disk(
                    _make_mod(path, _wan_a14b_state_dict(), {"general.name": "Wan2.1 T2V 14B"}),
                    _build_overrides(path, "misnamed Wan 2.1"),
                )

    def test_a14b_high_noise_filename(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan2.2-t2v-a14b-high_noise-Q4_K_M.gguf"
            f.touch()

            cfg = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(f, _wan_a14b_state_dict()),
                _build_overrides(f, "Wan A14B (high)"),
            )
            assert cfg.base == BaseModelType.Wan
            assert cfg.format == ModelFormat.GGUFQuantized
            assert cfg.variant == WanVariantType.T2V_A14B
            assert cfg.expert == "high"

    def test_a14b_low_noise_filename(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan2.2-t2v-a14b-low_noise-Q4_K_M.gguf"
            f.touch()

            cfg = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(f, _wan_a14b_state_dict()),
                _build_overrides(f, "Wan A14B (low)"),
            )
            assert cfg.expert == "low"

    def test_ti2v_5b_unambiguous(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan2.2-ti2v-5b-Q4_K_M.gguf"
            f.touch()

            cfg = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(f, _wan_ti2v_state_dict()),
                _build_overrides(f, "Wan TI2V-5B"),
            )
            assert cfg.variant == WanVariantType.TI2V_5B
            assert cfg.expert == "none"

    def test_rejects_non_gguf(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan-a14b.safetensors"
            f.touch()
            sd = {"patch_embedding.weight": torch.zeros(5120, 16, 1, 2, 2)}  # NOT a GGMLTensor

            with pytest.raises(NotAMatchError, match="GGUF"):
                Main_GGUF_Wan_Config.from_model_on_disk(
                    _make_mod(f, sd),
                    _build_overrides(f, "non-gguf"),
                )

    def test_rejects_unrecognised_state_dict(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "junk.gguf"
            f.touch()
            sd = {"random.key": _ggml((1, 1))}

            with pytest.raises(NotAMatchError, match="Wan transformer"):
                Main_GGUF_Wan_Config.from_model_on_disk(
                    _make_mod(f, sd),
                    _build_overrides(f, "junk"),
                )

    def test_native_upstream_a14b_high_noise(self):
        """QuantStack-style GGUF: native upstream keys + HighNoise filename."""
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
            f.touch()

            cfg = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(f, _wan_a14b_native_state_dict()),
                _build_overrides(f, "Wan A14B QuantStack (high)"),
            )
            assert cfg.base == BaseModelType.Wan
            assert cfg.format == ModelFormat.GGUFQuantized
            assert cfg.variant == WanVariantType.T2V_A14B
            assert cfg.expert == "high"

    def test_explicit_expert_override(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "wan2.2-a14b-flagship.gguf"
            f.touch()
            overrides = _build_overrides(f, "user-tagged")
            overrides["expert"] = "low"

            cfg = Main_GGUF_Wan_Config.from_model_on_disk(
                _make_mod(f, _wan_a14b_state_dict()),
                overrides,
            )
            assert cfg.expert == "low"
