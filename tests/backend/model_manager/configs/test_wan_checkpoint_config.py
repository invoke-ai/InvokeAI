"""Tests for the single-file Wan 2.2 checkpoint probe (Main_Checkpoint_Wan_Config).

Regression coverage for #9463: community Wan 2.2 fine-tunes ship as one
``.safetensors`` per transformer, which InvokeAI previously refused to identify
at all ("unidentified model") because only Diffusers folders and GGUF files had
a matching config class.
"""

from pathlib import Path
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Wan_Config,
    _detect_wan_expert,
    _find_wan_2_1_marker,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

A14B_DIM = 5120
TI2V_DIM = 3072


def _t(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


def _native_sd(in_channels: int = 16, dim: int = A14B_DIM, prefix: str = "") -> dict:
    """Native upstream / ComfyUI Wan key layout — what CivitAI fine-tunes ship."""
    sd = {
        f"{prefix}patch_embedding.weight": _t(dim, in_channels, 1, 2, 2),
        f"{prefix}text_embedding.0.weight": _t(dim, 4096),
        f"{prefix}text_embedding.2.weight": _t(dim, dim),
        f"{prefix}time_embedding.0.weight": _t(dim, 256),
        f"{prefix}head.head.weight": _t(64, dim),
        f"{prefix}head.modulation": _t(1, 2, dim),
        f"{prefix}blocks.0.self_attn.q.weight": _t(dim, dim),
        f"{prefix}blocks.0.cross_attn.q.weight": _t(dim, dim),
        f"{prefix}blocks.0.ffn.0.weight": _t(13824, dim),
    }
    return sd


def _diffusers_sd(in_channels: int = 48, dim: int = TI2V_DIM) -> dict:
    """Diffusers Wan key layout, as shipped by Wan-AI/*-Diffusers single files."""
    return {
        "patch_embedding.weight": _t(dim, in_channels, 1, 2, 2),
        "condition_embedder.text_embedder.linear_1.weight": _t(dim, 4096),
        "blocks.0.attn1.to_q.weight": _t(dim, dim),
        "blocks.0.ffn.net.0.proj.weight": _t(14336, dim),
        "proj_out.weight": _t(in_channels * 4, dim),
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


def _make_mod(path: Path, sd: dict) -> MagicMock:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    mod.metadata.return_value = {}
    return mod


def _probe(tmp_path: Path, filename: str, sd: dict) -> Main_Checkpoint_Wan_Config:
    path = tmp_path / filename
    path.touch()
    return Main_Checkpoint_Wan_Config.from_model_on_disk(_make_mod(path, sd), _build_overrides(path, path.stem))


class TestAccepts:
    def test_native_layout_t2v_a14b(self, tmp_path: Path) -> None:
        config = _probe(tmp_path, "Wan2.2-T2V-A14B-high_noise.safetensors", _native_sd(16))
        assert config.base == BaseModelType.Wan
        assert config.format == ModelFormat.Checkpoint
        assert config.variant == WanVariantType.T2V_A14B
        assert config.expert == "high"

    def test_native_layout_i2v_a14b(self, tmp_path: Path) -> None:
        config = _probe(tmp_path, "Wan2.2-I2V-A14B-LowNoise.safetensors", _native_sd(36))
        assert config.variant == WanVariantType.I2V_A14B
        assert config.expert == "low"

    def test_diffusers_layout_ti2v_5b(self, tmp_path: Path) -> None:
        config = _probe(tmp_path, "wanDamme-RapidWan2.2-5B-ti2v-4step.safetensors", _diffusers_sd(48))
        assert config.variant == WanVariantType.TI2V_5B
        assert config.expert == "none"

    @pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
    def test_comfyui_key_prefix(self, tmp_path: Path, prefix: str) -> None:
        config = _probe(tmp_path, "wan22_t2v_high_noise.safetensors", _native_sd(16, prefix=prefix))
        assert config.variant == WanVariantType.T2V_A14B
        assert config.expert == "high"

    def test_comfyui_fp8_scaled(self, tmp_path: Path) -> None:
        """fp8_scaled files carry extra scale tensors; they must not confuse the probe."""
        sd = _native_sd(16)
        sd["patch_embedding.weight"] = _t(A14B_DIM, 16, 1, 2, 2, dtype=torch.float8_e4m3fn)
        sd["blocks.0.self_attn.q.weight"] = _t(A14B_DIM, A14B_DIM, dtype=torch.float8_e4m3fn)
        sd["blocks.0.self_attn.q.scale_weight"] = _t(1, dtype=torch.float32)
        sd["scaled_fp8"] = _t(1, dtype=torch.float8_e4m3fn)

        config = _probe(tmp_path, "Wan2.2-T2V-A14B-HighNoise-fp8_scaled.safetensors", sd)
        assert config.variant == WanVariantType.T2V_A14B
        assert config.expert == "high"

    def test_untagged_community_filename(self, tmp_path: Path) -> None:
        """The #9463 case: a fine-tune whose filename never says "Wan 2.2".

        Unlike the GGUF probe, the checkpoint probe must not demand a version
        marker in the name — rejecting these was the reported bug.
        """
        config = _probe(tmp_path, "SmoothMix_HighNoise.safetensors", _native_sd(16))
        assert config.variant == WanVariantType.T2V_A14B
        assert config.expert == "high"


def _ggml(*shape: int) -> GGMLTensor:
    return GGMLTensor(
        data=torch.zeros((1,), dtype=torch.uint8),
        ggml_quantization_type=gguf.GGMLQuantizationType.Q4_0,
        tensor_shape=torch.Size(shape),
        compute_dtype=torch.float32,
    )


class TestRejects:
    def test_gguf_file(self, tmp_path: Path) -> None:
        """A .gguf goes to Main_GGUF_Wan_Config; the suffix guard turns it away first."""
        with pytest.raises(NotAMatchError, match="safetensors"):
            _probe(tmp_path, "wan2.2-t2v-a14b-high_noise-Q4_K_M.gguf", _native_sd(16))

    def test_gguf_tensors(self, tmp_path: Path) -> None:
        """Belt-and-braces behind the suffix guard: GGML tensors are never a match
        for this config regardless of what the file is called."""
        sd = {
            "patch_embedding.weight": _ggml(A14B_DIM, 16, 1, 2, 2),
            "text_embedding.0.weight": _ggml(A14B_DIM, 4096),
            "blocks.0.self_attn.q.weight": _ggml(A14B_DIM, A14B_DIM),
        }
        with pytest.raises(NotAMatchError, match="GGUF"):
            _probe(tmp_path, "wan2.2-t2v-a14b-high_noise.safetensors", sd)

    def test_wan_2_1_i2v_via_clip_image_embedder(self, tmp_path: Path) -> None:
        sd = _native_sd(36)
        sd["img_emb.proj.0.weight"] = _t(1280, 1280)
        with pytest.raises(NotAMatchError, match="Wan 2.1"):
            _probe(tmp_path, "some-i2v-model.safetensors", sd)

    def test_wan_2_1_1_3b_via_inner_dim(self, tmp_path: Path) -> None:
        with pytest.raises(NotAMatchError, match="Wan 2.1"):
            _probe(tmp_path, "renamed-t2v-model.safetensors", _native_sd(16, dim=1536))

    def test_animate(self, tmp_path: Path) -> None:
        """Wan Animate is 36-channel with undecorated block weights and no VACE
        blocks, so nothing else turns it away. Its face-adapter and motion-encoder
        branches have no counterpart in WanTransformer3DModel — real
        wan2.2_animate_14B_bf16.safetensors carries 127 such keys out of 1441 — and
        strict=False would drop every one of them silently.

        It also carries img_emb, so this must be reported as Animate rather than as
        a Wan 2.1 I2V model.
        """
        sd = _native_sd(36)
        sd["face_adapter.fuser_blocks.0.k_norm.weight"] = _t(A14B_DIM)
        sd["motion_encoder.dec.direction.weight"] = _t(512, 512)
        sd["img_emb.proj.0.bias"] = _t(1280)
        with pytest.raises(NotAMatchError, match="Animate"):
            _probe(tmp_path, "wan2.2_animate_14B_bf16.safetensors", sd)

    def test_s2v(self, tmp_path: Path) -> None:
        """Wan S2V is 16-channel and otherwise key-identical to plain T2V-A14B, so
        only its audio branches distinguish it. Prefixes taken from the real
        wan2.2_s2v_14B_bf16.safetensors header (165 such keys of 1260)."""
        sd = _native_sd(16)
        sd["audio_injector.injector.0.k.weight"] = _t(A14B_DIM, A14B_DIM)
        sd["casual_audio_encoder.encoder.final_proj.weight"] = _t(A14B_DIM, 1024)
        sd["cond_encoder.weight"] = _t(A14B_DIM, 16, 1, 2, 2)
        sd["frame_packer.proj.weight"] = _t(A14B_DIM, 16)
        with pytest.raises(NotAMatchError, match="S2V"):
            _probe(tmp_path, "wan2.2_s2v_14B_bf16.safetensors", sd)

    def test_fun_control_camera(self, tmp_path: Path) -> None:
        """The most dangerous of the unsupported variants: 36 channels, key-identical
        to plain I2V-A14B apart from control_adapter, AND shipped as a properly
        tagged high/low pair — so the expert-pairing check would pass and it would
        render as an ordinary I2V, silently ignoring every camera input."""
        sd = _native_sd(36)
        sd["control_adapter.conv.weight"] = _t(A14B_DIM, 36, 1, 2, 2)
        sd["control_adapter.residual_blocks.0.conv1.weight"] = _t(A14B_DIM, A14B_DIM)
        with pytest.raises(NotAMatchError, match="Fun-Control"):
            _probe(tmp_path, "wan2.2_fun_camera_high_noise_14B_bf16.safetensors", sd)

    def test_vace(self, tmp_path: Path) -> None:
        """VACE exists for both Wan 2.1 and 2.2; either way this loader has no
        control branch, so it must refuse rather than silently ignore the input."""
        sd = _native_sd(16)
        sd["vace_blocks.0.after_proj.weight"] = _t(A14B_DIM, A14B_DIM)
        with pytest.raises(NotAMatchError, match="VACE"):
            _probe(tmp_path, "vace-model.safetensors", sd)

    def test_wan_2_1_filename(self, tmp_path: Path) -> None:
        with pytest.raises(NotAMatchError, match="Wan 2.1"):
            _probe(tmp_path, "Wan2.1-T2V-14B.safetensors", _native_sd(16))

    def test_unrecognised_state_dict(self, tmp_path: Path) -> None:
        with pytest.raises(NotAMatchError, match="Wan transformer"):
            _probe(tmp_path, "junk.safetensors", {"random.key": _t(4, 4)})

    def test_lora_carrying_a_full_patch_embedding(self, tmp_path: Path) -> None:
        """Wan I2V adapters bundle a replacement patch_embedding (in_channels 16->36)
        plus the text projection, which is everything ``_has_wan_keys`` looks for. The
        main-model probe must not claim them — Main outranks LoRA in
        ``matches_sort_key``, so it would pull the file out of every LoRA picker."""
        sd = {
            "diffusion_model.patch_embedding.weight": _t(A14B_DIM, 36, 1, 2, 2),
            "diffusion_model.text_embedding.0.weight": _t(A14B_DIM, 4096),
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": _t(32, A14B_DIM),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": _t(A14B_DIM, 32),
        }
        with pytest.raises(NotAMatchError, match="LoRA"):
            _probe(tmp_path, "Wan2.2-I2V-A14B-adapter-low_noise.safetensors", sd)

    @pytest.mark.parametrize("suffix", [".ckpt", ".pt", ".pth", ".bin"])
    def test_non_safetensors_containers(self, tmp_path: Path, suffix: str) -> None:
        """The loader reads these with safetensors.load_file, so claiming one would
        install cleanly and then die with an opaque header error at generation time."""
        with pytest.raises(NotAMatchError, match="safetensors"):
            _probe(tmp_path, f"Wan2.2-T2V-A14B-HighNoise{suffix}", _native_sd(16))

    def test_unknown_channel_count(self, tmp_path: Path) -> None:
        with pytest.raises(NotAMatchError, match="variant"):
            _probe(tmp_path, "weird-wan22.safetensors", _native_sd(24))


class TestWan21Marker:
    def test_clean_wan_2_2_state_dicts_have_no_marker(self) -> None:
        assert _find_wan_2_1_marker(_native_sd(16)) is None
        assert _find_wan_2_1_marker(_native_sd(36)) is None
        assert _find_wan_2_1_marker(_diffusers_sd(48)) is None

    def test_diffusers_layout_image_embedder_is_detected(self) -> None:
        sd = _diffusers_sd(36, dim=A14B_DIM)
        sd["condition_embedder.image_embedder.norm1.weight"] = _t(1280)
        assert _find_wan_2_1_marker(sd) is not None

    def test_marker_survives_comfyui_prefix(self) -> None:
        sd = _native_sd(36, prefix="model.diffusion_model.")
        sd["model.diffusion_model.img_emb.proj.0.weight"] = _t(1280, 1280)
        assert _find_wan_2_1_marker(sd) is not None


class TestEndToEndIdentification:
    """Drive the real install path — a file on disk through ModelConfigFactory.

    The class-level tests above call ``from_model_on_disk`` directly, so they'd
    still pass if the config were left out of the ``AnyModelConfig`` union. These
    reproduce #9463 as reported: the model imports as "unidentified" unless the
    config is actually wired into the factory, and the resulting record must
    dispatch to a registered loader.
    """

    @staticmethod
    def _write(tmp_path: Path, filename: str, sd: dict) -> Path:
        from safetensors.torch import save_file

        path = tmp_path / filename
        save_file(sd, path)
        return path

    @pytest.mark.parametrize(
        "filename, state_dict, expected_variant, expected_expert",
        [
            ("Wan2.2-A14B-SmoothMix-T2V-HighNoise.safetensors", _native_sd(16), WanVariantType.T2V_A14B, "high"),
            ("Wan2.2-A14B-SmoothMix-I2V-LowNoise.safetensors", _native_sd(36), WanVariantType.I2V_A14B, "low"),
            ("wanDamme-RapidWan2.2-5B-ti2v-4step.safetensors", _diffusers_sd(48), WanVariantType.TI2V_5B, "none"),
        ],
    )
    def test_community_models_from_the_issue_are_identified(
        self,
        tmp_path: Path,
        filename: str,
        state_dict: dict,
        expected_variant: WanVariantType,
        expected_expert: str,
    ) -> None:
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory

        path = self._write(tmp_path, filename, state_dict)
        config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True).config

        assert config is not None
        assert isinstance(config, Main_Checkpoint_Wan_Config)
        assert config.variant == expected_variant
        assert config.expert == expected_expert

    def test_record_survives_a_serialization_round_trip(self, tmp_path: Path) -> None:
        """Identification alone isn't enough — the config also has to be a member of
        the ``AnyModelConfig`` union, or the record can't be read back out of the
        model-records DB or served over the API."""
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory

        path = self._write(tmp_path, "Wan2.2-A14B-SmoothMix-T2V-HighNoise.safetensors", _native_sd(16))
        config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True).config
        assert config is not None

        restored = ModelConfigFactory.from_json(config.model_dump_json())
        assert isinstance(restored, Main_Checkpoint_Wan_Config)
        assert restored.variant == WanVariantType.T2V_A14B
        assert restored.expert == "high"

    def test_identified_record_has_a_registered_loader(self, tmp_path: Path) -> None:
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
        from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
        from invokeai.backend.model_manager.load.model_loaders.wan import WanCheckpointModel
        from invokeai.backend.model_manager.taxonomy import SubModelType

        path = self._write(tmp_path, "Wan2.2-A14B-SmoothMix-T2V-HighNoise.safetensors", _native_sd(16))
        config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True).config
        assert config is not None

        implementation, _, _ = ModelLoaderRegistry.get_implementation(config, SubModelType.Transformer)
        assert implementation is WanCheckpointModel


class TestExpertFilenameHeuristic:
    """The A14B expert is not recoverable from the weights and single-file releases
    carry no metadata declaring it, so the filename is the only signal there is.

    Two conventions are both common in the wild and both have to work: an explicit
    ``high_noise``/``low_noise``, and a bare ``HIGH``/``LOW`` token (the whole Kijai
    fp8 catalogue). Scored against 108 real filenames pulled from the Kijai,
    Comfy-Org and QuantStack repos, this table's rules give 0 missed and 0
    mislabelled.
    """

    @pytest.mark.parametrize(
        "name, expected",
        [
            # --- explicit noise markers, all spellings ---
            ("wan2.2-t2v-a14b-high_noise-Q4_K_M", "high"),
            ("Wan2.2-T2V-A14B-High-Noise-Q4_K_M", "high"),
            ("wan_a14b_highnoise_q4", "high"),
            ("wan2.2-t2v-a14b-low_noise-Q4_K_M", "low"),
            ("Wan2.2-A14B-LowNoise-Q4", "low"),
            ("Wan2.2 A14B high noise", "high"),
            ("wan22.low.noise.v3", "low"),
            ("wan22_noise_high_expert", "high"),
            ("low_noise14B", "low"),
            # --- bare HIGH/LOW: real releases that MUST be detected (#9463 follow-up).
            # Regressing these leaves most single-file A14B models unpairable.
            ("Wan2_2-T2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ", "high"),
            ("Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ", "low"),
            ("Wan2_2-Fun-InP-A14B-HIGH_fp8_e4m3fn_scaled_KJ", "high"),
            ("Wan2_2-T2V-A14B-LOW-HoloCine-full_fp8_e4m3fn_scaled_KJ", "low"),
            ("SmoothMix_I2V_v2_High-Q4_K_M", "high"),
            ("smoothMixWan22I2VT2V_t2vHigh-Q6_K", "high"),
            # A marker sitting next to an unrelated descriptor is still a marker.
            ("Wan2.2_NSFW_i2v_14b_high_lighting_fp16_v2.1", "high"),
            ("wan2.2_T2V_fp8_LOW_lightning_edition", "low"),
            ("wan22EnhancedNSFWSVICamera_nsfwFASTMOVEFP8Low", "low"),
            # A marker fused to an adjacent run of characters, not just to "noise".
            ("WAN2.2t2vLOWNOISEFP8", "low"),
            # --- bare high/low used as an adjective about something else ---
            ("Wan2.2-A14B-T2V-lowCFG-merge", "none"),
            ("wan2.2-a14b-t2v-4step-low-cfg-merge", "none"),
            ("Wan22_A14B_T2V_low_step_v2", "none"),
            ("Wan2.2_A14B_highRes_finetune", "none"),
            ("Wan2.2_A14B_lowVRAM", "none"),
            ("Wan2.2-A14B-HighQuality", "none"),
            ("wan-14B_vace_phantom_v2_LowSteps[Causvid]_fp8_e4m3fn", "none"),
            # ...but only the FOLLOWING token disqualifies. "Low Angle" is a camera
            # angle; "Angle HIGH" is the high-noise expert of a camera-angle LoRA.
            ("Extream Low Angle HIGH - Wan2.2 - v1 low-angle shot", "high"),
            ("Wan22_I2V_VBVR_HIGH_rank_64_fp16", "high"),
            ("Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp16_v2.1", "low"),
            # --- a file that serves BOTH experts must not be tagged with one.
            # For a LoRA, 'none' means "apply to both", which is the right answer. ---
            ("Wan2.2-I2V-AnalBlasting-HIGH-LOW", "none"),
            ("Nier yorha HIGH + LOW - Wan2.2 - nier", "none"),
            ("Anal Insertion I2V LOW + HIGH - Wan2.2 - v2", "none"),
            ("wan2.2-i2v-low-to-high-lora", "none"),
            # --- no marker at all ---
            ("wan2.2-ti2v-5b-Q4_K_M", "none"),
            ("wan-A14B-flagship", "none"),
            # --- token matching: a marker must never fire on a substring. The last
            # two were mismatched by the original substring heuristic as well. ---
            ("wan22-slow-motion-a14b", "none"),
            ("wan22-highway-lora-merge", "none"),
            ("wan22-flow-shift-tune", "none"),
            ("wan22-slow-noise-test", "none"),
            ("wan22_flownoise_v1", "none"),
            # --- an explicit noise marker outranks a bare token found earlier ---
            ("wan2.2_t2v_low_high_noise_14B_fp16", "high"),
        ],
    )
    def test_filename_heuristic(self, name: str, expected: str) -> None:
        assert _detect_wan_expert(name) == expected

    @pytest.mark.parametrize(
        "declared, expected",
        [
            ({"model_type": "Wan22-I2V-A14B-low"}, "low"),
            ({"model_type": "Wan2_2-T2V-A14B-HIGH"}, "high"),
            ({"general.name": "Wan2.2 T2V A14B high noise"}, "high"),
            ({"model_type": "Wan2_2-I2V-A14B"}, "none"),
            ({"format": "pt"}, "none"),
        ],
    )
    def test_expert_falls_back_to_declared_metadata(self, tmp_path: Path, declared: dict, expected: str) -> None:
        """Sampled 2026-08-13: every Wan 2.2 safetensors in Kijai/WanVideo_comfy_fp8_scaled
        carries __metadata__["model_type"] naming the expert, while the Comfy-Org
        repackaged files carry no __metadata__ at all — hence filename first, metadata
        as the fallback, and neither alone is sufficient."""
        from safetensors.torch import save_file

        # A name with no marker at all, so only the metadata can settle the expert.
        path = tmp_path / "my-wan-model.safetensors"
        save_file(_native_sd(36), path, metadata={k: str(v) for k, v in declared.items()})

        config = Main_Checkpoint_Wan_Config.from_model_on_disk(ModelOnDisk(path), _build_overrides(path, path.stem))
        assert config.expert == expected

    def test_filename_outranks_metadata(self, tmp_path: Path) -> None:
        """Renaming the file is the only lever a user has to correct a mis-detected
        expert — there is no UI for the field, and the Wan model loader's error tells
        them to use it. An embedded model_type must not override that."""
        from safetensors.torch import save_file

        path = tmp_path / "Wan2.2-A14B-I2V-high_noise.safetensors"
        save_file(_native_sd(36), path, metadata={"model_type": "Wan22-I2V-A14B-low"})

        config = Main_Checkpoint_Wan_Config.from_model_on_disk(ModelOnDisk(path), _build_overrides(path, path.stem))
        assert config.expert == "high"

    @pytest.mark.parametrize("name", ["Wan2.2_TI2V_5B_lowVRAM", "Wan2.2-TI2V-5B-Turbo-lowSteps", "wan22-5b-LOW"])
    def test_ti2v_5b_never_gets_an_expert(self, tmp_path: Path, name: str) -> None:
        """TI2V-5B is single-transformer, so the expert field is meaningless — and a
        stray 'low' would put the model in the Transformer (Low Noise) picker, where
        it can never be a valid A14B partner."""
        config = _probe(tmp_path, f"{name}.safetensors", _diffusers_sd(48))
        assert config.variant == WanVariantType.TI2V_5B
        assert config.expert == "none"
