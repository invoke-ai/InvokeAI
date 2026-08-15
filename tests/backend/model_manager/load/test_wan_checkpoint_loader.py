"""Tests for the single-file Wan 2.2 checkpoint loader (WanCheckpointModel).

Round-trips a real (tiny) ``WanTransformer3DModel`` through the on-disk formats
the community actually ships, so the shape-driven architecture inference is
checked against diffusers' own module tree rather than against a hand-written
expectation.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load.model_loaders.wan import (
    _WAN_NATIVE_TO_DIFFUSERS_RENAMES,
    WanCheckpointModel,
    _build_wan_transformer_config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType, WanVariantType

# A structurally faithful but tiny Wan transformer. attention_head_dim must stay
# at 128 — the loader derives num_attention_heads as inner_dim // 128, matching
# the whole Wan 2.2 family.
TINY_MODEL_KWARGS = {
    "patch_size": (1, 2, 2),
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 128,
    "num_attention_heads": 1,
    "ffn_dim": 64,
    "text_dim": 32,
}


def _tiny_model():
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel(**TINY_MODEL_KWARGS)


def _to_native_layout(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert the loader's native->diffusers rename table.

    Applied longest-replacement-first so that, e.g., ``condition_embedder.
    text_embedder.linear_1`` is rewritten before the shorter ``scale_shift_table``
    rules can bite into it.
    """
    rules = sorted(_WAN_NATIVE_TO_DIFFUSERS_RENAMES, key=lambda pair: len(pair[1]), reverse=True)
    # norm2/norm3 swap: the forward table routes through a placeholder, so mirror
    # that here rather than trying to reuse the entries directly.
    rules = [(diffusers, native) for native, diffusers in rules if not native.startswith("norm")]

    native_sd: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = key
        for needle, replacement in rules:
            new_key = new_key.replace(needle, replacement)
        new_key = new_key.replace("norm2", "norm__placeholder").replace("norm3", "norm2")
        new_key = new_key.replace("norm__placeholder", "norm3")
        native_sd[new_key] = value
    return native_sd


def _make_loader() -> WanCheckpointModel:
    loader = object.__new__(WanCheckpointModel)
    loader._ram_cache = MagicMock()
    return loader


def _load(path: Path, variant: WanVariantType = WanVariantType.T2V_A14B):
    config = MagicMock()
    config.path = str(path)
    config.variant = variant

    with (
        patch("invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_torch_device"),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.bfloat16,
        ),
    ):
        return _make_loader()._load_from_singlefile(config)


class TestArchitectureInference:
    def test_matches_the_model_it_came_from(self) -> None:
        sd = _tiny_model().state_dict()
        inferred = _build_wan_transformer_config(sd, WanVariantType.T2V_A14B, source="test")
        assert inferred == TINY_MODEL_KWARGS

    def test_reports_the_missing_key(self) -> None:
        sd = _tiny_model().state_dict()
        del sd["proj_out.weight"]
        with pytest.raises(RuntimeError, match="proj_out.weight"):
            _build_wan_transformer_config(sd, WanVariantType.T2V_A14B, source="test")

    def test_counts_layers_from_the_highest_block_index(self) -> None:
        model = _tiny_model()
        sd = model.state_dict()
        assert _build_wan_transformer_config(sd, WanVariantType.T2V_A14B, source="test")["num_layers"] == 2
        # Dropping the last block's keys must drop the inferred layer count too —
        # the count comes from the weights, not from a per-variant lookup table.
        trimmed = {k: v for k, v in sd.items() if not k.startswith("blocks.1.")}
        assert _build_wan_transformer_config(trimmed, WanVariantType.T2V_A14B, source="test")["num_layers"] == 1


class TestEndToEnd:
    def test_diffusers_layout_round_trip(self, tmp_path: Path) -> None:
        reference = _tiny_model()
        path = tmp_path / "wan22-t2v-a14b-high_noise.safetensors"
        save_file(reference.state_dict(), path)

        model = _load(path)

        assert set(model.state_dict().keys()) == set(reference.state_dict().keys())
        assert all(t.dtype == torch.bfloat16 for t in model.state_dict().values() if t.is_floating_point())

    def test_native_layout_round_trip(self, tmp_path: Path) -> None:
        reference = _tiny_model()
        native_sd = _to_native_layout(reference.state_dict())
        # Sanity check that the fixture really is in the native layout.
        assert "text_embedding.0.weight" in native_sd
        assert "condition_embedder.text_embedder.linear_1.weight" not in native_sd

        path = tmp_path / "SmoothMix_HighNoise.safetensors"
        save_file(native_sd, path)

        model = _load(path)

        assert set(model.state_dict().keys()) == set(reference.state_dict().keys())

    def test_comfyui_prefixed_round_trip(self, tmp_path: Path) -> None:
        reference = _tiny_model()
        prefixed = {f"model.diffusion_model.{k}": v for k, v in reference.state_dict().items()}
        path = tmp_path / "wan22_t2v_low_noise.safetensors"
        save_file(prefixed, path)

        model = _load(path)

        assert set(model.state_dict().keys()) == set(reference.state_dict().keys())

    def test_fp8_scaled_is_dequantized_and_scales_are_dropped(self, tmp_path: Path) -> None:
        reference = _tiny_model()
        sd = {k: v.clone() for k, v in reference.state_dict().items()}

        target = "blocks.0.attn1.to_q.weight"
        sd[target] = torch.full_like(sd[target], 0.5).to(torch.float8_e4m3fn)
        sd["blocks.0.attn1.to_q.scale_weight"] = torch.tensor([4.0])
        sd["blocks.0.attn1.to_q.scale_input"] = torch.tensor([1.0])
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)

        path = tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors"
        save_file(sd, path)

        model = _load(path)

        loaded = model.state_dict()
        assert set(loaded.keys()) == set(reference.state_dict().keys())
        # 0.5 * 4.0, materialised at the compute dtype.
        assert loaded[target].dtype == torch.bfloat16
        assert torch.allclose(loaded[target].float(), torch.full_like(loaded[target].float(), 2.0))

    def test_scale_bookkeeping_never_reaches_the_model(self, tmp_path: Path) -> None:
        """``load_state_dict(strict=False)`` silently ignores unexpected keys, so
        assert on what the loader actually hands over rather than on the result."""
        target = "blocks.0.attn1.to_q.weight"
        sd = _tiny_model().state_dict()
        sd[target] = torch.full_like(sd[target], 0.5)
        sd["blocks.0.attn1.to_q.scale_weight"] = torch.tensor([4.0])
        sd["blocks.0.attn1.to_q.scale_input"] = torch.tensor([1.0])
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)

        path = tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors"
        save_file(sd, path)

        model = MagicMock()
        model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        with patch("diffusers.WanTransformer3DModel", return_value=model):
            _load(path)

        handed_over = model.load_state_dict.call_args.args[0]
        assert not [k for k in handed_over if k.endswith((".scale_weight", ".scale_input")) or k == "scaled_fp8"]
        # ...without eating scale_shift_table, which is a real Wan parameter.
        assert "scale_shift_table" in handed_over
        # The scale is applied whatever the weight's dtype — `_dequantize_comfyui_fp8`
        # has no fp8 gate, deliberately, because "scaled" checkpoints are not all fp8.
        # Asserted rather than left implicit: this fixture's weight is bf16, so without
        # this line the test would construct a 4x-scaled weight and say nothing about it.
        assert torch.allclose(handed_over[target].float(), torch.full_like(handed_over[target].float(), 2.0))

    def test_extra_modules_are_refused_not_dropped(self, tmp_path: Path) -> None:
        """`strict=False` silently discards weights the model has nowhere to put.

        Several Wan 2.2 derivatives are supersets of the plain transformer — real
        wan2.2_fun_camera_high_noise_14B_bf16.safetensors adds 6 `control_adapter.*`
        keys, S2V adds 165, Animate adds 127. They build a correctly-shaped model and
        report zero missing keys, so without this check they load clean and then
        generate with their entire conditioning branch absent.

        The probe turns away the families we know by name; this is the generic
        backstop for the ones nobody has enumerated yet.
        """
        sd = _tiny_model().state_dict()
        sd["control_adapter.conv.weight"] = torch.zeros(128, 16, 1, 2, 2)
        sd["control_adapter.residual_blocks.0.conv1.weight"] = torch.zeros(128, 128)
        path = tmp_path / "wan22-fun-camera-high_noise.safetensors"
        save_file(sd, path)

        with pytest.raises(RuntimeError, match="control_adapter"):
            _load(path)

    def test_all_in_one_bundled_components_are_dropped_not_refused(self, tmp_path: Path) -> None:
        """The "all-in-one" packaging convention bundles transformer + VAE + CLIP in one
        file so ComfyUI's `Load Checkpoint` node can supply all three.
        Phr00t/WAN2.2-14B-Rapid-AllInOne and its ~110 GGUF conversions
        (befox/WAN2.2-14B-Rapid-AllInOne-GGUF) ship this way.

        These loaded fine before the unexpected-key backstop existed — InvokeAI sources
        the VAE and encoder from separately-wired models and simply ignores the bundled
        copies — so refusing them is a regression, not a safety check.
        """
        sd = _tiny_model().state_dict()
        sd["vae.decoder.conv_in.weight"] = torch.zeros(96, 16, 3, 3)
        sd["text_encoders.umt5xxl.shared.weight"] = torch.zeros(256, 32)
        sd["model_ema.patch_embedding.weight"] = torch.zeros(128, 16, 1, 2, 2)
        path = tmp_path / "wan2.2-t2v-rapid-aio-v10-high_noise.safetensors"
        save_file(sd, path)

        model = _load(path)
        # The transformer itself still loaded, and none of the bundled weights reached it.
        assert not hasattr(model, "vae")
        assert not hasattr(model, "text_encoders")

    def test_merged_lora_residue_is_dropped_not_refused(self, tmp_path: Path) -> None:
        """`configs.main._has_wan_transformer_block_weights` deliberately admits main
        models that retain merged-in LoRA tensors, using a *positive* structural test
        rather than a "reject anything with lora keys" exclusion. The loader has to
        agree, or the probe accepts a file that the loader then refuses with a reason
        blaming Animate/S2V/Fun-Camera.
        """
        sd = _tiny_model().state_dict()
        sd["blocks.0.attn1.to_q.lora_down.weight"] = torch.zeros(8, 128)
        sd["blocks.0.attn1.to_q.lora_up.weight"] = torch.zeros(128, 8)
        sd["blocks.0.attn1.to_q.alpha"] = torch.zeros(())
        path = tmp_path / "Wan2.2-T2V-A14B-high_noise-merged.safetensors"
        save_file(sd, path)

        _load(path)  # must not raise

    def test_unknown_extra_module_is_still_refused(self, tmp_path: Path) -> None:
        """The allowlist above must not turn the backstop off. A conditioning branch
        nobody has enumerated yet still has to fail loudly rather than load degraded.
        """
        sd = _tiny_model().state_dict()
        sd["vae.decoder.conv_in.weight"] = torch.zeros(96, 16, 3, 3)  # benign, alongside
        sd["mystery_adapter.proj.weight"] = torch.zeros(128, 128)
        path = tmp_path / "wan22-unknown-variant-high_noise.safetensors"
        save_file(sd, path)

        with pytest.raises(RuntimeError, match="mystery_adapter"):
            _load(path)

    def test_missing_parameter_is_reported(self, tmp_path: Path) -> None:
        sd = _tiny_model().state_dict()
        del sd["blocks.1.attn1.to_q.weight"]
        path = tmp_path / "wan22-truncated-high_noise.safetensors"
        save_file(sd, path)

        with pytest.raises(RuntimeError, match="blocks.1.attn1.to_q.weight"):
            _load(path)


class TestSubmodelGuard:
    @pytest.mark.parametrize("submodel", [None, SubModelType.VAE, SubModelType.TextEncoder])
    def test_only_the_transformer_submodel_is_served(self, submodel) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Wan_Config

        config = MagicMock(spec=Main_Checkpoint_Wan_Config)
        with pytest.raises(ValueError, match="Transformer"):
            _make_loader()._load_model(config, submodel)

    def test_wrong_config_class_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="Main_Checkpoint_Wan_Config"):
            _make_loader()._load_model(MagicMock(), SubModelType.Transformer)
