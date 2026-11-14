"""Tests for the single-file Wan 2.2 checkpoint loader (WanCheckpointModel).

Round-trips a real (tiny) ``WanTransformer3DModel`` through the on-disk formats
the community actually ships, so the shape-driven architecture inference is
checked against diffusers' own module tree rather than against a hand-written
expectation.
"""

import json
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.wan import (
    _WAN_NATIVE_TO_DIFFUSERS_RENAMES,
    WanCheckpointModel,
    _build_wan_transformer_config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType, WanVariantType
from invokeai.backend.quantization.fp8_scaled import is_scale_metadata_key, set_full_precision_hints_respected
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from tests.fixtures.quantized_payloads import (
    comfy_quant_marker,
    mxfp8_tensors,
    nvfp4_signed_tensors,
    quantize_convrot,
    quantize_scaled_fp8,
)
from tests.fixtures.safetensors_files import needs_mapping_that_shows_file_writes, overwrite_payloads

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


def _recording_model() -> tuple[MagicMock, dict[str, torch.Tensor]]:
    """A stand-in transformer that snapshots the state dict it is handed.

    Reading `call_args.args[0]` afterwards is not enough any more: the loader drops its own references
    to that dict before the FP8 cast (so the bf16 originals are not held alive beside their fp8 copies),
    and the assertion would then run against an emptied dict.
    """
    model = MagicMock()
    handed_over: dict[str, torch.Tensor] = {}

    def record(state_dict, *args, **kwargs):
        handed_over.update(state_dict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    model.load_state_dict.side_effect = record
    return model, handed_over


def _make_loader() -> WanCheckpointModel:
    loader = object.__new__(WanCheckpointModel)
    loader._ram_cache = MagicMock()
    loader._torch_device = torch.device("cpu")
    return loader


def _load(
    path: Path,
    variant: WanVariantType = WanVariantType.T2V_A14B,
    loader: WanCheckpointModel | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    config = MagicMock()
    config.path = str(path)
    config.variant = variant

    with (
        patch("invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_torch_device"),
        patch(
            "invokeai.backend.model_manager.load.model_loaders.wan.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=dtype,
        ),
    ):
        return (loader or _make_loader())._load_from_singlefile(config)


class TestArchitectureInference:
    def test_matches_the_model_it_came_from(self) -> None:
        sd = _tiny_model().state_dict()
        inferred = _build_wan_transformer_config(sd, source="test")
        assert inferred == TINY_MODEL_KWARGS

    def test_reports_the_missing_key(self) -> None:
        sd = _tiny_model().state_dict()
        del sd["proj_out.weight"]
        with pytest.raises(RuntimeError, match="proj_out.weight"):
            _build_wan_transformer_config(sd, source="test")

    def test_out_channels_is_read_from_proj_out_not_assumed_equal_to_in_channels(self) -> None:
        """I2V-A14B takes 36 channels in (16 noise + 16 ref-image latents + 4 mask) and
        returns 16 — only the noise prediction comes back. Every other fixture here is
        16-in/16-out, which cannot tell the two apart."""
        from diffusers import WanTransformer3DModel

        asymmetric = {**TINY_MODEL_KWARGS, "in_channels": 36, "out_channels": 16}
        sd = WanTransformer3DModel(**asymmetric).state_dict()

        inferred = _build_wan_transformer_config(sd, source="test")

        assert inferred["in_channels"] == 36
        assert inferred["out_channels"] == 16
        assert inferred == asymmetric

    def test_counts_layers_from_the_highest_block_index(self) -> None:
        model = _tiny_model()
        sd = model.state_dict()
        assert _build_wan_transformer_config(sd, source="test")["num_layers"] == 2
        # Dropping the last block's keys must drop the inferred layer count too —
        # the count comes from the weights, not from a per-variant lookup table.
        trimmed = {k: v for k, v in sd.items() if not k.startswith("blocks.1.")}
        assert _build_wan_transformer_config(trimmed, source="test")["num_layers"] == 1


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

    def test_an_nvfp4_checkpoint_is_refused_rather_than_folded_over_its_packed_codes(self, tmp_path: Path) -> None:
        """Wan reads scaled fp8 but never calls `pop_nvfp4_layers`, and the shared fold has no dtype
        gate, so an nvfp4 layer was multiplied by its block-scale grid and logged as dequantized.
        The load then died on the width -- "size mismatch [128, 64] vs [128, 128]" -- which names
        neither the scheme nor the remedy, after a line claiming success. Pinned at the loader
        because that is where the misleading sequence was visible.
        """
        reference = _tiny_model()
        sd = {k: v.clone() for k, v in reference.state_dict().items()}

        target = "blocks.0.attn1.to_q"
        rows, columns = sd[f"{target}.weight"].shape
        packed, _ = nvfp4_signed_tensors(target, torch.randint(0, 2, (rows, columns), dtype=torch.bool))
        sd.update(packed)
        path = tmp_path / "Wan2.2-A14B-HighNoise-nvfp4.safetensors"
        save_file(sd, path)

        message = r"Wan checkpoint Wan2\.2-A14B-HighNoise-nvfp4\.safetensors.*does not support nvfp4"
        with pytest.raises(ValueError, match=message):
            _load(path)

    def test_scale_bookkeeping_never_reaches_the_model(self, tmp_path: Path) -> None:
        """``load_state_dict(strict=False)`` silently ignores unexpected keys, so
        assert on what the loader actually hands over rather than on the result."""
        target = "blocks.0.attn1.to_q.weight"
        sd = _tiny_model().state_dict()
        sd[target] = torch.full_like(sd[target], 0.5).to(torch.float8_e4m3fn)
        sd["blocks.0.attn1.to_q.scale_weight"] = torch.tensor([4.0])
        sd["blocks.0.attn1.to_q.scale_input"] = torch.tensor([1.0])
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)

        path = tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors"
        save_file(sd, path)

        model, handed_over = _recording_model()
        loader = _make_loader()
        loader._should_use_fp8 = lambda *_args: False
        with patch("diffusers.WanTransformer3DModel", return_value=model):
            _load(path, loader=loader)

        assert not [k for k in handed_over if k.endswith((".scale_weight", ".scale_input")) or k == "scaled_fp8"]
        # ...without eating scale_shift_table, which is a real Wan parameter.
        assert "scale_shift_table" in handed_over
        # Folded with its scale: 0.5 * 4.0.
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

        model, handed_over = _recording_model()
        with patch("diffusers.WanTransformer3DModel", return_value=model):
            _load(path)

        # Assert on the dict actually handed over. `hasattr(model, "vae")` would be a
        # tautology — a freshly built WanTransformer3DModel has no such attribute either
        # way — and it would not catch the bundled weights being cast and RAM-reserved
        # before load_state_dict discarded them, which is the cost this avoids.
        assert [k for k in handed_over if k.startswith(("vae.", "text_encoders.", "model_ema."))] == []
        assert "patch_embedding.weight" in handed_over

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

    @pytest.mark.parametrize(
        "residue",
        [
            pytest.param(["blocks.0.attn1.to_q.lora_A.weight", "blocks.0.attn1.to_q.lora_B.weight"], id="peft"),
            pytest.param(["blocks.0.attn1.to_q.lokr_w1", "blocks.0.attn1.to_q.lokr_w2"], id="lokr"),
            pytest.param(["blocks.0.attn1.to_q.hada_w1_a", "blocks.0.attn1.to_q.hada_w2_a"], id="loha"),
            pytest.param(["blocks.0.attn1.to_q.dora_scale"], id="dora"),
        ],
    )
    def test_every_lycoris_family_the_probe_accepts_also_loads(self, tmp_path: Path, residue: list[str]) -> None:
        """The allowlist has to cover the same families as `LoRA_LyCORIS_Wan_Config`'s
        suffix set, or the probe accepts a merged file the loader then refuses — with a
        message blaming Animate/S2V/Fun-Camera, which is no help at all."""
        sd = _tiny_model().state_dict()
        for key in residue:
            sd[key] = torch.zeros(8, 8)
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


class TestHeaderFirstRead:
    """What the header settles before any payload is read, and what the loaded model still holds of the file."""

    @staticmethod
    def _load_recording(
        path: Path, loader: WanCheckpointModel | None = None
    ) -> tuple[torch.nn.Module, list[tuple], list[set[str]]]:
        """Load through the real reader, logging each payload it hands out and each reservation, in order.

        Also returns, per read, the earlier payloads that are not bf16 and were still alive: source tensors that a
        conversion on arrival has already replaced by then, unless the load keeps them as stored.
        """
        events: list[tuple] = []
        alive_at_read: list[set[str]] = []
        sources: list[tuple[str, weakref.ref]] = []
        real_safe_open = safetensors.safe_open

        def handed_out(key: str, tensor: torch.Tensor) -> torch.Tensor:
            events.append(("read", key))
            alive = [(source, ref()) for source, ref in sources]
            alive_at_read.append({source for source, t in alive if t is not None and t.dtype != torch.bfloat16})
            del alive
            sources.append((key, weakref.ref(tensor)))
            return tensor

        class Handle:
            def __init__(self, handle) -> None:
                self._handle = handle

            def __getattr__(self, name: str):
                return getattr(self._handle, name)

            def get_tensor(self, key: str) -> torch.Tensor:
                return handed_out(key, self._handle.get_tensor(key))

            def get_tensors(self) -> dict[str, torch.Tensor]:
                # What `load_file` reads through: the whole file.
                return {key: handed_out(key, tensor) for key, tensor in self._handle.get_tensors().items()}

        @contextmanager
        def safe_open(*args, **kwargs):
            with real_safe_open(*args, **kwargs) as handle:
                yield Handle(handle)

        loader = loader or _make_loader()
        loader._ram_cache.make_room.side_effect = lambda size: events.append(("reserve", size))
        dtype = torch.float32 if "int8" in path.name else torch.bfloat16
        with patch.object(safetensors, "safe_open", safe_open), patch.object(safetensors.torch, "safe_open", safe_open):
            model = _load(path, loader=loader, dtype=dtype)
        return model, events, alive_at_read

    @staticmethod
    def _load_mapped(path: Path, dtype: torch.dtype) -> torch.nn.Module:
        """The same load with the reader on safetensors' default mmap backend: the control for the overwrite probe."""
        real_safe_open = safetensors.safe_open

        def mapped(*args, backend: str = "mmap", **kwargs):
            return real_safe_open(*args, **kwargs)

        with patch.object(safetensors, "safe_open", mapped):
            return _load(path, dtype=dtype)

    @staticmethod
    def _all_in_one(tmp_path: Path) -> tuple[torch.nn.Module, Path]:
        """An fp16 transformer under ComfyUI's prefix, bundled with a VAE and a text encoder."""
        reference = _tiny_model()
        sd = {f"model.diffusion_model.{k}": v.half() for k, v in reference.state_dict().items()}
        sd["vae.decoder.conv_in.weight"] = torch.ones(64, 64, dtype=torch.float16)
        sd["text_encoders.umt5xxl.transformer.shared.weight"] = torch.ones(64, 64, dtype=torch.float16)
        path = tmp_path / "wan2.2_t2v_high_noise_14B_aio.safetensors"
        save_file(sd, path)
        return reference, path

    def test_bundled_components_are_never_read(self, tmp_path: Path) -> None:
        reference, path = self._all_in_one(tmp_path)

        _, events, _ = self._load_recording(path)

        read = [key for kind, key in events if kind == "read"]
        assert sorted(read) == sorted(f"model.diffusion_model.{k}" for k in reference.state_dict())

    def test_room_for_the_cast_size_is_made_before_the_first_read(self, tmp_path: Path) -> None:
        reference, path = self._all_in_one(tmp_path)

        model, events, _ = self._load_recording(path)

        bf16_size = sum(v.numel() for v in reference.state_dict().values()) * 2
        assert events[0] == ("reserve", bf16_size)
        assert [event for event in events if event[0] == "reserve"] == [("reserve", bf16_size)]
        assert all(t.dtype == torch.bfloat16 for t in model.state_dict().values())

    def test_each_tensor_is_cast_before_the_next_is_read(self, tmp_path: Path) -> None:
        """The peak stays the finished state dict plus the tensor in flight, never the whole fp16 source."""
        _, path = self._all_in_one(tmp_path)

        _, _, alive_at_read = self._load_recording(path)

        assert len(alive_at_read) > 2
        assert not set().union(*alive_at_read)

    def test_scaled_fp8_room_is_the_folded_size_and_its_metadata_is_never_read(self, tmp_path: Path) -> None:
        reference = _tiny_model()
        sd = {k: v.clone() for k, v in reference.state_dict().items()}
        target = "blocks.0.attn1.to_q.weight"
        sd[target] = torch.full_like(sd[target], 0.5).to(torch.float8_e4m3fn)
        sd["blocks.0.attn1.to_q.scale_weight"] = torch.full((128,), 4.0)
        sd["blocks.0.attn1.to_q.scale_input"] = torch.tensor([1.0])
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)
        path = tmp_path / "Wan2.2-A14B-HighNoise-fp8_scaled.safetensors"
        save_file(sd, path)

        model, events, alive_at_read = self._load_recording(path)

        assert events[0] == ("reserve", sum(v.numel() for v in reference.state_dict().values()) * 2)
        read = {key for kind, key in events if kind == "read"}
        assert read == set(reference.state_dict()) | {"blocks.0.attn1.to_q.scale_weight"}
        # Only the fp8 weight is still alive while its scale is read; it is folded right after.
        assert max(len(alive) for alive in alive_at_read) == 1
        folded = model.state_dict()[target]
        assert folded.dtype == torch.bfloat16
        assert torch.equal(folded.float(), torch.full_like(folded.float(), 2.0))

    def test_an_int8_checkpoint_reserves_before_any_weight_is_read(self, tmp_path: Path) -> None:
        """Only the markers are read first. The reservation is every payload at its stored width: dense float32 at
        the float32 compute dtype, the codes one byte each, the scales float32."""
        _, sd = _int8_checkpoint(BLOCK_LAYERS)
        path = _write(tmp_path, sd)

        _, events, _ = self._load_recording(path)

        reserves = [event for event in events if event[0] == "reserve"]
        first_reserve = events.index(reserves[0])
        assert all(key.endswith(".comfy_quant") for _, key in events[:first_reserve])
        assert reserves == [("reserve", sum(v.nbytes for k, v in sd.items() if not k.endswith(".comfy_quant")))]
        assert len(events[first_reserve + 1 :]) == len(sd) - len(BLOCK_LAYERS)

    @needs_mapping_that_shows_file_writes
    def test_a_bf16_checkpoint_leaves_nothing_reading_the_file(self, tmp_path: Path) -> None:
        """bf16 on disk is the case where the cast is a no-op: a mapped tensor would come through it still mapped."""
        reference = _tiny_model().to(torch.bfloat16)
        path = tmp_path / "wan22-t2v-a14b-high_noise.safetensors"
        save_file(reference.state_dict(), path)
        mapped = self._load_mapped(path, torch.bfloat16)

        model = _load(path)
        overwrite_payloads(path)

        assert any(not torch.equal(mapped.state_dict()[k], v) for k, v in reference.state_dict().items()), (
            "the probe sees no change through mmap either, so it cannot tell a copy from a view"
        )
        for key, value in reference.state_dict().items():
            assert torch.equal(model.state_dict()[key], value), key

    @needs_mapping_that_shows_file_writes
    def test_an_int8_checkpoint_leaves_nothing_reading_the_file(self, tmp_path: Path) -> None:
        """The int8 codes are kept as stored, so they are never copied by a cast either."""
        _, sd = _int8_checkpoint(BLOCK_LAYERS)
        path = _write(tmp_path, sd)
        mapped = self._load_mapped(path, torch.float32)
        mapped_before = {key: value.clone() for key, value in mapped.state_dict().items()}

        model = _load(path, dtype=torch.float32)
        loaded = {key: value.clone() for key, value in model.state_dict().items()}
        overwrite_payloads(path)

        assert any(not torch.equal(v, mapped_before[k]) for k, v in mapped.state_dict().items()), (
            "the probe sees no change through mmap either, so it cannot tell a copy from a view"
        )
        for key, value in model.state_dict().items():
            assert torch.equal(value, loaded[key]), key

    def test_an_mx_exponent_grid_is_refused_by_name(self, tmp_path: Path) -> None:
        """An E8M0 grid has to reach the MXFP8 refusal; a header dtype the reader cannot map would stop it short."""
        reference = _tiny_model()
        sd = {k: v.clone() for k, v in reference.state_dict().items()}
        target = "blocks.0.attn1.to_q"
        sd[f"{target}.weight"] = sd[f"{target}.weight"].to(torch.float8_e4m3fn)
        sd[f"{target}.weight_scale"] = torch.full((128, 4), 1.0).to(torch.float8_e8m0fnu)
        path = tmp_path / "Wan2.2-A14B-HighNoise-mxfp8.safetensors"
        save_file(sd, path)

        with pytest.raises(NotImplementedError, match="MXFP8"):
            _load(path)


# --- ComfyUI int8_tensorwise (convrot) -------------------------------------------------------------------------

GROUP = 16  # a power of 4 that divides every tiny input width, 32 (text_dim) up to 256 (freq_dim)
MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP, "per_row": True}

ATTENTION = [
    f"blocks.{block}.{attn}.{proj}"
    for block in range(2)
    for attn in ("attn1", "attn2")
    for proj in ("to_q", "to_k", "to_v", "to_out.0")
]
FFN = [f"blocks.{block}.ffn.net.{index}{suffix}" for block in range(2) for index, suffix in ((0, ".proj"), (2, ""))]
BLOCK_LAYERS = ATTENTION + FFN
# `WanTimeTextImageEmbedding.forward` reads `next(iter(time_embedder.parameters())).dtype`, so an
# `Int8ConvrotLinear` there (buffers only) would raise StopIteration at the first forward.
EMBEDDERS = [
    "condition_embedder.text_embedder.linear_1",
    "condition_embedder.text_embedder.linear_2",
    "condition_embedder.time_embedder.linear_1",
    "condition_embedder.time_embedder.linear_2",
    "condition_embedder.time_proj",
]


def _int8_checkpoint(
    layers: list[str],
    *,
    marker: dict | None = MARKER,
    dense_dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """A diffusers-layout state dict with ``layers`` stored as int8 + convrot, and the dense model it decodes to.

    The reference carries the *round trip* of each payload (what a correct loader must reconstruct), taken from the
    fixture's own quantizer rather than from the code under test.
    """
    torch.manual_seed(0)
    reference = _tiny_model().float()
    sd = {key: value.detach().clone().to(dense_dtype) for key, value in reference.state_dict().items()}
    for path in layers:
        weight = reference.get_submodule(path).weight
        payload = quantize_convrot(weight.detach().float(), group_size=GROUP)
        with torch.no_grad():
            weight.copy_(payload.dequantized)
        sd[f"{path}.weight"] = payload.codes
        sd[f"{path}.weight_scale"] = payload.scale
        if marker is not None:
            sd[f"{path}.comfy_quant"] = comfy_quant_marker(marker)
    return reference, sd


def _write(
    tmp_path: Path,
    sd: dict[str, torch.Tensor],
    *,
    native: bool = True,
    prefix: str = "",
    header_layers: list[str] | None = None,
    header_prefix: str = "",
) -> Path:
    """Save ``sd`` as the community ships it: native or diffusers names, optional ComfyUI prefix, optional header."""
    if native:
        sd = _to_native_layout(sd)
    metadata = None
    if header_layers is not None:
        # The release names header entries in native names; `header_prefix` is the ComfyUI wrapper some tools keep.
        names = _to_native_layout({f"{path}.weight": torch.empty(0) for path in header_layers})
        entry = {key: value for key, value in MARKER.items() if key != "per_row"}
        layers = {f"{header_prefix}{name[: -len('.weight')]}": entry for name in names}
        metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    path = tmp_path / "wan2.2_t2v_high_noise_14B_int8_convrot.safetensors"
    save_file({f"{prefix}{key}": value.contiguous() for key, value in sd.items()}, path, metadata=metadata)
    return path


def _forward(model: torch.nn.Module) -> torch.Tensor:
    torch.manual_seed(1)
    dtype = next(model.buffers()).dtype if not list(model.parameters()) else next(model.parameters()).dtype
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, 16, 1, 4, 4, dtype=dtype),
            timestep=torch.tensor([500]),
            encoder_hidden_states=torch.randn(1, 6, 32, dtype=dtype),
            return_dict=False,
        )[0]


def _assert_decodes_like_the_reference(model: torch.nn.Module, reference: torch.nn.Module, layers: list[str]) -> None:
    """The kept layers reconstruct the reference weight, and the whole model computes what the reference computes."""
    for path in layers:
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.int8, path
        assert layer.convrot is True and layer.group_size == GROUP, path
        reconstructed = layer._dequantized_weight(torch.device("cpu"), torch.float32)
        torch.testing.assert_close(reconstructed, reference.get_submodule(path).weight, rtol=1e-4, atol=1e-5)
    expected = _forward(reference)
    assert expected.abs().max() > 0, "reference output is degenerate, the comparison would pass vacuously"
    torch.testing.assert_close(_forward(model), expected, rtol=1e-3, atol=1e-4)


class TestInt8Convrot:
    @pytest.mark.parametrize("native", [True, False], ids=["native-keys", "diffusers-keys"])
    @pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bare", "comfyui-prefix"])
    def test_layout_variants_keep_the_layers_int8_and_decode_to_the_reference(
        self, tmp_path: Path, native: bool, prefix: str
    ) -> None:
        reference, sd = _int8_checkpoint(BLOCK_LAYERS)
        path = _write(tmp_path, sd, native=native, prefix=prefix)

        model = _load(path, dtype=torch.float32)

        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    def test_markers_declared_only_in_the_header_are_honoured(self, tmp_path: Path) -> None:
        reference, sd = _int8_checkpoint(BLOCK_LAYERS, marker=None)
        path = _write(tmp_path, sd, header_layers=BLOCK_LAYERS)

        model = _load(path, dtype=torch.float32)

        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    @pytest.mark.parametrize("header_prefix", ["", "model.diffusion_model."], ids=["bare-header", "prefixed-header"])
    def test_a_prefixed_file_with_header_only_markers_loads(self, tmp_path: Path, header_prefix: str) -> None:
        """The loader compares header names with the keys it has already stripped, so a header that kept the ComfyUI
        wrapper still has to find its layers."""
        reference, sd = _int8_checkpoint(BLOCK_LAYERS, marker=None)
        path = _write(
            tmp_path, sd, prefix="model.diffusion_model.", header_layers=BLOCK_LAYERS, header_prefix=header_prefix
        )

        model = _load(path, dtype=torch.float32)

        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    def test_activation_scales_are_dropped_not_loaded(self, tmp_path: Path) -> None:
        """`input_scale` is W8A8 metadata this path does not use. Left in the dict it would be an unexpected key,
        which the Wan loader refuses."""
        reference, sd = _int8_checkpoint(BLOCK_LAYERS)
        for layer in BLOCK_LAYERS:
            sd[f"{layer}.input_scale"] = torch.ones(())
        path = _write(tmp_path, sd)

        model = _load(path, dtype=torch.float32)

        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    def test_stray_fp8_metadata_is_dropped_as_on_the_fp8_path(self, tmp_path: Path) -> None:
        """The fp8 branch drops a top-level `scaled_fp8` flag and a weight scale beside a dense weight silently. Left
        in the dict here they would be unexpected keys, which the Wan loader blames on an Animate/S2V variant."""
        reference, sd = _int8_checkpoint(BLOCK_LAYERS)
        sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)
        sd["proj_out.weight_scale"] = torch.ones(())
        path = _write(tmp_path, sd)

        model = _load(path, dtype=torch.float32)

        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    def test_a_quantized_condition_embedder_is_widened_and_the_forward_runs(self, tmp_path: Path) -> None:
        """The time embedder's dtype probe reads a parameter, so those layers cannot stay int8. They are decoded
        densely at load, and the result must still be the reference weight."""
        reference, sd = _int8_checkpoint(BLOCK_LAYERS + EMBEDDERS)
        path = _write(tmp_path, sd)

        model = _load(path, dtype=torch.float32)

        for embedder in EMBEDDERS:
            layer = model.get_submodule(embedder)
            assert not isinstance(layer, Int8ConvrotLinear), embedder
            torch.testing.assert_close(layer.weight, reference.get_submodule(embedder).weight, rtol=1e-4, atol=1e-5)
        _assert_decodes_like_the_reference(model, reference, BLOCK_LAYERS)

    @pytest.mark.parametrize("embedders", [[], EMBEDDERS], ids=["standard-build", "embedders-quantized"])
    def test_the_reservation_charges_int8_payloads_one_byte_and_widened_layers_the_compute_width(
        self, tmp_path: Path, embedders: list[str]
    ) -> None:
        """A kept payload is one byte per code plus its fp32 scale column; everything else, the widened embedders and
        the F16 remainder included, arrives at the compute width. The blanket cast and its size would charge the
        codes two bytes and the scales two instead of four."""
        _, sd = _int8_checkpoint(BLOCK_LAYERS + embedders, dense_dtype=torch.float16)
        path = _write(tmp_path, sd)
        loader = _make_loader()

        _load(path, loader=loader)

        kept = set(BLOCK_LAYERS)
        expected = 0
        for key, tensor in _tiny_model().state_dict().items():
            layer = key[: -len(".weight")] if key.endswith(".weight") else None
            if layer in kept:
                expected += tensor.numel() * 1 + tensor.shape[0] * torch.float32.itemsize
            else:
                expected += tensor.numel() * torch.bfloat16.itemsize
        loader._ram_cache.make_room.assert_called_once_with(expected)

    def test_fp8_storage_leaves_the_int8_buffers_alone(self, tmp_path: Path) -> None:
        """The loader ends in the FP8 Storage pass for every path. It walks parameters and an ``Int8ConvrotLinear``
        holds buffers, so the codes and their fp32 scales must come out untouched while the dense remainder is
        cast."""
        dense_layer = BLOCK_LAYERS[-1]
        quantized = BLOCK_LAYERS[:-1]
        _, sd = _int8_checkpoint(quantized)
        path = _write(tmp_path, sd)
        loader = _make_loader()
        loader._should_use_fp8 = lambda *_args: True
        loader._torch_dtype = torch.bfloat16
        loader._logger = MagicMock()

        model = _load(path, loader=loader)

        for name in quantized:
            layer = model.get_submodule(name)
            assert isinstance(layer, Int8ConvrotLinear), name
            assert layer.weight.dtype is torch.int8, name
            assert layer.weight_scale.dtype is torch.float32, name
        # The pass really ran: a block Linear the checkpoint left dense is now stored as fp8.
        assert model.get_submodule(dense_layer).weight.dtype is torch.float8_e4m3fn

    def test_an_int8_weight_without_a_marker_is_refused(self, tmp_path: Path) -> None:
        """Cast as raw codes it would be a model of small integers that loads without complaint."""
        _, sd = _int8_checkpoint(BLOCK_LAYERS, marker=None)
        path = _write(tmp_path, sd)

        # The file name is in the message so a user with two experts can tell which one failed.
        message = r"Wan 2\.2 \(wan2\.2_t2v_high_noise_14B_int8_convrot\.safetensors\) checkpoint has \d+ int8 weight"
        with pytest.raises(ValueError, match=message + r".*no readable int8_tensorwise marker"):
            _load(path)

    def test_an_int4_marked_file_is_refused(self, tmp_path: Path) -> None:
        _, sd = _int8_checkpoint(BLOCK_LAYERS, marker={"format": "int4_tensorwise", "convrot": True})
        path = _write(tmp_path, sd)

        with pytest.raises(ValueError, match="marked 'int4_tensorwise'"):
            _load(path)

    def test_a_marker_without_a_format_is_refused(self, tmp_path: Path) -> None:
        """Files marked only `{"convrot": true}` do not say what they are, so nothing is guessed."""
        _, sd = _int8_checkpoint(BLOCK_LAYERS, marker={"convrot": True})
        path = _write(tmp_path, sd)

        with pytest.raises(ValueError, match="no readable int8_tensorwise marker"):
            _load(path)

    def test_a_file_mixing_int8_with_scaled_fp8_is_refused(self, tmp_path: Path) -> None:
        """Inside the int8 branch the fp8 fold is skipped, so an fp8 layer would load without its scale."""
        reference, sd = _int8_checkpoint(BLOCK_LAYERS)
        fp8 = quantize_scaled_fp8(reference.get_submodule("proj_out").weight.detach().float())
        sd["proj_out.weight"] = fp8.codes
        sd["proj_out.weight_scale"] = fp8.scale
        path = _write(tmp_path, sd)

        with pytest.raises(ValueError, match="mixing convrot layers with scaled fp8"):
            _load(path)


# --- ComfyUI fp8_scaled ------------------------------------------------------------------------------------------

FP8_MARKER = {"format": "float8_e4m3fn"}
E5M2_MAX = 57344.0
# MXFP8 scales sit in cuBLAS tiles of 128 rows by 4 blocks of 32 weights; only the 128x128 attention projections fit.
MX_BLOCK = 32


@pytest.fixture
def hints_respected():
    set_full_precision_hints_respected(True)
    yield
    set_full_precision_hints_respected(None)


def _scaled_fp8_checkpoint(
    layers: list[str],
    *,
    markers: bool = True,
    e5m2: tuple[str, ...] = (),
    hinted: tuple[str, ...] = (),
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A diffusers-layout state dict with ``layers`` stored as ComfyUI scaled fp8, and the dense model it decodes to.

    The reference carries each layer's round trip (codes times scale) from the fixture's own quantizer, which is
    what a correct loader must reconstruct. The third value is the scales by layer path. ``e5m2`` layers are stored in
    that float8 flavour, ``hinted`` ones carry a per-layer ``full_precision_matrix_mult`` marker.
    """
    torch.manual_seed(0)
    reference = _tiny_model().float()
    sd = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    scales: dict[str, torch.Tensor] = {}
    for path in layers:
        weight = reference.get_submodule(path).weight
        if path in e5m2:
            scale = (weight.detach().abs().max() / E5M2_MAX).float()
            codes = (weight.detach() / scale).to(torch.float8_e5m2)
            decoded = codes.float() * scale
        else:
            payload = quantize_scaled_fp8(weight.detach().float())
            scale, codes, decoded = payload.scale, payload.codes, payload.dequantized
        with torch.no_grad():
            weight.copy_(decoded)
        sd[f"{path}.weight"] = codes
        sd[f"{path}.weight_scale"] = scale
        scales[path] = scale
        if markers:
            extra = {"full_precision_matrix_mult": True} if path in hinted else {}
            sd[f"{path}.comfy_quant"] = comfy_quant_marker({**FP8_MARKER, **extra})
    sd["scaled_fp8"] = torch.zeros(1, dtype=torch.float8_e4m3fn)
    return reference, sd, scales


def _mxfp8_checkpoint(layers: list[str]) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """``layers`` (128x128 projections) stored as MXFP8 with all-ones codes and a varied exponent grid, plus the dense
    model they decode to. The expected weight is the grid itself, expanded over its blocks of 32."""
    torch.manual_seed(0)
    reference = _tiny_model().float()
    sd = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    for path in layers:
        tensors, block_scale = mxfp8_tensors(path, torch.randint(120, 128, (128, 128 // MX_BLOCK)).float())
        sd.update(tensors)
        sd[f"{path}.comfy_quant"] = comfy_quant_marker({"format": "mxfp8"})
        with torch.no_grad():
            reference.get_submodule(path).weight.copy_(block_scale.repeat_interleave(MX_BLOCK, dim=1))
    return reference, sd


def _storage_loader(monkeypatch: pytest.MonkeyPatch, *, storage: bool, fp8_matmul: bool = False) -> WanCheckpointModel:
    """A loader whose settings are as given and whose FP8 Storage cast is the real one."""
    loader = _make_loader()
    loader._should_use_fp8 = lambda *_args: storage
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: fp8_matmul)
    return loader


def _round_trip_through_fp8_storage(reference: torch.nn.Module, layers: list[str]) -> None:
    """Make ``reference`` hold what FP8 Storage leaves behind in the layers the checkpoint shipped dense (biases too)."""
    for path in layers:
        for param in reference.get_submodule(path).parameters():
            with torch.no_grad():
                param.copy_(param.to(torch.float8_e4m3fn).to(param.dtype))


def _assert_forwards_like(model: torch.nn.Module, reference: torch.nn.Module, **tolerance) -> None:
    expected = _forward(reference)
    assert expected.abs().max() > 0, "reference output is degenerate, the comparison would pass vacuously"
    torch.testing.assert_close(_forward(model), expected, **tolerance)


class TestScaledFp8:
    @pytest.mark.parametrize("native", [True, False], ids=["native-keys", "diffusers-keys"])
    @pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bare", "comfyui-prefix"])
    def test_fp8_storage_keeps_the_scaled_layers_packed_and_forwards_like_the_folded_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool, prefix: str
    ) -> None:
        reference, sd, scales = _scaled_fp8_checkpoint(ATTENTION)
        path = _write(tmp_path, sd, native=native, prefix=prefix)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        for name in ATTENTION:
            layer = model.get_submodule(name)
            assert layer.weight.dtype is torch.float8_e4m3fn, name
            # The file's own scale, on the module the weight landed in rather than the one it was named after.
            torch.testing.assert_close(layer.weight_scale, scales[name], rtol=0, atol=0)
        for name in FFN:
            layer = model.get_submodule(name)
            assert layer.weight.dtype is torch.float8_e4m3fn, name
            assert getattr(layer, "weight_scale", None) is None, name
        _round_trip_through_fp8_storage(reference, FFN)
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    def test_scale_keys_never_reach_the_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unexpected key is blamed on an Animate/S2V variant, and `strict=False` would drop it silently
        otherwise, so assert on what the loader hands over."""
        _, sd, _ = _scaled_fp8_checkpoint(ATTENTION)
        for layer in ATTENTION:
            sd[f"{layer}.input_scale"] = torch.ones(())
        path = _write(tmp_path, sd)
        model, handed_over = _recording_model()
        with patch("diffusers.WanTransformer3DModel", return_value=model):
            _load(path, loader=_storage_loader(monkeypatch, storage=False), dtype=torch.float32)

        assert not [k for k in handed_over if is_scale_metadata_key(k)]
        assert "scale_shift_table" in handed_over

    def test_without_fp8_storage_or_the_fp8_matmul_the_file_is_folded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference, sd, _ = _scaled_fp8_checkpoint(ATTENTION)
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=False), dtype=torch.float32)

        assert not any(p.dtype is torch.float8_e4m3fn for p in model.parameters())
        assert not any(name.endswith("weight_scale") for name, _ in model.named_buffers())
        _assert_forwards_like(model, reference, rtol=1e-4, atol=1e-5)

    def test_the_fp8_matmul_keeps_the_layers_packed_without_casting_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference, sd, _ = _scaled_fp8_checkpoint(ATTENTION)
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=False, fp8_matmul=True), dtype=torch.float32)

        for name in ATTENTION:
            assert model.get_submodule(name).weight.dtype is torch.float8_e4m3fn, name
        for name in FFN:
            assert model.get_submodule(name).weight.dtype is torch.float32, name
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    def test_quantized_embedders_are_folded_so_the_condition_embedder_can_read_its_dtype(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`condition_embedder` is a model-declared skip pattern: a scaled layer there cannot stay fp8, and has to be
        folded with its scale rather than cast without it."""
        reference, sd, _ = _scaled_fp8_checkpoint(ATTENTION + EMBEDDERS)
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        for name in EMBEDDERS:
            layer = model.get_submodule(name)
            assert layer.weight.dtype is torch.float32, name
            torch.testing.assert_close(layer.weight, reference.get_submodule(name).weight, rtol=1e-6, atol=1e-7)
        _round_trip_through_fp8_storage(reference, FFN)
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    def test_a_scaled_layer_the_split_folds_is_not_re_encoded_unscaled_by_fp8_storage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """float8_e5m2 cannot stay packed, so its layer is folded with its scale. It then has no `weight_scale` and
        matches no skip pattern, which is exactly what the layerwise cast would otherwise take for a dense weight and
        store as unscaled e4m3: roughly 1% of its weights flushed to zero."""
        folded = "blocks.0.attn1.to_q"
        reference, sd, scales = _scaled_fp8_checkpoint(ATTENTION, e5m2=(folded,))
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        layer = model.get_submodule(folded)
        assert layer.weight.dtype is torch.float32
        torch.testing.assert_close(layer.weight, reference.get_submodule(folded).weight, rtol=1e-6, atol=1e-7)
        for name in ATTENTION:
            if name != folded:
                kept = model.get_submodule(name)
                assert kept.weight.dtype is torch.float8_e4m3fn, name
                torch.testing.assert_close(kept.weight_scale, scales[name], rtol=0, atol=0)
        _round_trip_through_fp8_storage(reference, FFN)
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    def test_a_file_whose_scaled_layers_are_all_folded_still_gets_no_unscaled_cast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing left to attach, the load must not fall through to the generic cast, which would store every
        folded layer unscaled. The dense remainder still goes to fp8."""
        reference, sd, _ = _scaled_fp8_checkpoint(ATTENTION, e5m2=tuple(ATTENTION))
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        for name in ATTENTION:
            assert model.get_submodule(name).weight.dtype is torch.float32, name
            torch.testing.assert_close(
                model.get_submodule(name).weight, reference.get_submodule(name).weight, rtol=1e-6, atol=1e-7
            )
        for name in FFN:
            assert model.get_submodule(name).weight.dtype is torch.float8_e4m3fn, name
        _round_trip_through_fp8_storage(reference, FFN)
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("storage", [False, True], ids=["storage-off", "storage-on"])
    def test_an_mxfp8_file_is_decoded_and_its_layers_stay_at_the_compute_dtype(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: bool
    ) -> None:
        """A block-wise scale has no fp8 matmul to feed, so those layers are folded at load. Under FP8 Storage they
        must not then be re-encoded without it: blocks whose scale is far below 1 would flush to zero."""
        reference, sd = _mxfp8_checkpoint(ATTENTION)
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=storage), dtype=torch.float32)

        for name in ATTENTION:
            weight = model.get_submodule(name).weight
            assert weight.dtype is torch.float32, name
            torch.testing.assert_close(weight, reference.get_submodule(name).weight, rtol=0, atol=0)
        if storage:
            _round_trip_through_fp8_storage(reference, FFN)
        apply_custom_layers_to_model(model)
        _assert_forwards_like(model, reference, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("header_prefix", ["", "model.diffusion_model."], ids=["bare-header", "prefixed-header"])
    def test_a_precision_hint_declared_only_in_the_header_reaches_the_renamed_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hints_respected: None, header_prefix: str
    ) -> None:
        """Header entries name layers natively. The flag has to land on the layer it was written for, in the model's
        own names, and on no other."""
        _, sd, _ = _scaled_fp8_checkpoint(ATTENTION, markers=False)
        hinted = ATTENTION[0]
        native_name = next(iter(_to_native_layout({f"{hinted}.weight": torch.empty(0)}))).removesuffix(".weight")
        header = {
            "_quantization_metadata": json.dumps(
                {
                    "format_version": "1.0",
                    "layers": {f"{header_prefix}{native_name}": {**FP8_MARKER, "full_precision_matrix_mult": True}},
                }
            )
        }
        path = tmp_path / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        save_file(
            {f"model.diffusion_model.{k}": v.contiguous() for k, v in _to_native_layout(sd).items()},
            path,
            metadata=header,
        )

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        assert model.get_submodule(hinted)._fp8_full_precision_matmul is True
        assert not any(model.get_submodule(name)._fp8_full_precision_matmul for name in ATTENTION[1:])

    @pytest.mark.parametrize("native", [True, False], ids=["native-keys", "diffusers-keys"])
    def test_a_precision_hint_carried_by_the_layers_own_marker_reaches_the_renamed_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hints_respected: None, native: bool
    ) -> None:
        """The marker tensors are deleted by the extraction, so their hints have to be read first, in the file's names,
        and then follow the weight's rename."""
        hinted = ATTENTION[-1]
        _, sd, _ = _scaled_fp8_checkpoint(ATTENTION, hinted=(hinted,))
        path = _write(tmp_path, sd, native=native)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=True), dtype=torch.float32)

        assert model.get_submodule(hinted)._fp8_full_precision_matmul is True
        assert not any(model.get_submodule(name)._fp8_full_precision_matmul for name in ATTENTION[:-1])

    def test_the_reservation_charges_kept_layers_one_byte_and_the_rest_the_compute_width(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kept: one byte per code plus the float32 scale. Folded, the same layers would be charged the compute
        width, which is what made a 14B expert reserve twice what it holds."""
        _, sd, _ = _scaled_fp8_checkpoint(ATTENTION)
        path = _write(tmp_path, sd)
        loader = _storage_loader(monkeypatch, storage=True)

        _load(path, loader=loader, dtype=torch.float32)

        kept = set(ATTENTION)
        expected = 0
        for key, tensor in _tiny_model().state_dict().items():
            if key.endswith(".weight") and key[: -len(".weight")] in kept:
                expected += tensor.numel() * 1 + torch.float32.itemsize
            else:
                expected += tensor.numel() * torch.float32.itemsize
        loader._ram_cache.make_room.assert_called_once_with(expected)

    def test_kept_layers_are_read_after_the_one_reservation_and_stay_as_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kept or not, the reservation is settled from the header: only the markers are read before it. A kept layer's
        codes and float32 scale arrive as stored and are never widened or copied, and its input scale is read only
        because the layer stays fp8. What is folded or cast is dead by the next read."""
        _, sd, scales = _scaled_fp8_checkpoint(ATTENTION + EMBEDDERS)
        calibrated = ATTENTION[0]
        # The spelling ComfyUI's own fp8_scaled files carry.
        sd[f"{calibrated}.scale_input"] = torch.tensor(0.5)
        codes = {name: sd[f"{name}.weight"].clone() for name in ATTENTION}
        path = _write(tmp_path, sd, native=False)
        path = path.rename(path.with_name("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"))

        model, events, alive_at_read = TestHeaderFirstRead._load_recording(
            path, loader=_storage_loader(monkeypatch, storage=True)
        )

        reserves = [index for index, event in enumerate(events) if event[0] == "reserve"]
        assert len(reserves) == 1
        assert all(key.endswith(".comfy_quant") for _, key in events[: reserves[0]])
        read = {key for kind, key in events[reserves[0] + 1 :] if kind == "read"}
        assert read == (
            set(_tiny_model().state_dict())
            | {f"{name}.weight_scale" for name in ATTENTION + EMBEDDERS}
            | {f"{calibrated}.scale_input"}
        )
        # At each weight read, the earlier payloads still alive (bf16 aside) are the kept codes and at most their
        # scales: the folded embedders' codes and scales and the cast float32 sources are gone.
        kept_codes = {f"{name}.weight" for name in ATTENTION}
        kept_scales = {f"{name}.weight_scale" for name in ATTENTION} | {f"{calibrated}.scale_input"}
        reads = [key for kind, key in events if kind == "read"]
        weight_reads = 0
        for index, (key, alive) in enumerate(zip(reads, alive_at_read, strict=True)):
            if key in sd and not is_scale_metadata_key(key):
                assert {source for source in alive if not is_scale_metadata_key(source)} == kept_codes & set(
                    reads[:index]
                ), key
                assert {source for source in alive if is_scale_metadata_key(source)} <= kept_scales, key
                weight_reads += 1
        assert weight_reads == len(_tiny_model().state_dict())
        for name in ATTENTION:
            layer = model.get_submodule(name)
            assert layer.weight.dtype is torch.float8_e4m3fn, name
            # Its own storage, one byte per code: neither a widened copy nor a slice of a larger buffer.
            assert layer.weight.untyped_storage().nbytes() == layer.weight.numel(), name
            assert torch.equal(layer.weight.view(torch.uint8), codes[name].view(torch.uint8)), name
            assert layer.weight_scale.dtype is torch.float32, name
            torch.testing.assert_close(layer.weight_scale, scales[name], rtol=0, atol=0)
        assert model.get_submodule(calibrated).input_scale.item() == 0.5
        for name in EMBEDDERS:
            assert model.get_submodule(name).weight.dtype is torch.bfloat16, name

    @pytest.mark.parametrize("storage", [False, True], ids=["storage-off", "storage-on"])
    def test_a_scale_beside_a_weight_that_is_not_fp8_is_folded_into_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: bool
    ) -> None:
        """ComfyUI's legacy `scaled_fp8` ops apply `scale_weight` whatever the weight's dtype: a "scaled" checkpoint is
        not all fp8. Such a layer is folded, and FP8 Storage leaves it at the compute dtype like any other folded
        scaled layer rather than re-encoding it without its scale."""
        target = ATTENTION[0]
        sd = {key: value.detach().clone() for key, value in _tiny_model().state_dict().items()}
        stored = sd[f"{target}.weight"].to(torch.bfloat16)
        sd[f"{target}.weight"] = stored
        sd[f"{target}.scale_weight"] = torch.tensor([0.25])
        path = _write(tmp_path, sd)

        model = _load(path, loader=_storage_loader(monkeypatch, storage=storage), dtype=torch.float32)

        loaded = model.get_submodule(target).weight
        assert loaded.dtype is torch.float32
        # A power-of-two scale, so the product is exact.
        torch.testing.assert_close(loaded, stored.float() * 0.25, rtol=0, atol=0)
        if storage:
            assert model.get_submodule(FFN[0]).weight.dtype is torch.float8_e4m3fn

    @pytest.mark.parametrize(
        ("storage", "fp8_matmul"),
        [(False, False), (False, True), (True, False)],
        ids=["folded", "fp8-matmul", "fp8-storage"],
    )
    def test_the_reservation_covers_what_the_load_holds_within_a_few_scalars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: bool, fp8_matmul: bool
    ) -> None:
        """The reservation is planned from the header, before any scale is read, so it has to agree with what the read
        ends up holding: kept per-tensor layers with both input-scale spellings, an e5m2 layer and the embedders the
        split folds, a scaled weight that is not fp8, and dense float32 tensors. Measured before the FP8 Storage cast,
        which shrinks the model afterwards; the slack is the scalars the plan counts for layers that end up folded."""
        e5m2 = ATTENTION[1]
        _, sd, _ = _scaled_fp8_checkpoint(ATTENTION + EMBEDDERS, e5m2=(e5m2,))
        sd[f"{ATTENTION[0]}.scale_input"] = torch.tensor(0.5)
        # An uncalibrated placeholder, which the extraction drops.
        sd[f"{ATTENTION[2]}.input_scale"] = torch.tensor(1.0)
        sd[f"{FFN[0]}.weight"] = sd[f"{FFN[0]}.weight"].to(torch.bfloat16)
        sd[f"{FFN[0]}.scale_weight"] = torch.tensor([0.5])
        path = _write(tmp_path, sd)
        loader = _storage_loader(monkeypatch, storage=storage, fp8_matmul=fp8_matmul)
        loader._apply_fp8_to_nn_module = lambda *_args, **_kwargs: None

        model = _load(path, loader=loader, dtype=torch.float32)

        held = sum(t.nelement() * t.element_size() for t in model.parameters()) + sum(
            b.nelement() * b.element_size()
            for name, b in model.named_buffers()
            if name.endswith((".weight_scale", ".input_scale"))
        )
        reserved = loader._ram_cache.make_room.call_args_list
        assert len(reserved) == 1
        slack = reserved[0].args[0] - held
        assert 0 <= slack <= 16 * torch.float32.itemsize
        if storage or fp8_matmul:
            assert model.get_submodule(ATTENTION[0]).weight.dtype is torch.float8_e4m3fn
            assert slack > 0, "nothing was folded after the plan counted it, so this would not test the bound"
