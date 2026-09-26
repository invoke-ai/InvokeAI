"""Qwen-Image's single-file loaders keep Comfy's nvfp4 builds packed.

Both Comfy files mix nvfp4 with scaled fp8, and an fp8 fold multiplies every `.weight_scale` into its weight -- nvfp4's
block scales included. So what these tests pin is the order: the nvfp4 layers leave the state dict before any fp8
handling, come back packed under the paths the model uses (the encoder's legacy `model.X` keys become
`model.language_model.X`), and one reservation is made before anything is widened. The transformer's scaled fp8 layers
are folded unless fp8 compute or the model's fp8 storage setting keeps them; the encoder's are always folded. The compute
dtype is bf16, as in production: the packed global scale must not be cast.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.configs.qwen_vl_encoder import QwenVLEncoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import qwen_image
from invokeai.backend.model_manager.load.model_loaders.qwen_image import (
    QwenImageCheckpointModel,
    QwenVLEncoderCheckpointLoader,
)
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_signed_tensors, quantize_convrot

COMPUTE_DTYPE = torch.bfloat16


def _nvfp4_tensors(path: str, shape: tuple[int, int]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Comfy ships an activation scale beside each layer, which the loader has no use for -- so it has
    to be dropped rather than reach `load_state_dict`, and it is here for that reason alone."""
    tensors, expected = nvfp4_signed_tensors(path, torch.randint(0, 2, shape, dtype=torch.bool))
    tensors[f"{path}.input_scale"] = torch.tensor(1.0)
    return tensors, expected


def _packed_bytes(rows: int, columns: int) -> int:
    return rows * columns // 2 + rows * (columns // 16) + 4


def _marker(fmt: str) -> torch.Tensor:
    return comfy_quant_marker({"format": fmt})


#: Both single-file encoder cells drive the same entry, so the driver is declared once.
ENCODER_SEAM = Seam(
    loader=QwenVLEncoderCheckpointLoader,
    module=qwen_image,
    entry="_load_text_encoder_from_singlefile",
    # The loader imports `load_file` inside the method, so the name it resolves is the package's.
    load_file_host=safetensors.torch,
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _patch_common(monkeypatch: pytest.MonkeyPatch, state_dict: dict, metadata: dict) -> list[bool]:
    """Patch file access and devices; return a log for `_record_fold`."""
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: state_dict)
    monkeypatch.setattr(qwen_image, "read_safetensors_metadata", lambda _path, _logger: metadata)
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: COMPUTE_DTYPE)
    return []


def _record_fold(monkeypatch: pytest.MonkeyPatch, loader, log: list[bool], *names: str) -> None:
    """At every call of the named `qwen_image` functions, log whether room had been made yet."""
    for name in names or ("_dequantize_comfyui_fp8",):
        original = getattr(qwen_image, name)

        def recording(*args, _original=original, **kwargs):
            log.append(loader._ram_cache.make_room.called)
            return _original(*args, **kwargs)

        monkeypatch.setattr(qwen_image, name, recording)


class _TinyQwenImageTransformer(torch.nn.Module):
    """The diffusers module names Comfy's Qwen-Image build uses, at toy width."""

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(64, 128)
        block.img_mlp = torch.nn.Module()
        block.img_mlp.net = torch.nn.ModuleList([torch.nn.Module()])
        block.img_mlp.net[0].proj = torch.nn.Linear(64, 256)
        block.txt_mlp = torch.nn.Module()
        block.txt_mlp.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(64, 128)])
        self.transformer_blocks = torch.nn.ModuleList([block])
        self.img_in = torch.nn.Linear(64, 128)


@pytest.mark.parametrize("mode", ["fold", "fp8_compute", "fp8_storage"])
def test_the_transformer_keeps_nvfp4_packed_and_scaled_fp8_only_where_something_uses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mode: str
) -> None:
    """Scaled fp8 layers stay fp8, with their own scale, for the fp8 matmul or for fp8 storage the model asks for; with
    neither they are folded. Under fp8 storage the dense remainder goes to fp8 too, but not the scaled layers, whose
    cast hooks would drop their scale."""
    import diffusers

    from invokeai.backend.model_manager.load import load_default
    from invokeai.backend.model_manager.taxonomy import SubModelType

    torch.manual_seed(0)
    state_dict: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    header: dict[str, dict[str, str]] = {}
    for path, shape in (
        ("transformer_blocks.0.attn.to_q", (128, 64)),
        ("transformer_blocks.0.img_mlp.net.0.proj", (256, 64)),
    ):
        tensors, weight = _nvfp4_tensors(path, shape)
        state_dict.update(tensors)
        state_dict[f"{path}.bias"] = torch.randn(shape[0])
        expected[path] = weight
        header[path] = {"format": "nvfp4"}
    fp8_values = torch.randint(-8, 9, (128, 64)).float()
    fp8 = "transformer_blocks.0.txt_mlp.net.2"
    state_dict[f"{fp8}.weight"] = fp8_values.to(torch.float8_e4m3fn)
    state_dict[f"{fp8}.weight_scale"] = torch.tensor(0.5)
    state_dict[f"{fp8}.input_scale"] = torch.tensor(1.0)
    state_dict[f"{fp8}.bias"] = torch.randn(128)
    header[fp8] = {"format": "float8_e4m3fn"}
    state_dict["img_in.weight"] = torch.randn(128, 64)
    state_dict["img_in.bias"] = torch.randn(128)

    log = _patch_common(monkeypatch, state_dict, {"_quantization_metadata": json.dumps({"layers": header})})
    monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _TinyQwenImageTransformer, raising=False)
    monkeypatch.setattr(qwen_image, "should_keep_fp8_weights", lambda _device: mode == "fp8_compute")
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_args: True)
    checkpoint = tmp_path / "qwen_image_nvfp4.safetensors"
    checkpoint.touch()
    loader = object.__new__(QwenImageCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()
    _record_fold(monkeypatch, loader, log, "dequantize_fp8_scaled", "split_fp8_scaled_layers")
    # The storage cast allocates an fp8 copy per weight; the loaded state dict must no longer hold the originals.
    entries_at_storage_cast: list[int] = []
    loader._apply_fp8_to_nn_module = lambda *args, **kwargs: (
        entries_at_storage_cast.append(len(state_dict)),
        QwenImageCheckpointModel._apply_fp8_to_nn_module(*args, **kwargs),
    )
    biases = {path: state_dict[f"{path}.bias"] for path in expected}
    config = Main_Checkpoint_QwenImage_Config.model_construct(
        path=str(checkpoint),
        name="qwen_image_nvfp4",
        default_settings=SimpleNamespace(fp8_storage=mode == "fp8_storage"),
    )

    model = loader._load_model(config, SubModelType.Transformer)

    for path, weight in expected.items():
        module = model.get_submodule(path)
        assert isinstance(module, NVFP4Linear), path
        assert module.weight.dtype is torch.uint8, path
        assert module.weight_scale_2.dtype is torch.float32, path
        x = torch.randn(3, module.in_features, dtype=COMPUTE_DTYPE)
        expected_out = torch.nn.functional.linear(x, weight.to(COMPUTE_DTYPE), biases[path].to(COMPUTE_DTYPE))
        torch.testing.assert_close(module(x), expected_out)
    fp8_layer = model.transformer_blocks[0].txt_mlp.net[2]
    if mode == "fold":
        assert torch.equal(fp8_layer.weight, (fp8_values * 0.5).to(COMPUTE_DTYPE))
        assert getattr(fp8_layer, "weight_scale", None) is None
    else:
        assert fp8_layer.weight.dtype is torch.float8_e4m3fn
        assert torch.equal(fp8_layer.weight.float(), fp8_values)
        assert torch.equal(fp8_layer.weight_scale.float(), torch.tensor(0.5))
    assert model.img_in.weight.dtype is (torch.float8_e4m3fn if mode == "fp8_storage" else COMPUTE_DTYPE)
    assert getattr(model.img_in, "weight_scale", None) is None
    # Through the cache's custom layers, as denoising runs it: a cast hook on the kept layer would upcast its codes
    # without the scale before CustomLinear could apply it.
    fp8_bias = fp8_layer.bias.detach().to(COMPUTE_DTYPE)
    apply_custom_layers_to_model(model)
    x = torch.randn(3, 64, dtype=COMPUTE_DTYPE)
    expected_fp8_out = torch.nn.functional.linear(x, (fp8_values * 0.5).to(COMPUTE_DTYPE), fp8_bias)
    torch.testing.assert_close(model.transformer_blocks[0].txt_mlp.net[2](x), expected_fp8_out)
    # The reservation lands before the fold and before the split, whichever runs.
    assert log == ([True, True] if mode == "fold" else [True])
    assert entries_at_storage_cast == ([0] if mode == "fp8_storage" else [])

    # Counted by hand: the packed layers as stored; their biases, the fp8 layer's bias and the dense input projection at
    # bf16; the fp8 weight at bf16 when folded and one byte per element when kept. The side-channel scales, recovered
    # before the reservation, are not in the state dict it sizes.
    packed = _packed_bytes(128, 64) + _packed_bytes(256, 64)
    fp8_weight = 128 * 64 * (COMPUTE_DTYPE.itemsize if mode == "fold" else 1)
    dense = (128 + 256) + 128 + (128 * 64 + 128)
    loader._ram_cache.make_room.assert_called_once_with(packed + fp8_weight + dense * COMPUTE_DTYPE.itemsize)


class _TinyQwenVL(torch.nn.Module):
    """transformers' `Qwen2_5_VLForConditionalGeneration` layout at toy width: the language model and the vision tower
    under `model.`, the LM head at the top."""

    _checkpoint_conversion_mapping: dict = {}

    def __init__(self, _config) -> None:
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(64, 128)
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(64, 128, bias=False)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([layer])
        self.model.visual = torch.nn.Module()
        self.model.visual.proj = torch.nn.Linear(8, 4)
        self.lm_head = torch.nn.Linear(64, 32, bias=False)


def test_the_encoder_keeps_marker_named_nvfp4_layers_packed_under_their_transformers_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import transformers

    torch.manual_seed(1)
    state_dict: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    for legacy, shape in (("model.layers.0.self_attn.q_proj", (128, 64)), ("model.layers.0.mlp.down_proj", (128, 64))):
        tensors, weight = _nvfp4_tensors(legacy, shape)
        state_dict.update(tensors)
        state_dict[f"{legacy}.comfy_quant"] = _marker("nvfp4")
        expected[legacy.replace("model.", "model.language_model.", 1)] = weight
    state_dict["model.layers.0.self_attn.q_proj.bias"] = torch.randn(128)
    fp8_values = torch.randint(-8, 9, (32, 64)).float()
    fp8_marker = _marker("float8_e4m3fn")
    # The fp8_scaled encoder's own scale spelling, beside the marker the nvfp4 build adds.
    state_dict["lm_head.weight"] = fp8_values.to(torch.float8_e4m3fn)
    state_dict["lm_head.scale_weight"] = torch.tensor(0.5)
    state_dict["lm_head.scale_input"] = torch.tensor(1.0)
    state_dict["lm_head.comfy_quant"] = fp8_marker
    state_dict["visual.proj.weight"] = torch.randn(4, 8)
    state_dict["visual.proj.bias"] = torch.randn(4)

    log = _patch_common(monkeypatch, state_dict, {})
    monkeypatch.setattr(transformers, "Qwen2_5_VLForConditionalGeneration", _TinyQwenVL, raising=False)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(torch_dtype=None, tie_word_embeddings=False),
    )
    checkpoint = tmp_path / "qwen_2.5_vl_7b_nvfp4.safetensors"
    checkpoint.touch()
    loader = object.__new__(QwenVLEncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    _record_fold(monkeypatch, loader, log)
    q_bias = state_dict["model.layers.0.self_attn.q_proj.bias"]
    visual_weight = state_dict["visual.proj.weight"]

    model = loader._load_text_encoder_from_singlefile(
        QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint))
    )

    for path, weight in expected.items():
        module = model.get_submodule(path)
        assert isinstance(module, NVFP4Linear), path
        assert module.weight_scale_2.dtype is torch.float32, path
        x = torch.randn(3, module.in_features, dtype=COMPUTE_DTYPE)
        bias = q_bias.to(COMPUTE_DTYPE) if module.bias is not None else None
        torch.testing.assert_close(module(x), torch.nn.functional.linear(x, weight.to(COMPUTE_DTYPE), bias))
    assert torch.equal(model.lm_head.weight, (fp8_values * 0.5).to(COMPUTE_DTYPE))
    assert torch.equal(model.model.visual.proj.weight, visual_weight.to(COMPUTE_DTYPE))
    assert log == [True]

    # Counted by hand: the packed layers as stored, the q_proj bias, the folded LM head with its two scale scalars and
    # the vision projection at bf16, and the LM head's 27-byte marker as stored.
    packed = 2 * _packed_bytes(128, 64)
    widened = 128 + (32 * 64 + 2) + (4 * 8 + 4)
    loader._ram_cache.make_room.assert_called_once_with(packed + widened * COMPUTE_DTYPE.itemsize + fp8_marker.numel())


def test_an_int8_convrot_checkpoint_is_refused_before_the_cache_is_evicted(monkeypatch, tmp_path) -> None:
    """This encoder has no int8 branch, and `int8_tensorwise` shares the fp8 key layout, so the fold
    applied the scale and skipped the inverse rotation.

    The refusal has to come before `make_room`, thirty lines further down, and before the config
    fetch between them: reserving room for a load that cannot finish evicts whatever else the user
    had resident, and the fetch reaches for the network to do it. That
    ordering is the half that went missing once already -- the guard was reported as moved ahead of
    the reservation while the call was not in the tree at all, and nothing here would have noticed.
    """
    payload = quantize_convrot(torch.randn(64, 256), group_size=64)
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": payload.codes,
        "model.layers.0.self_attn.q_proj.weight_scale": payload.scale,
        "model.layers.0.self_attn.q_proj.comfy_quant": comfy_quant_marker(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
        ),
    }
    checkpoint = tmp_path / "qwen_2.5_vl_7b_int8_convrot.safetensors"
    checkpoint.touch()
    run = prepare(ENCODER_SEAM, monkeypatch, state_dict=state_dict, metadata=None)

    with pytest.raises(ValueError, match="quantized with convrot"):
        run.load(QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint)))

    assert run.reserved == []


def test_an_nvfp4_layer_missing_its_global_scale_is_refused_before_the_cache_is_evicted(monkeypatch, tmp_path) -> None:
    """The degraded half-state, at the seam rather than at the detector.

    A packed uint8 weight with a block-scale grid and no `weight_scale_2` is the shape a guard keyed
    on `weight_scale_2` -- what the decode keys on -- lets straight through. `_find_nvfp4_layers`
    refuses it, and `test_nvfp4.py` pins that; what only a seam can answer is whether this loader
    still reaches the detector *before* it asks the cache for room, which nothing but the order of
    those two statements secures.
    """
    state_dict = {
        # 128 rows and a 16-block grid: a whole number of cuBLAS tiles, i.e. a layer a real build
        # could hold. The refusal reads only the key pairing and the uint8 dtype, so it fires either
        # way -- but a shape no build can contain is the wrong thing to assert the state against.
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(128, 128, dtype=torch.uint8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(128, 16).to(torch.float8_e4m3fn),
    }
    checkpoint = tmp_path / "qwen_2.5_vl_7b_nvfp4_half.safetensors"
    checkpoint.touch()
    run = prepare(ENCODER_SEAM, monkeypatch, state_dict=state_dict, metadata=None)

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        run.load(QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint)))

    assert run.reserved == []


def test_the_transformer_refuses_an_nvfp4_layer_missing_its_global_scale_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The same degraded half-state at the transformer seam, which had no `Seam` of its own.

    The encoder cell above covers the other Qwen-Image seam; this one is the transformer. Declared
    through the shared driver rather than the hand-rolled loader the cells above use, because
    `run.reserved` is the assertion: the recorded reservations are what say "before", where a
    `MagicMock` would only say "not called at all".
    """
    import diffusers

    state_dict = {
        "img_in.weight": torch.randn(128, 64),
        "img_in.bias": torch.randn(128),
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(128, 32, dtype=torch.uint8),
        "transformer_blocks.0.attn.to_q.weight_scale": torch.zeros(128, 4).to(torch.float8_e4m3fn),
    }
    checkpoint = tmp_path / "qwen_image_nvfp4_half.safetensors"
    checkpoint.touch()
    seam = Seam(
        loader=QwenImageCheckpointModel,
        module=qwen_image,
        entry="_load_from_singlefile",
        load_file_host=safetensors.torch,
        compute_dtype=COMPUTE_DTYPE,
        patches_device=True,
        # This entry reads neither: `_torch_dtype` is read nowhere in `qwen_image`, and the fp8
        # storage cast is called from `_load_model`. Stubbing them would hide one that started.
        sets_torch_dtype=False,
        casts_fp8_storage=False,
    )
    run = prepare(
        seam,
        monkeypatch,
        state_dict=state_dict,
        metadata=None,
        geometry=lambda patch: patch.setattr(
            diffusers, "QwenImageTransformer2DModel", _TinyQwenImageTransformer, raising=False
        ),
    )

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        run.load(Main_Checkpoint_QwenImage_Config.model_construct(path=str(checkpoint), name="qwen_image_nvfp4_half"))

    assert run.reserved == []
