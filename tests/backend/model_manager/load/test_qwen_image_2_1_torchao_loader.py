"""Qwen-Image-2.1's single-file loader reads unsloth's torchao FP8 build as a ComfyUI scaled-fp8 checkpoint.

The file stores each quantized Linear as `<layer>._weight_qdata` plus a `[out, 1]` scale. What these tests pin is the
end state of the module: the layer keeps its fp8 codes with a flat per-row scale when something uses fp8, and is folded
to exactly `codes * scale` when nothing does.
"""

import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage21_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import qwen_image_2_1
from invokeai.backend.model_manager.load.model_loaders.qwen_image_2_1 import QwenImage21CheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from tests.fixtures.loader_seams import Seam, prepare

FP8 = torch.float8_e4m3fn


class _TinyTransformer(torch.nn.Module):
    """The module paths of the released model that matter here, at toy width: a quantized attention Linear, the
    `norm_out.linear` that matches the `norm` skip pattern, and a dense input projection."""

    _skip_layerwise_casting_patterns = ["norm"]

    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(16, 32)
        block.img_mlp = torch.nn.Module()
        block.img_mlp.gate_layer = torch.nn.Linear(16, 32)
        block.img_mlp.proj = torch.nn.Linear(16, 32)
        self.transformer_blocks = torch.nn.ModuleList([block])
        self.norm_out = torch.nn.Module()
        self.norm_out.linear = torch.nn.Linear(16, 32)
        self.img_in = torch.nn.Linear(16, 32)


SEAM = Seam(
    loader=QwenImage21CheckpointModel,
    module=qwen_image_2_1,
    entry="_load_model",
    compute_dtype=torch.bfloat16,
    # The loader imports `load_file` inside the method, so the name it resolves is the package's.
    load_file_host=safetensors.torch,
    patches_device=True,
)

QUANTIZED = ("transformer_blocks.0.attn.to_q", "norm_out.linear")
# ComfyUI-layout fused projection, gate rows first; the loader splits it into `gate_layer` and `proj`.
FUSED = "transformer_blocks.0.img_mlp.gate_up"


def _checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The file's tensors, its codes and its per-row scales, keyed by layer path."""
    torch.manual_seed(0)
    state_dict: dict[str, torch.Tensor] = {
        # Already in the compute dtype, so the cast leaves them as they are.
        "img_in.weight": torch.randn(32, 16).to(torch.bfloat16),
        "img_in.bias": torch.randn(32).to(torch.bfloat16),
    }
    codes: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for path in QUANTIZED:
        codes[path] = (torch.randn(32, 16) * 40).to(FP8)
        scales[path] = torch.rand(32, 1) + 0.5
        state_dict[f"{path}._weight_qdata"] = codes[path]
        state_dict[f"{path}._weight_scale"] = scales[path]
        state_dict[f"{path}.bias"] = torch.randn(32)
    codes[FUSED] = (torch.randn(64, 16) * 40).to(FP8)
    scales[FUSED] = torch.rand(64, 1) + 0.5
    state_dict[f"{FUSED}._weight_qdata"] = codes[FUSED]
    state_dict[f"{FUSED}._weight_scale"] = scales[FUSED]
    state_dict[f"{FUSED}.bias"] = torch.randn(64)
    return {f"model.diffusion_model.{key}": value for key, value in state_dict.items()}, codes, scales


def _load(
    monkeypatch: pytest.MonkeyPatch, tmp_path, *, keep_fp8: bool
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """The model, and the tensors `load_file` returned for the real file the loader read."""
    state_dict, _, _ = _checkpoint()
    checkpoint = tmp_path / "Qwen-Image-2.1-FP8.safetensors"
    safetensors.torch.save_file(state_dict, checkpoint)
    loaded: dict[str, torch.Tensor] = {}
    real_load_file = safetensors.torch.load_file

    def recording_load_file(path):
        tensors = real_load_file(path)
        loaded.update(tensors)
        return tensors

    monkeypatch.setattr(safetensors.torch, "load_file", recording_load_file)
    config = Main_Checkpoint_QwenImage21_Config.model_construct(
        path=str(checkpoint), name="qwen21", default_settings=MainModelDefaultSettings(fp8_storage=False)
    )

    def install(patch: pytest.MonkeyPatch) -> None:
        patch.setattr(load_default, "should_keep_fp8_weights", lambda _device: keep_fp8)
        patch.setattr(qwen_image_2_1, "_build_transformer", lambda _sd: _TinyTransformer())

    run = prepare(SEAM, monkeypatch, metadata=None, geometry=install)
    return run.load(config, SubModelType.Transformer), loaded


def test_the_fp8_layers_stay_quantized_with_a_flat_per_row_scale(monkeypatch, tmp_path) -> None:
    model, _ = _load(monkeypatch, tmp_path, keep_fp8=True)
    _, codes, scales = _checkpoint()

    layer = model.transformer_blocks[0].attn.to_q
    assert layer.weight.dtype is FP8
    assert torch.equal(layer.weight.view(torch.uint8), codes["transformer_blocks.0.attn.to_q"].view(torch.uint8))
    assert layer.weight_scale.shape == (32,)
    assert torch.equal(layer.weight_scale, scales["transformer_blocks.0.attn.to_q"].reshape(-1))
    # The fused projection is split after the translation: each half keeps its rows' codes and scales.
    mlp = model.transformer_blocks[0].img_mlp
    for half, rows in ((mlp.gate_layer, slice(0, 32)), (mlp.proj, slice(32, 64))):
        assert half.weight.dtype is FP8
        assert torch.equal(half.weight.view(torch.uint8), codes[FUSED][rows].view(torch.uint8))
        assert half.weight_scale.shape == (32,)
        assert torch.equal(half.weight_scale, scales[FUSED][rows].reshape(-1))
    # `norm_out.linear` matches the `norm` skip pattern, so it is folded with its scale rather than kept.
    assert model.norm_out.linear.weight.dtype is torch.bfloat16
    assert model.img_in.weight.dtype is torch.bfloat16


@pytest.mark.parametrize("keep_fp8", [True, False])
def test_a_folded_layer_is_exactly_the_codes_times_the_scale(monkeypatch, tmp_path, keep_fp8: bool) -> None:
    model, _ = _load(monkeypatch, tmp_path, keep_fp8=keep_fp8)
    _, codes, scales = _checkpoint()

    folded = codes["norm_out.linear"].float() * scales["norm_out.linear"]
    assert torch.equal(model.norm_out.linear.weight, folded.to(torch.bfloat16))
    if not keep_fp8:
        assert model.transformer_blocks[0].attn.to_q.weight.dtype is torch.bfloat16


PREFIX = "model.diffusion_model."


def test_a_folded_load_does_not_leave_parameters_pinning_the_file_mapping(monkeypatch, tmp_path) -> None:
    """`img_in` is stored in the compute dtype, so neither the fold nor the cast replaces it; assigned as a parameter it
    would be a slice of the file's mapping, which the cache's size calculation cannot see."""
    model, loaded = _load(monkeypatch, tmp_path, keep_fp8=False)

    for name in ("img_in.weight", "img_in.bias"):
        parameter = model.get_parameter(name)
        assert parameter.data_ptr() != loaded[PREFIX + name].data_ptr(), name
        assert torch.equal(parameter, loaded[PREFIX + name])


def test_a_kept_fp8_load_does_not_copy_the_codes(monkeypatch, tmp_path) -> None:
    model, loaded = _load(monkeypatch, tmp_path, keep_fp8=True)

    codes = model.transformer_blocks[0].attn.to_q.weight
    assert codes.data_ptr() == loaded[PREFIX + "transformer_blocks.0.attn.to_q._weight_qdata"].data_ptr()
