"""Krea-2's single-file loader keeps Comfy's nvfp4 build packed.

Comfy names the nvfp4 layers of `krea2_turbo_nvfp4.safetensors` in the safetensors header, in the native key scheme
(`blocks.N.attn.wq`), and the loader renames native keys to diffusers ones. So what these tests pin is the order in
the loader: the layers leave the state dict before the conversion and the scale handling of either side-channel
branch, come back under their diffusers paths -- packed where the model has a plain Linear, decoded under its
precision-sensitive modules, dropped where a merged file bundles another model -- and the one reservation counts them
as they end up. The compute dtype is bf16, as in production: the packed global scale must not be cast with it.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config
from invokeai.backend.model_manager.load.model_loaders import krea2
from invokeai.backend.model_manager.load.model_loaders.krea2 import Krea2CheckpointModel
from invokeai.backend.model_manager.taxonomy import Krea2VariantType
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_signed_tensors

COMPUTE_DTYPE = torch.bfloat16

# Native checkpoint layer -> (diffusers module path, [out, in]). Rows are whole 128-row tiles, columns 64 wide.
NVFP4_LAYERS = {
    "blocks.0.attn.wq": ("transformer_blocks.0.attn.to_q", (128, 64)),
    "blocks.0.attn.wo": ("transformer_blocks.0.attn.to_out.0", (128, 64)),
    "blocks.0.mlp.up": ("transformer_blocks.0.ff.up", (256, 64)),
    "txtfusion.refiner_blocks.0.attn.gate": ("text_fusion.refiner_blocks.0.attn.to_gate", (128, 64)),
    # Under Krea-2's `time_embed` skip pattern: decoded, not packed.
    "tmlp.0": ("time_embed.linear_1", (128, 64)),
}
PACKED = [native for native in NVFP4_LAYERS if native != "tmlp.0"]
# A merged file's bundled encoder layer, named in the same header: not this model's, so dropped rather than refused.
BUNDLED = "text_encoders.qwen3vl.layers.0.q_proj"


class _TinyKrea2(torch.nn.Module):
    """The diffusers module names the native keys convert to, nothing else."""

    _skip_layerwise_casting_patterns = ["time_embed", "norm"]

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(64, 128, bias=False)
        block.attn.to_k = torch.nn.Linear(CONVROT_GROUP_SIZE, 128, bias=False)
        block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(64, 128, bias=False)])
        block.ff = torch.nn.Module()
        block.ff.up = torch.nn.Linear(64, 256, bias=False)
        self.transformer_blocks = torch.nn.ModuleList([block])
        refiner = torch.nn.Module()
        refiner.attn = torch.nn.Module()
        refiner.attn.to_gate = torch.nn.Linear(64, 128, bias=False)
        self.text_fusion = torch.nn.Module()
        self.text_fusion.refiner_blocks = torch.nn.ModuleList([refiner])
        self.time_embed = torch.nn.Module()
        self.time_embed.linear_1 = torch.nn.Linear(64, 128)
        self.img_in = torch.nn.Linear(8, 4)


def _nvfp4_tensors(path: str, shape: tuple[int, int]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return nvfp4_signed_tensors(path, torch.randint(0, 2, shape, dtype=torch.bool))


def _packed_bytes(rows: int, columns: int) -> int:
    return rows * columns // 2 + rows * (columns // 16) + 4


@pytest.fixture(params=["dense", "int8"])
def loaded(request, monkeypatch: pytest.MonkeyPatch, tmp_path) -> SimpleNamespace:
    """The loader run over a native nvfp4 file whose `blocks.0.attn.wk` neighbour is dense or int8 -- the two
    side-channel branches the nvfp4 layers have to pass through untouched."""
    import diffusers
    import safetensors.torch

    torch.manual_seed(0)
    state_dict: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    header_layers: dict[str, dict[str, str]] = {}
    for native, (diffusers_path, shape) in NVFP4_LAYERS.items():
        tensors, weight = _nvfp4_tensors(native, shape)
        state_dict.update(tensors)
        expected[diffusers_path] = weight
        header_layers[native] = {"format": "nvfp4"}
    for native, shape in ((BUNDLED, (128, 64)), ("last.up", (128, 64))):
        # `last.up` is the final-block projection the conversion discards; one repack quantizes it, and its global
        # scale has to go with it rather than be refused as an orphan.
        tensors, _ = _nvfp4_tensors(native, shape)
        state_dict.update(tensors)
        header_layers[native] = {"format": "nvfp4"}

    state_dict["tmlp.0.bias"] = torch.randn(128)
    state_dict["first.weight"] = torch.randn(4, 8)
    state_dict["first.bias"] = torch.randn(4)
    if request.param == "dense":
        state_dict["blocks.0.attn.wk.weight"] = torch.randn(128, CONVROT_GROUP_SIZE)
    else:
        state_dict["blocks.0.attn.wk.weight"] = torch.randint(-127, 128, (128, CONVROT_GROUP_SIZE), dtype=torch.int8)
        state_dict["blocks.0.attn.wk.weight_scale"] = torch.ones(128, 1)
        state_dict["blocks.0.attn.wk.comfy_quant"] = comfy_quant_marker({"format": "int8_tensorwise", "convrot": False})
    dense = {
        name: state_dict[name] for name in ("tmlp.0.bias", "first.weight", "first.bias", "blocks.0.attn.wk.weight")
    }

    checkpoint_path = tmp_path / "krea2_turbo_nvfp4.safetensors"
    checkpoint_path.touch()
    config = Main_Checkpoint_Krea2_Config.model_construct(
        path=str(checkpoint_path), variant=Krea2VariantType.Turbo, fp8_storage=None
    )
    loader = object.__new__(Krea2CheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()
    # Supplied because the loader reads it when it decides whether to keep fp8 weights packed:
    # a fixture built with `object.__new__` has to provide every attribute that path touches.
    loader._torch_device = torch.device("cpu")
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model

    monkeypatch.setattr(diffusers, "Krea2Transformer2DModel", _TinyKrea2, raising=False)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: state_dict)
    monkeypatch.setattr(
        krea2,
        "read_safetensors_metadata",
        lambda _path, _logger: {"_quantization_metadata": json.dumps({"layers": header_layers})},
    )
    monkeypatch.setattr(krea2, "should_keep_fp8_weights", lambda _device: False)
    monkeypatch.setattr(krea2.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(krea2.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: COMPUTE_DTYPE)

    model = loader._load_from_singlefile(config)
    return SimpleNamespace(model=model, loader=loader, expected=expected, dense=dense, neighbour=request.param)


def test_header_named_nvfp4_layers_are_packed_under_their_diffusers_paths(loaded) -> None:
    for native in PACKED:
        path, _ = NVFP4_LAYERS[native]
        module = loaded.model.get_submodule(path)
        assert isinstance(module, NVFP4Linear), path
        assert module.weight.dtype is torch.uint8, path
        # Installed after the cast, which would otherwise round the global scale to the compute dtype.
        assert module.weight_scale_2.dtype is torch.float32, path
        # A float32 activation, not the compute dtype: torch 2.7's CPU bf16 GEMM faults with an illegal
        # instruction on part of GitHub's windows runner fleet. These values are exact in either width, and the
        # decode's own dtype threading is pinned by tests/backend/quantization/test_nvfp4.py.
        x = torch.randn(3, module.in_features, dtype=torch.float32)
        torch.testing.assert_close(module(x), x @ loaded.expected[path].T)


def test_the_neighbouring_layers_load_as_their_own_branch_leaves_them(loaded) -> None:
    model = loaded.model

    time_embed = model.time_embed.linear_1
    assert type(time_embed) is torch.nn.Linear
    # Separately from the values: `torch.equal` promotes, so it alone would accept any width.
    assert time_embed.weight.dtype is COMPUTE_DTYPE
    assert torch.equal(time_embed.weight, loaded.expected["time_embed.linear_1"].to(COMPUTE_DTYPE))
    assert torch.equal(time_embed.bias, loaded.dense["tmlp.0.bias"].to(COMPUTE_DTYPE))
    assert torch.equal(model.img_in.weight, loaded.dense["first.weight"].to(COMPUTE_DTYPE))

    to_k = model.transformer_blocks[0].attn.to_k
    if loaded.neighbour == "int8":
        assert isinstance(to_k, Int8ConvrotLinear)
    else:
        assert torch.equal(to_k.weight, loaded.dense["blocks.0.attn.wk.weight"].to(COMPUTE_DTYPE))


def test_a_bundled_models_packed_layer_is_dropped(loaded) -> None:
    assert not hasattr(loaded.model, "text_encoders")


def test_one_reservation_counts_the_nvfp4_layers_as_they_are_held(loaded) -> None:
    """Counted by hand: the packed layers at their stored size, the decoded one and the dense tensors at the compute
    dtype, an int8 neighbour at one byte per code with its float32 scale. The bundled layer is not counted."""
    packed = sum(_packed_bytes(*NVFP4_LAYERS[native][1]) for native in PACKED)
    decoded = 128 * 64 * COMPUTE_DTYPE.itemsize
    dense = (128 + 4 * 8 + 4) * COMPUTE_DTYPE.itemsize
    if loaded.neighbour == "int8":
        neighbour = 128 * CONVROT_GROUP_SIZE + 128 * torch.float32.itemsize
    else:
        neighbour = 128 * CONVROT_GROUP_SIZE * COMPUTE_DTYPE.itemsize
    loaded.loader._ram_cache.make_room.assert_called_once_with(packed + decoded + dense + neighbour)
