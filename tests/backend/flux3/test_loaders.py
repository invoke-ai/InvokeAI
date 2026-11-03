"""The FLUX.3 single-file and GGUF loaders, driven over tiny native checkpoints.

Asserted is the forward: a loaded model must predict what the model the checkpoint was written from
predicts. That covers the prefix strip, the key conversion, the output adaLN reordering and the QK norms,
which diffusers' transformer runs in float32.
"""

import gguf
import pytest
import torch
from diffusers import Flux3Transformer2DModel

from invokeai.backend.flux3.conversion import convert_flux3_native_to_diffusers
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux3_Config, Main_GGUF_Flux3_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import flux3 as flux3_loaders
from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3CheckpointModel, Flux3GGUFCheckpointModel
from invokeai.backend.model_manager.taxonomy import Flux3VariantType, SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.flux3.flux3_fixture import TINY_CONFIG, forward, tiny_flux3_model, to_native, write_flux3_gguf
from tests.fixtures.loader_seams import Seam, prepare

FLOW_MAP = pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "base"])


def _variant(flow_map: bool) -> Flux3VariantType:
    return Flux3VariantType.Klein if flow_map else Flux3VariantType.KleinBase


@FLOW_MAP
@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bfl", "comfy-prefixed"])
def test_a_single_file_loads_as_the_model_it_was_written_from(monkeypatch, flow_map: bool, prefix: str) -> None:
    import safetensors.torch

    reference = tiny_flux3_model(flow_map)
    native = {prefix + name: value for name, value in to_native(reference).items()}
    seam = Seam(
        loader=Flux3CheckpointModel,
        module=flux3_loaders,
        entry="_load_model",
        load_file_host=safetensors.torch,
        patches_device=True,
    )
    run = prepare(seam, monkeypatch, state_dict=native)
    config = Main_Checkpoint_Flux3_Config.model_construct(
        path="flux3.safetensors", name="flux3", variant=_variant(flow_map)
    )

    model = run.load(config, SubModelType.Transformer)

    assert model.config.flow_map is flow_map
    assert run.reserved, "the cache is asked for room before the weights are cast"
    for name, module in model.named_modules():
        if name.endswith((".norm_q", ".norm_k")):
            assert module.weight.dtype == torch.float32, name
    torch.testing.assert_close(forward(model, flow_map), forward(reference, flow_map), rtol=0, atol=0)


def test_a_quantized_side_channel_is_refused_by_the_single_file_loader(monkeypatch) -> None:
    import safetensors.torch

    native = to_native(tiny_flux3_model(flow_map=True))
    native["single_blocks.0.q_proj.weight_scale"] = torch.ones(1)
    seam = Seam(
        loader=Flux3CheckpointModel, module=flux3_loaders, entry="_load_model", load_file_host=safetensors.torch
    )
    run = prepare(seam, monkeypatch, state_dict=native)
    config = Main_Checkpoint_Flux3_Config.model_construct(
        path="x.safetensors", name="x", variant=Flux3VariantType.Klein
    )

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(config, SubModelType.Transformer)
    assert run.reserved == [], "a refused file must not reserve cache room first"


@FLOW_MAP
@pytest.mark.parametrize(
    # The `gguf` package writes no K-quants, so the packed path is driven through the block types it can write.
    "qtype",
    [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q5_1, gguf.GGMLQuantizationType.Q4_0],
    ids=lambda qtype: qtype.name,
)
def test_a_gguf_loads_and_predicts_what_its_dequantized_weights_mean(
    monkeypatch, tmp_path, flow_map: bool, qtype: gguf.GGMLQuantizationType
) -> None:
    native = to_native(tiny_flux3_model(flow_map))
    path = tmp_path / "flux3_image_klein_sd_test.gguf"
    meant = write_flux3_gguf(path, native, qtype=qtype)
    seam = Seam(loader=Flux3GGUFCheckpointModel, module=flux3_loaders, entry="_load_model", patches_device=True)
    run = prepare(seam, monkeypatch)
    config = Main_GGUF_Flux3_Config.model_construct(path=str(path), name="flux3", variant=_variant(flow_map))

    model = run.load(config, SubModelType.Transformer)
    apply_custom_layers_to_model(model)

    packed = [p for p in model.parameters() if isinstance(p, GGMLTensor)]
    assert packed, "the quantized block weights stay packed"
    # Unquantized tensors (the fixture's BF16 linears) are unpacked at load rather than decoded every forward.
    assert {p._ggml_quantization_type for p in packed} == {qtype}
    for name, module in model.named_modules():
        if name.endswith((".norm_q", ".norm_k")):
            assert not isinstance(module.weight, GGMLTensor) and module.weight.dtype == torch.float32, name
    reference = Flux3Transformer2DModel(**TINY_CONFIG, flow_map=flow_map).eval()
    reference.load_state_dict(convert_flux3_native_to_diffusers(dict(meant)))
    # Q8_0 blocks dequantize through their fp16 scale, so the two forwards agree to fp16 precision.
    torch.testing.assert_close(forward(model, flow_map), forward(reference, flow_map), rtol=1e-3, atol=1e-3)
