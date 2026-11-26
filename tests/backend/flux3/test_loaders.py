"""The FLUX.3 single-file and GGUF loaders, driven over tiny native checkpoints.

Asserted is the forward: a loaded model must predict what the model the checkpoint was written from
predicts. That covers the prefix strip, the key conversion, the output adaLN reordering and the QK norms,
which diffusers' transformer runs in float32.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux3_Config, Main_GGUF_Flux3_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import flux3 as flux3_loaders
from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3CheckpointModel, Flux3GGUFCheckpointModel
from invokeai.backend.model_manager.load.model_util import calc_tokenizer_size_by_fs
from invokeai.backend.model_manager.taxonomy import Flux3VariantType, SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from tests.backend.flux3.flux3_fixture import (
    flux3_model_from_native,
    forward,
    tiny_flux3_model,
    to_native,
    write_flux3_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

FLOW_MAP = pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "base"])


def _variant(flow_map: bool) -> Flux3VariantType:
    return Flux3VariantType.Klein if flow_map else Flux3VariantType.KleinBase


def test_diffusers_tokenizer_size_uses_text_encoder_assets(tmp_path) -> None:
    (tmp_path / "flux3.safetensors").write_bytes(b"large model payload")
    tokenizer_dir = tmp_path / "text_encoder"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_bytes(b"tokenizer assets")
    (tokenizer_dir / "config.json").write_bytes(b"not a tokenizer asset")
    loader = object.__new__(flux3_loaders.Flux3DiffusersModel)

    size = loader.get_size_fs(SimpleNamespace(), tmp_path, SubModelType.Tokenizer)

    assert size == calc_tokenizer_size_by_fs(tokenizer_dir)
    assert size == len(b"tokenizer assets")


@pytest.mark.parametrize("has_headroom", [False, True], ids=["concat-only-headroom", "enough-headroom"])
def test_gguf_adaln_conversion_is_admitted_before_decode_and_concat(monkeypatch, tmp_path, has_headroom: bool) -> None:
    native = to_native(tiny_flux3_model(flow_map=True))
    path = tmp_path / "flux3_image_klein_sd_test.gguf"
    writer = gguf.GGUFWriter(str(path), "flux")
    adaln_key = "final_layer.image.adaLN_modulation.1.weight"
    for name, tensor in native.items():
        quantization_type = gguf.GGMLQuantizationType.Q5_1 if name == adaln_key else gguf.GGMLQuantizationType.BF16
        raw = gguf.quants.quantize(tensor.detach().to(torch.float32).numpy(), quantization_type)
        writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=quantization_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    monkeypatch.setattr(flux3_loaders.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        flux3_loaders.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    output_bytes = native[adaln_key].numel() * torch.float32.itemsize
    conversion_peak_bytes = max(output_bytes * (1 + HOST_DEQUANT_PEAK_COPIES), 2 * output_bytes)
    available_bytes = 2**40 if has_headroom else 2 * output_bytes
    make_room = MagicMock(
        side_effect=lambda _cache_bytes, *, host_bytes_needed: SimpleNamespace(
            ram_headroom_satisfied=host_bytes_needed <= available_bytes
        )
    )
    loader = object.__new__(Flux3GGUFCheckpointModel)
    loader._torch_device = torch.device("cpu")
    loader._torch_dtype = torch.float32
    loader._ram_cache = SimpleNamespace(make_room=make_room)
    loader._logger = MagicMock()
    config = Main_GGUF_Flux3_Config.model_construct(path=str(path), name="flux3", variant=_variant(True))
    converted = False
    original_convert = flux3_loaders.convert_flux3_native_to_diffusers

    def observe_conversion(state_dict):
        nonlocal converted
        converted = True
        return original_convert(state_dict)

    monkeypatch.setattr(flux3_loaders, "convert_flux3_native_to_diffusers", observe_conversion)

    if not has_headroom:
        with pytest.raises(MemoryError, match="Insufficient host memory"):
            loader._load_model(config, SubModelType.Transformer)
        assert converted is False
        make_room.assert_called_once()
        assert make_room.call_args.kwargs == {"host_bytes_needed": conversion_peak_bytes}
    else:
        model = loader._load_model(config, SubModelType.Transformer)
        assert model is not None
        assert converted is True
        assert make_room.call_args_list[0].args == (0,)
        assert make_room.call_args_list[0].kwargs == {"host_bytes_needed": conversion_peak_bytes}


def test_mxfp8_decode_headroom_is_checked_before_extraction(monkeypatch) -> None:
    state_dict = to_native(tiny_flux3_model(flow_map=True))
    monkeypatch.setattr(flux3_loaders, "mxfp8_scale_decode_host_bytes", lambda _sd, _hints: 1234)
    make_room = MagicMock(return_value=SimpleNamespace(ram_headroom_satisfied=False))
    extracted = False
    original_extract = flux3_loaders.extract_fp8_scaled_layers

    def observe_extract(*args, **kwargs):
        nonlocal extracted
        extracted = True
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(flux3_loaders, "extract_fp8_scaled_layers", observe_extract)

    with pytest.raises(MemoryError, match="Insufficient host memory"):
        flux3_loaders.flux3_transformer_from_single_file(state_dict, torch.float32, reserve=make_room)

    assert extracted is False
    make_room.assert_called_once_with(0, host_bytes_needed=1234)


def test_uncached_flux3_loader_accepts_the_host_decode_check(monkeypatch) -> None:
    monkeypatch.setattr(flux3_loaders, "mxfp8_scale_decode_host_bytes", lambda _sd, _hints: 1234)

    model = flux3_loaders.flux3_transformer_from_single_file(to_native(tiny_flux3_model(flow_map=True)), torch.float32)

    assert model is not None


@FLOW_MAP
@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bfl", "comfy-prefixed"])
def test_a_single_file_loads_as_the_model_it_was_written_from(
    monkeypatch, tmp_path, flow_map: bool, prefix: str
) -> None:
    reference = tiny_flux3_model(flow_map)
    native = {prefix + name: value for name, value in to_native(reference).items()}
    seam = Seam(
        loader=Flux3CheckpointModel,
        module=flux3_loaders,
        entry="_load_model",
        patches_device=True,
    )
    run = prepare(seam, monkeypatch, state_dict=native, metadata=None)
    checkpoint = tmp_path / "flux3.safetensors"
    safetensors.torch.save_file({"_fixture": torch.zeros(1, dtype=torch.int8)}, checkpoint)
    config = Main_Checkpoint_Flux3_Config.model_construct(
        path=str(checkpoint), name="flux3", variant=_variant(flow_map)
    )

    model = run.load(config, SubModelType.Transformer)

    assert model.config.flow_map is flow_map
    assert run.reserved, "the cache is asked for room before the weights are cast"
    for name, module in model.named_modules():
        if name.endswith((".norm_q", ".norm_k")):
            assert module.weight.dtype == torch.float32, name
    torch.testing.assert_close(forward(model, flow_map), forward(reference, flow_map), rtol=0, atol=0)


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
    reference = flux3_model_from_native(meant, flow_map)
    # Q8_0 blocks dequantize through their fp16 scale, so the two forwards agree to fp16 precision.
    torch.testing.assert_close(forward(model, flow_map), forward(reference, flow_map), rtol=1e-3, atol=1e-3)
