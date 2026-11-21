import logging
from typing import Callable
from unittest.mock import MagicMock

import gguf
import pytest
import torch

import invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load as full_load_module
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType
from tests.backend.model_manager.load.model_cache.cached_model.utils import (
    DummyModule,
    TiedWeightsModule,
    parameterize_keep_ram_copy,
    parameterize_mps_and_cuda,
)


class NonTorchModel:
    """A model that does not sub-class torch.nn.Module."""

    def __init__(self):
        self.linear = torch.nn.Linear(10, 32)

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HookCheckingLinear(torch.nn.Linear):
    """Child hook that requires a sibling key omitted by partial alias state dicts."""

    def __init__(self):
        super().__init__(64, 128, bias=False)
        self.register_buffer("scale", torch.ones(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "scale" not in state_dict:
            raise AssertionError("partial alias state dict reached child hook")
        state_dict.pop(prefix + "scale")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class TaggedParameter(torch.nn.Parameter):
    marker = "tagged"


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_total_bytes(device: str, keep_ram_copy: bool):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    assert cached_model.total_bytes() == 100


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_is_in_vram(device: str, keep_ram_copy: bool):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    assert not cached_model.is_in_vram()
    assert cached_model.cur_vram_bytes() == 0

    cached_model.full_load_to_vram()
    assert cached_model.is_in_vram()
    assert cached_model.cur_vram_bytes() == 100

    cached_model.full_unload_from_vram()
    assert not cached_model.is_in_vram()
    assert cached_model.cur_vram_bytes() == 0


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_load_and_unload(device: str, keep_ram_copy: bool):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    assert cached_model.full_load_to_vram() == 100
    assert cached_model.is_in_vram()
    assert all(p.device.type == device for p in cached_model.model.parameters())

    assert cached_model.full_unload_from_vram() == 100
    assert not cached_model.is_in_vram()
    assert all(p.device.type == "cpu" for p in cached_model.model.parameters())


@parameterize_keep_ram_copy
def test_a_tie_broken_by_the_move_is_restored(keep_ram_copy: bool):
    """A tied weight is in the state dict under both names, and `nn.Module.to()` copies each module's parameters
    without a memo across the modules that share one: the weight arrives as two tensors, so the model holds memory it
    never reads through the second name and `total_bytes` (which counts it once) under-reports what is resident.

    The split is staged here rather than provoked with a real device, so this runs on any machine: the load has to
    put both names back on one tensor, whether it restores them from the RAM copy or re-points them afterwards.
    """
    model = TiedWeightsModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cpu"), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    model.head.weight = torch.nn.Parameter(model.embed.weight.clone())
    assert model.head.weight.data_ptr() != model.embed.weight.data_ptr()

    cached_model.full_load_to_vram()

    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to observe both transfer directions")
def test_tied_full_load_without_ram_copy_moves_one_source_copy_before_module_to(monkeypatch):
    """Moving a no-shadow model deduplicates tied tensors in both directions before normal module moves."""
    model = TiedWeightsModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cuda"), total_bytes=100, keep_ram_copy=False
    )
    move_calls = []
    original_move_shared = full_load_module.move_shared

    def track_move_shared(tensor, device, moved):
        before = len(moved)
        result = original_move_shared(tensor, device, moved)
        if len(moved) > before:
            move_calls.append(tensor)
        return result

    monkeypatch.setattr(full_load_module, "move_shared", track_move_shared)
    module_to = MagicMock(wraps=model.to)
    monkeypatch.setattr(model, "to", module_to)

    cached_model.full_load_to_vram()
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()
    assert len(move_calls) == 1
    module_to.assert_called_once()

    move_calls.clear()
    cached_model.full_unload_from_vram()
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()
    assert len(move_calls) == 1
    assert module_to.call_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to exercise both transfer directions")
def test_tied_transfer_does_not_call_child_state_dict_hooks():
    model = TiedWeightsModule()
    model.head = HookCheckingLinear()
    model.head.weight = model.embed.weight
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cuda"), total_bytes=100, keep_ram_copy=False
    )

    cached_model.full_load_to_vram()
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()
    cached_model.full_unload_from_vram()
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to exercise both transfer directions")
def test_tied_transfer_preserves_parameter_subclass():
    model = TiedWeightsModule()
    ordinary = model.embed.weight
    tagged = TaggedParameter(ordinary.detach())
    model.head.weight = tagged
    model.register_buffer("shared_buffer", ordinary.data)
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cuda"), total_bytes=100, keep_ram_copy=False
    )

    cached_model.full_load_to_vram()
    assert isinstance(model.embed.weight, torch.nn.Parameter)
    assert isinstance(model.head.weight, TaggedParameter)
    assert model.head.weight.marker == "tagged"
    assert model.embed.weight is not model.head.weight
    assert model.embed.weight.data_ptr() == model.head.weight.data_ptr()
    assert model.shared_buffer.data_ptr() == model.embed.weight.data_ptr()
    cached_model.full_unload_from_vram()
    assert isinstance(model.embed.weight, torch.nn.Parameter)
    assert isinstance(model.head.weight, TaggedParameter)
    assert model.head.weight.marker == "tagged"
    assert model.embed.weight is not model.head.weight
    assert model.embed.weight.data_ptr() == model.head.weight.data_ptr()
    assert model.shared_buffer.data_ptr() == model.embed.weight.data_ptr()


@pytest.mark.parametrize("keep_ram_copy", [False, True])
def test_tied_invoke_int8_parameter_uses_specialized_conversion(keep_ram_copy: bool):
    """An InvokeInt8Params tie must quantize through its own CUDA conversion path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise bitsandbytes conversion")
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeInt8Params, InvokeLinear8bitLt

    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(128, 64)
    model.linear = InvokeLinear8bitLt(input_features=64, output_features=128, has_fp16_weights=False)
    model.linear.weight = InvokeInt8Params(
        data=model.embed.weight.data,
        requires_grad=False,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    )
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cuda"), total_bytes=100, keep_ram_copy=keep_ram_copy
    )

    for _ in range(2):
        cached_model.full_load_to_vram()
        assert model.linear.weight.CB is not None
        assert model.linear.weight.SCB is not None
        output = model.linear(torch.randn(2, 64, device="cuda"))
        assert output.shape == (2, 128)
        cached_model.full_unload_from_vram()


@pytest.mark.parametrize("keep_ram_copy", [False, True])
def test_last_unlock_preserves_tied_invoke_int8_state_for_reload(keep_ram_copy: bool):
    """The unlock dtype restore must not replace quantized data under a live CB/SCB pair."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise bitsandbytes conversion")
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeInt8Params, InvokeLinear8bitLt

    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(128, 64)
    model.linear = InvokeLinear8bitLt(input_features=64, output_features=128, has_fp16_weights=False)
    model.linear.weight = InvokeInt8Params(
        data=model.embed.weight.data,
        requires_grad=False,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    )
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=keep_ram_copy,
        execution_device="cuda",
        storage_device="cpu",
        logger=logging.getLogger(__name__),
        shared_cpu_weights=None,
    )
    try:
        cache.put("model", model)
        record = cache.get("model")
        assert record is not None
        assert cache.lock(record, None)
        inputs = torch.randn(2, 64, device="cuda")
        expected = model.linear(inputs)
        cache.unlock(record)
        assert model.linear.weight.data.dtype == torch.int8
        assert model.linear.weight.CB is not None
        assert model.linear.weight.SCB is not None
        record.cached_model.full_unload_from_vram()
        record.cached_model.full_load_to_vram()
        actual = model.linear(inputs)
        assert torch.allclose(actual, expected, atol=1e-5)
    finally:
        cache.shutdown()


def test_restore_counted_dtypes_skips_untied_specialized_parameter():
    class SpecializedParameter(torch.nn.Parameter):
        def __new__(cls, data):
            return super().__new__(cls, data, requires_grad=False)

        def cuda(self, device=None):
            return self

    model = torch.nn.Module()
    model.untied = SpecializedParameter(torch.randn(8, 8))
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("cpu"), total_bytes=100, keep_ram_copy=False
    )
    model.untied.data = torch.ones_like(model.untied, dtype=torch.int8)
    model.untied.CB = model.untied.data
    model.untied.SCB = torch.ones(8)

    cached_model.restore_counted_dtypes()

    assert model.untied.data.dtype == torch.int8
    assert model.untied.CB is not None
    assert model.untied.SCB is not None


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_a_tied_weight_survives_a_device_round_trip(device: str, keep_ram_copy: bool):
    model = TiedWeightsModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )

    cached_model.full_load_to_vram()

    assert model.head.weight.device.type == device
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()

    cached_model.full_unload_from_vram()

    assert model.head.weight.device.type == "cpu"
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


@parameterize_mps_and_cuda
def test_cached_model_get_cpu_state_dict(device: str):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=True
    )
    assert not cached_model.is_in_vram()

    # The CPU state dict can be accessed and has the expected properties.
    cpu_state_dict = cached_model.get_cpu_state_dict()
    assert cpu_state_dict is not None
    assert len(cpu_state_dict) == len(model.state_dict())
    assert all(p.device.type == "cpu" for p in cpu_state_dict.values())

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.is_in_vram()

    # The CPU state dict is still available, and still on the CPU.
    cpu_state_dict = cached_model.get_cpu_state_dict()
    assert cpu_state_dict is not None
    assert len(cpu_state_dict) == len(model.state_dict())
    assert all(p.device.type == "cpu" for p in cpu_state_dict.values())


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_load_and_inference(device: str, keep_ram_copy: bool):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    assert not cached_model.is_in_vram()

    # Run inference on the CPU.
    x = torch.randn(1, 10)
    output1 = model(x)
    assert output1.device.type == "cpu"

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.is_in_vram()

    # Run inference on the GPU.
    output2 = model(x.to(device))
    assert output2.device.type == device

    # The outputs should be the same for both runs.
    assert torch.allclose(output1, output2.to("cpu"))


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_non_torch_model(device: str, keep_ram_copy: bool):
    model = NonTorchModel()
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    assert not cached_model.is_in_vram()

    # The model does not have a CPU state dict.
    assert cached_model.get_cpu_state_dict() is None

    # Attempting to load the model into VRAM should have no effect.
    cached_model.full_load_to_vram()
    assert not cached_model.is_in_vram()
    assert cached_model.cur_vram_bytes() == 0

    # Attempting to unload the model from VRAM should have no effect.
    cached_model.full_unload_from_vram()
    assert not cached_model.is_in_vram()
    assert cached_model.cur_vram_bytes() == 0

    # Running inference on the CPU should work.
    output1 = model.run_inference(torch.randn(1, 10))
    assert output1.device.type == "cpu"


def _gguf_q8_0_weight(weight: torch.Tensor) -> torch.nn.Parameter:
    raw = gguf.quantize(weight.numpy(), gguf.GGMLQuantizationType.Q8_0)
    return torch.nn.Parameter(
        GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, weight.shape, torch.float32),
        requires_grad=False,
    )


def _sdnq_int8_weight(weight: torch.Tensor) -> torch.nn.Parameter:
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    data = torch.round(weight / scale).to(torch.int8)
    return torch.nn.Parameter(
        SDNQTensor(data, SDNQQuantizationType.INT8_SYM, weight.shape, torch.float32, scale=scale),
        requires_grad=False,
    )


def _packed_payloads(weight: torch.Tensor) -> list[torch.Tensor]:
    """The tensors a packed weight keeps as Python attributes, beside its wrapper."""
    if isinstance(weight, SDNQTensor):
        return [weight.quantized_data, weight._scale]
    assert isinstance(weight, GGMLTensor)
    return [weight.quantized_data]


def _model_with_packed_weight(
    module_name: str, make_packed_weight: Callable[[torch.Tensor], torch.nn.Parameter]
) -> torch.nn.Module:
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 64, bias=False)
    linear.weight = make_packed_weight(linear.weight.detach())
    model = torch.nn.Module()
    model.add_module(module_name, linear)
    return model


parameterize_packed_weight = pytest.mark.parametrize(
    "make_packed_weight", [_gguf_q8_0_weight, _sdnq_int8_weight], ids=["gguf", "sdnq"]
)


@parameterize_mps_and_cuda
@parameterize_packed_weight
@parameterize_keep_ram_copy
def test_a_quantized_weight_runs_on_the_device_and_returns_to_the_cpu(
    device: str, make_packed_weight: Callable[[torch.Tensor], torch.nn.Parameter], keep_ram_copy: bool
):
    model = _model_with_packed_weight("proj_in", make_packed_weight)
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device(device), total_bytes=100, keep_ram_copy=keep_ram_copy
    )
    x = torch.randn(2, 64)
    expected = model.proj_in(x)

    cached_model.full_load_to_vram()

    assert {t.device.type for t in _packed_payloads(model.proj_in.weight)} == {device}
    assert torch.allclose(model.proj_in(x.to(device)).cpu(), expected, atol=1e-4)

    cached_model.full_unload_from_vram()

    assert {t.device.type for t in _packed_payloads(model.proj_in.weight)} == {"cpu"}


@parameterize_packed_weight
def test_a_quantized_weight_shared_by_two_layers_moves_as_one_copy(
    make_packed_weight: Callable[[torch.Tensor], torch.nn.Parameter],
):
    """Moving each layer's parameter on its own would give the shared weight two copies on the device, and the re-tie
    after the move cannot rejoin them: a packed weight has no storage of its own to recognize the alias by. The meta
    device takes the same replace-the-parameter path the load uses on every device, so this runs without a GPU.
    """
    model = _model_with_packed_weight("proj_in", make_packed_weight)
    model.add_module("proj_out", torch.nn.Linear(64, 64, bias=False))
    model.proj_out.weight = model.proj_in.weight
    cached_model = CachedModelOnlyFullLoad(
        model=model, compute_device=torch.device("meta"), total_bytes=100, keep_ram_copy=False
    )

    cached_model.full_load_to_vram()

    assert model.proj_in.weight.quantized_data.device.type == "meta"
    assert model.proj_out.weight.quantized_data is model.proj_in.weight.quantized_data
