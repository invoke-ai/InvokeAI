import itertools

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from tests.backend.model_manager.load.model_cache.cached_model.utils import (
    DummyModule,
    parameterize_keep_ram_copy,
    parameterize_mps_and_cuda,
)


@pytest.fixture
def model():
    model = DummyModule()
    apply_custom_layers_to_model(model)
    return model


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_total_bytes(device: str, model: DummyModule, keep_ram_copy: bool):
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    linear1_numel = 10 * 32 + 32
    linear2_numel = 32 * 64 + 64
    buffer1_numel = 64
    # Note that the non-persistent buffer (buffer2) is not included in .total_bytes() calculation.
    assert cached_model.total_bytes() == (linear1_numel + linear2_numel + buffer1_numel) * 4


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_cur_vram_bytes(device: str, model: DummyModule, keep_ram_copy: bool):
    # Model starts in CPU memory.
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    assert cached_model.cur_vram_bytes() == 0

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.cur_vram_bytes() > 0
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()
    assert all(p.device.type == device for p in model.parameters())
    assert all(p.device.type == device for p in model.buffers())


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_load(device: str, model: DummyModule, keep_ram_copy: bool):
    # Model starts in CPU memory.
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Partially load the model into VRAM.
    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)

    # Check that the model is partially loaded into VRAM.
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()
    assert loaded_bytes == sum(
        calc_tensor_size(p)
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers())
        if p.device.type == device and n != "buffer2"
    )

    # Check that the model's modules have device autocasting enabled.
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_unload(device: str, model: DummyModule, keep_ram_copy: bool):
    # Model starts in CPU memory.
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.cur_vram_bytes() == model_total_bytes

    # Partially unload the model from VRAM.
    bytes_to_free = int(model_total_bytes * 0.4)
    freed_bytes = cached_model.partial_unload_from_vram(bytes_to_free)

    # Check that the model is partially unloaded from VRAM.
    assert freed_bytes >= bytes_to_free
    assert freed_bytes < model_total_bytes
    assert freed_bytes == model_total_bytes - cached_model.cur_vram_bytes()
    assert freed_bytes == sum(
        calc_tensor_size(p) for p in itertools.chain(model.parameters(), model.buffers()) if p.device.type == "cpu"
    )

    # Check that the model's modules still have device autocasting enabled.
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_unload_keep_required_weights_in_vram(
    device: str, model: DummyModule, keep_ram_copy: bool
):
    # Model starts in CPU memory.
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.cur_vram_bytes() == model_total_bytes

    # Partially unload the model from VRAM, but request the required weights to be kept in VRAM.
    bytes_to_free = int(model_total_bytes)
    freed_bytes = cached_model.partial_unload_from_vram(bytes_to_free, keep_required_weights_in_vram=True)

    # Check that the model is partially unloaded from VRAM.
    assert freed_bytes < model_total_bytes
    assert freed_bytes == model_total_bytes - cached_model.cur_vram_bytes()
    assert freed_bytes == sum(
        calc_tensor_size(p) for p in itertools.chain(model.parameters(), model.buffers()) if p.device.type == "cpu"
    )
    # The parameters should be offloaded to the CPU, because they are in Linear layers.
    assert all(p.device.type == "cpu" for p in model.parameters())
    # The buffer should still be on the device, because it is in a layer that does not support autocast.
    assert all(p.device.type == device for p in model.buffers())

    # Check that the model's modules still have device autocasting enabled.
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_load_and_unload(device: str, model: DummyModule, keep_ram_copy: bool):
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )

    # Model starts in CPU memory.
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Full load the model into VRAM.
    loaded_bytes = cached_model.full_load_to_vram()
    assert loaded_bytes > 0
    assert loaded_bytes == model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()
    assert all(p.device.type == device for p in itertools.chain(model.parameters(), model.buffers()))
    assert not model.linear1.is_device_autocasting_enabled()
    assert not model.linear2.is_device_autocasting_enabled()

    # Full unload the model from VRAM.
    unloaded_bytes = cached_model.full_unload_from_vram()

    # Check that the model is fully unloaded from VRAM.
    assert unloaded_bytes > 0
    assert unloaded_bytes == model_total_bytes
    assert cached_model.cur_vram_bytes() == 0
    # Note that the non-persistent buffer (buffer2) is not required to be unloaded from VRAM.
    assert all(
        p.device.type == "cpu"
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers())
        if n != "buffer2"
    )


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_load_from_partial(device: str, model: DummyModule, keep_ram_copy: bool):
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )

    # Model starts in CPU memory.
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Partially load the model into VRAM.
    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()

    # Full load the rest of the model into VRAM.
    loaded_bytes_2 = cached_model.full_load_to_vram()
    assert loaded_bytes_2 > 0
    assert loaded_bytes_2 < model_total_bytes
    assert loaded_bytes + loaded_bytes_2 == cached_model.cur_vram_bytes()
    assert loaded_bytes + loaded_bytes_2 == model_total_bytes
    assert all(p.device.type == device for p in itertools.chain(model.parameters(), model.buffers()))
    assert not model.linear1.is_device_autocasting_enabled()
    assert not model.linear2.is_device_autocasting_enabled()


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_unload_from_partial(device: str, model: DummyModule, keep_ram_copy: bool):
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )

    # Model starts in CPU memory.
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Partially load the model into VRAM.
    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()

    # Full unload the model from VRAM.
    unloaded_bytes = cached_model.full_unload_from_vram()
    assert unloaded_bytes > 0
    assert unloaded_bytes == loaded_bytes
    assert cached_model.cur_vram_bytes() == 0
    # Note that the non-persistent buffer (buffer2) is not required to be unloaded from VRAM.
    assert all(
        p.device.type == "cpu"
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers())
        if n != "buffer2"
    )


@parameterize_mps_and_cuda
def test_cached_model_get_cpu_state_dict(device: str, model: DummyModule):
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device(device), keep_ram_copy=True)

    # Model starts in CPU memory.
    assert cached_model.cur_vram_bytes() == 0

    # The CPU state dict can be accessed and has the expected properties.
    cpu_state_dict = cached_model.get_cpu_state_dict()
    assert cpu_state_dict is not None
    assert len(cpu_state_dict) == len(model.state_dict())
    assert all(p.device.type == "cpu" for p in cpu_state_dict.values())

    # Full load the model into VRAM.
    cached_model.full_load_to_vram()
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()

    # The CPU state dict is still available, and still on the CPU.
    cpu_state_dict = cached_model.get_cpu_state_dict()
    assert cpu_state_dict is not None
    assert len(cpu_state_dict) == len(model.state_dict())
    assert all(p.device.type == "cpu" for p in cpu_state_dict.values())


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_full_load_and_inference(device: str, model: DummyModule, keep_ram_copy: bool):
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    # Model starts in CPU memory.
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Run inference on the CPU.
    x = torch.randn(1, 10)
    output1 = model(x)
    assert output1.device.type == "cpu"

    # Full load the model into VRAM.
    loaded_bytes = cached_model.full_load_to_vram()
    assert loaded_bytes > 0
    assert loaded_bytes == model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()
    assert all(p.device.type == device for p in itertools.chain(model.parameters(), model.buffers()))

    # Run inference on the GPU.
    output2 = model(x.to(device))
    assert output2.device.type == device

    # The outputs should be the same for both runs.
    assert torch.allclose(output1, output2.to("cpu"))


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_load_and_inference(device: str, model: DummyModule, keep_ram_copy: bool):
    # Model starts in CPU memory.
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Run inference on the CPU.
    x = torch.randn(1, 10)
    output1 = model(x)
    assert output1.device.type == "cpu"

    # Partially load the model into VRAM.
    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)

    # Check that the model is partially loaded into VRAM.
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()
    assert loaded_bytes == sum(
        calc_tensor_size(p)
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers())
        if p.device.type == device and n != "buffer2"
    )
    # Check that the model's modules have device autocasting enabled.
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()

    # Run inference on the GPU.
    output2 = model(x.to(device))
    assert output2.device.type == device

    # The output should be the same as the output from the CPU.
    assert torch.allclose(output1, output2.to("cpu"))
