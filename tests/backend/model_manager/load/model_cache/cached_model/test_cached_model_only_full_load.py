import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
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
