import itertools
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
    HostMemoryAdmissionError,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from tests.backend.model_manager.load.model_cache.cached_model.utils import (
    DummyModule,
    TiedRequiredParameterModule,
    TiedWeightsModule,
    parameterize_keep_ram_copy,
    parameterize_mps_and_cuda,
)


class _NF4StateDictModule(torch.nn.Module, CustomModuleMixin):
    """CPU-testable stand-in for bitsandbytes' packed weight and serialized quantization sidecars."""

    def __init__(self, regenerate_quant_state_on_cpu: bool = False):
        super().__init__()
        CustomModuleMixin.__init__(self)
        self.regenerate_quant_state_on_cpu = regenerate_quant_state_on_cpu
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.uint8).reshape(4, 4), requires_grad=False)
        self.absmax = torch.ones(4, dtype=torch.uint8)
        self.quant_map = torch.ones(16)
        self.nested_absmax = torch.ones(1)
        self.nested_quant_map = torch.ones(8)
        self.quant_state = torch.ones(12, dtype=torch.uint8)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = self.weight if keep_vars else self.weight.detach()
        for suffix, tensor in (
            ("weight.absmax", self.absmax),
            ("weight.quant_map", self.quant_map),
            ("weight.nested_absmax", self.nested_absmax),
            ("weight.nested_quant_map", self.nested_quant_map),
            ("weight.quant_state.bitsandbytes__nf4", self.quant_state),
        ):
            if (
                self.regenerate_quant_state_on_cpu
                and suffix == "weight.quant_state.bitsandbytes__nf4"
                and self.weight.device.type != "cpu"
            ):
                tensor = torch.ones_like(tensor, device="cpu")
            destination[prefix + suffix] = tensor if keep_vars else tensor.detach()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.weight = torch.nn.Parameter(state_dict.pop(prefix + "weight"), requires_grad=False)
        self.absmax = state_dict.pop(prefix + "weight.absmax")
        self.quant_map = state_dict.pop(prefix + "weight.quant_map")
        self.nested_absmax = state_dict.pop(prefix + "weight.nested_absmax")
        self.nested_quant_map = state_dict.pop(prefix + "weight.nested_quant_map")
        self.quant_state = state_dict.pop(prefix + "weight.quant_state.bitsandbytes__nf4")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight.float())


def _nf4_cached_model(
    keep_ram_copy: bool, regenerate_quant_state_on_cpu: bool = False
) -> tuple[_NF4StateDictModule, CachedModelWithPartialLoad]:
    model = torch.nn.Module()
    model.linear = _NF4StateDictModule(regenerate_quant_state_on_cpu)
    cached_model = CachedModelWithPartialLoad(model, torch.device("meta"), keep_ram_copy=keep_ram_copy)
    return model.linear, cached_model


@pytest.mark.parametrize("keep_ram_copy", [True, False])
def test_nf4_partial_load_admits_packed_weight_and_sidecars_as_one_group(keep_ram_copy: bool) -> None:
    layer, cached_model = _nf4_cached_model(keep_ram_copy)
    packed_bytes = cached_model._state_dict_bytes["linear.weight"]
    group_bytes = cached_model.total_bytes()

    assert cached_model.partial_load_to_vram(packed_bytes - 1) == 0
    assert all(tensor.device.type == "cpu" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == 0

    assert cached_model.partial_load_to_vram(packed_bytes) == 0
    assert all(tensor.device.type == "cpu" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == 0

    assert cached_model.partial_load_to_vram(group_bytes) == group_bytes
    assert all(tensor.device.type == "meta" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == group_bytes


@pytest.mark.parametrize("keep_ram_copy", [True, False])
@pytest.mark.parametrize("host_budget", ["below-packed", "packed-only"])
def test_nf4_partial_unload_refuses_packed_weight_without_sidecars(keep_ram_copy: bool, host_budget: str) -> None:
    layer, cached_model = _nf4_cached_model(keep_ram_copy)
    packed_bytes = cached_model._state_dict_bytes["linear.weight"]
    group_bytes = cached_model.total_bytes()
    cached_model.partial_load_to_vram(group_bytes)

    with pytest.raises(HostMemoryAdmissionError):
        cached_model.partial_unload_from_vram(
            group_bytes, host_bytes_available=packed_bytes - (1 if host_budget == "below-packed" else 0)
        )

    assert all(tensor.device.type == "meta" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == group_bytes


@pytest.mark.parametrize("keep_ram_copy", [True, False])
def test_nf4_partial_unload_admits_exact_group(keep_ram_copy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    layer, cached_model = _nf4_cached_model(keep_ram_copy)
    cpu_state_dict = {key: tensor.clone() for key, tensor in layer.state_dict().items()}
    group_bytes = cached_model.total_bytes()
    assert cached_model.partial_load_to_vram(group_bytes) == group_bytes

    def restore_cpu_state_dict(*_args) -> None:
        layer.load_state_dict(dict(cpu_state_dict))

    # Meta tensors cannot materialize back to CPU; keep this CPU-CI test focused on unload selection and accounting.
    monkeypatch.setattr(cached_model, "_load_state_dict_with_device_conversion", restore_cpu_state_dict)
    assert cached_model.partial_unload_from_vram(group_bytes, host_bytes_available=group_bytes) == group_bytes
    assert all(tensor.device.type == "cpu" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == 0


@pytest.mark.parametrize("keep_ram_copy", [True, False])
def test_nf4_regenerated_cpu_sidecar_does_not_recharge_resident_weight(
    keep_ram_copy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer, cached_model = _nf4_cached_model(keep_ram_copy, regenerate_quant_state_on_cpu=True)
    cpu_state_dict = {key: tensor.clone() for key, tensor in layer.state_dict().items()}
    group_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0
    assert cached_model.partial_load_to_vram(group_bytes) == group_bytes
    resident_bytes = cached_model.cur_vram_bytes()

    # bitsandbytes builds this serialization tensor anew on CPU for each state_dict() call.
    for _ in range(2):
        state_dict = layer.state_dict()
        assert state_dict["weight"].device.type == "meta"
        assert state_dict["weight.quant_state.bitsandbytes__nf4"].device.type == "cpu"
        layer.absmax = torch.ones(4, dtype=torch.uint8)
        assert not cached_model._group_is_on_device(
            cached_model.model.state_dict(), "linear.weight", torch.device("meta")
        )
        layer.absmax = torch.ones(4, dtype=torch.uint8, device="meta")
        assert cached_model.partial_load_to_vram(group_bytes) == 0
        assert cached_model.repair_required_tensors_on_compute_device() == 0
        assert cached_model.cur_vram_bytes() == resident_bytes

    if not keep_ram_copy:
        # Meta tensors cannot materialize back to CPU; exercise public unload selection and accounting only.
        monkeypatch.setattr(
            cached_model,
            "_load_state_dict_with_device_conversion",
            lambda *_args: layer.load_state_dict(dict(cpu_state_dict)),
        )
    assert cached_model.partial_unload_from_vram(group_bytes) == resident_bytes
    assert all(tensor.device.type == "cpu" for tensor in layer.state_dict().values())
    assert cached_model.cur_vram_bytes() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and bitsandbytes NF4 kernels")
@pytest.mark.parametrize("keep_ram_copy", [True, False])
def test_nf4_quant_state_sidecars_move_with_weight(keep_ram_copy: bool):
    """A no-shadow partial transfer must keep the NF4 weight and all quantization metadata together."""
    from invokeai.backend.quantization.bnb_nf4 import InvokeLinearNF4

    torch.manual_seed(0)
    source = torch.nn.Linear(64, 4)
    layer = InvokeLinearNF4(64, 4)
    layer.load_state_dict(source.state_dict())
    layer.to("cuda")
    input = torch.randn(2, 64, device="cuda")
    expected = layer(input)

    # Start from a pre-quantized CPU layer, as the model loader does.
    cpu_state_dict = {key: value.to("cpu") for key, value in layer.state_dict().items()}
    layer.load_state_dict(cpu_state_dict)
    model = torch.nn.Module()
    model.linear = layer

    cached_model = CachedModelWithPartialLoad(model, torch.device("cuda"), keep_ram_copy=keep_ram_copy)
    assert cached_model.cur_vram_bytes() == 0
    assert cached_model.partial_load_to_vram(cached_model.total_bytes()) == cached_model.total_bytes()

    assert model.linear.weight.device.type == "cuda"
    assert model.linear.weight.quant_state is not None
    assert model.linear.weight.quant_state.absmax.device.type == "cuda"
    assert model.linear.weight.quant_state.code.device.type == "cuda"
    assert torch.allclose(model.linear(input), expected, atol=1e-5)

    resident_bytes = cached_model.cur_vram_bytes()
    for _ in range(2):
        state_dict = model.linear.state_dict()
        assert state_dict["weight"].device.type == "cuda"
        assert state_dict["weight.quant_state.bitsandbytes__nf4"].device.type == "cpu"
        assert cached_model.partial_load_to_vram(cached_model.total_bytes()) == 0
        assert cached_model.repair_required_tensors_on_compute_device() == 0
        assert cached_model.cur_vram_bytes() == resident_bytes
        assert torch.allclose(model.linear(input), expected, atol=1e-5)

    assert (
        cached_model.partial_unload_from_vram(
            cached_model.total_bytes(), host_bytes_available=cached_model.total_bytes()
        )
        == resident_bytes
    )
    assert model.linear.weight.device.type == "cpu"
    assert model.linear.weight.quant_state.absmax.device.type == "cpu"
    assert model.linear.weight.quant_state.code.device.type == "cpu"
    assert cached_model.cur_vram_bytes() == 0

    cached_model.full_load_to_vram()
    assert model.linear.weight.quant_state.absmax.device.type == "cuda"
    assert model.linear.weight.quant_state.code.device.type == "cuda"
    assert torch.allclose(model.linear(input), expected, atol=1e-5)


def test_sidecars_group_under_both_paths_of_shared_child():
    shared = torch.nn.Linear(2, 2, bias=False)
    model = torch.nn.Module()
    model.a = shared
    model.b = shared
    cached_model = CachedModelWithPartialLoad(model, torch.device("cpu"), keep_ram_copy=False)

    state_dict = {
        "a.weight": shared.weight,
        "a.weight.absmax": torch.ones(1),
        "b.weight": shared.weight,
        "b.weight.absmax": torch.ones(1),
    }

    grouped = cached_model._group_state_dict_keys_by_module_prefix(state_dict)

    assert grouped["a"] == ["a.weight", "a.weight.absmax"]
    assert grouped["b"] == ["b.weight", "b.weight.absmax"]


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


@parameterize_keep_ram_copy
def test_a_tied_weight_is_counted_once(keep_ram_copy: bool):
    """The state dict lists a tied weight under every name it has, so summing it charges the same memory twice.

    The cache decides what to evict from that total: a Qwen3 4B encoder looked 0.7 GiB larger than it is, an 8B
    encoder 1.2 GiB.
    """
    model = TiedWeightsModule()

    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device("cpu"), keep_ram_copy=keep_ram_copy
    )

    assert cached_model.total_bytes() == calc_tensor_size(model.embed.weight) + calc_tensor_size(
        model.linear.weight
    ) + calc_tensor_size(model.linear.bias)


@parameterize_keep_ram_copy
def test_a_budget_too_small_for_a_tied_group_leaves_all_of_it_behind(keep_ram_copy: bool):
    """The bytes of a tied group are charged to one of its names. Letting the other name — charged 0, so it fits any
    budget — pull the group in would move the memory past the budget the caller computed from free VRAM, and leave
    the charged name behind on the CPU.

    Runs against the meta device so it covers machines without an accelerator; the tensors it moves are real.
    """
    model = TiedWeightsModule()
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device("meta"), keep_ram_copy=keep_ram_copy
    )
    # Room for the untied linear, one byte short of the tied matrix.
    budget = calc_tensor_size(model.linear.weight) + calc_tensor_size(model.linear.bias)

    loaded_bytes = cached_model.partial_load_to_vram(budget)

    assert loaded_bytes <= budget
    assert not model.embed.weight.is_meta
    assert not model.head.weight.is_meta
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_a_tied_group_with_a_required_member_is_kept_on_the_device():
    """`keep_required_weights_in_vram` protects the weights a partially-loaded model cannot run without. Offloading
    such a weight through a tied name that is not itself required would leave the next forward with its tensors on
    two devices."""
    model = TiedRequiredParameterModule()
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    cached_model.full_load_to_vram()

    freed_bytes = cached_model.partial_unload_from_vram(cached_model.total_bytes(), keep_required_weights_in_vram=True)

    assert freed_bytes == 0
    assert model.required.is_meta
    assert model.linear.weight.is_meta


def test_partial_unload_checks_selected_host_bytes_before_transfer():
    """Host admission prices the selected tensors, not the full resident partial model."""
    with torch.device("meta"):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    cached_model.full_load_to_vram()
    selected_bytes = calc_tensor_size(model[0].weight) + calc_tensor_size(model[0].bias)
    moved = patch.object(cached_model, "_load_state_dict_with_device_conversion", MagicMock())
    with moved as transfer:
        with pytest.raises(HostMemoryAdmissionError):
            cached_model.partial_unload_from_vram(selected_bytes, host_bytes_available=selected_bytes - 1)
        transfer.assert_not_called()
        assert (
            cached_model.partial_unload_from_vram(selected_bytes, host_bytes_available=selected_bytes) == selected_bytes
        )
        transfer.assert_called_once()


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_a_tied_weight_stays_one_tensor_across_the_device(device: str, keep_ram_copy: bool):
    """Moving each name separately put a second copy on the compute device and untied the two, so the model spent
    VRAM it never reads through the second name."""
    model = TiedWeightsModule()
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        allocated_before = torch.cuda.memory_allocated()

    loaded_bytes = cached_model.full_load_to_vram()

    assert loaded_bytes == cached_model.total_bytes()
    assert model.head.weight.device.type == device
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()
    if device == "cuda":
        # The device really holds one copy, not two tensors that happen to compare equal. The bound allows the
        # allocator's block rounding but not a second embedding matrix.
        assert torch.cuda.memory_allocated() - allocated_before < cached_model.total_bytes() + calc_tensor_size(
            model.embed.weight
        )

    freed_bytes = cached_model.partial_unload_from_vram(cached_model.total_bytes())

    assert freed_bytes == loaded_bytes
    assert model.head.weight.device.type == "cpu"
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


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


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_load_chunked_full_residency(device: str, model: DummyModule, keep_ram_copy: bool):
    """Paced passes (max_bytes) must converge to exactly the state one uncapped call reaches."""
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # A cap far below the total forces several passes.
    max_bytes = model_total_bytes // 4
    passes = 0
    truncated = True
    while truncated:
        remaining_budget = model_total_bytes - cached_model.cur_vram_bytes()
        _, truncated = cached_model.partial_load_to_vram_chunk(remaining_budget, max_bytes=max_bytes)
        passes += 1
        assert passes < 100, "paced load did not converge"

    assert passes > 1, "cap did not actually split the load into multiple passes"
    assert cached_model.cur_vram_bytes() == model_total_bytes
    assert all(p.device.type == device for p in model.parameters())
    # Fully resident: the settled pass must have disabled device autocasting.
    assert not model.linear1.is_device_autocasting_enabled()
    assert not model.linear2.is_device_autocasting_enabled()

    # Inference still works after the paced load.
    x = torch.randn(1, 10).to(device)
    assert model(x).device.type == device


@parameterize_mps_and_cuda
@parameterize_keep_ram_copy
def test_cached_model_partial_load_chunked_matches_uncapped_budget(
    device: str, model: DummyModule, keep_ram_copy: bool
):
    """With a capacity budget below the total, paced passes settle at the same residency as one
    uncapped call with the same budget, and autocasting stays enabled while truncated."""
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    model_total_bytes = cached_model.total_bytes()
    target_vram_bytes = int(model_total_bytes * 0.6)

    # Reference: an identical (by architecture) model loaded with one uncapped call.
    reference_model = DummyModule()
    apply_custom_layers_to_model(reference_model)
    reference_cached = CachedModelWithPartialLoad(
        model=reference_model, compute_device=torch.device(device), keep_ram_copy=keep_ram_copy
    )
    reference_loaded = reference_cached.partial_load_to_vram(target_vram_bytes)

    passes = 0
    truncated = True
    while truncated:
        remaining_budget = target_vram_bytes - cached_model.cur_vram_bytes()
        _, truncated = cached_model.partial_load_to_vram_chunk(remaining_budget, max_bytes=1024)
        passes += 1
        if truncated:
            # Mid-stream the model must stay runnable: autocasting stays enabled.
            assert model.linear1.is_device_autocasting_enabled()
            assert model.linear2.is_device_autocasting_enabled()
        assert passes < 100, "paced load did not converge"

    assert passes > 1, "cap did not actually split the load into multiple passes"
    assert cached_model.cur_vram_bytes() == reference_loaded
    assert cached_model.cur_vram_bytes() < model_total_bytes
    # Still partially loaded, so autocasting remains enabled after settling too.
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_partial_transfer_failure_invalidates_vram_accounting(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed transfer must not leave a stale cached VRAM byte count."""
    if operation == "load":
        model = torch.nn.Linear(4, 4)
        cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
        transfer = cached_model.partial_load_to_vram
    else:
        model = torch.nn.Linear(4, 4, device="meta")
        cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
        transfer = cached_model.partial_unload_from_vram

    cached_model.cur_vram_bytes()

    def fail_transfer(*args, **kwargs):
        raise RuntimeError("simulated transfer failure")

    monkeypatch.setattr(cached_model, "_load_state_dict_with_device_conversion", fail_transfer)

    with pytest.raises(RuntimeError, match="simulated transfer failure"):
        transfer(cached_model.total_bytes())

    assert cached_model._cur_vram_bytes is None


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_transfer_failure_after_one_module_recounts_actual_residency(operation: str, monkeypatch: pytest.MonkeyPatch):
    """OOM/device errors can interrupt the module-by-module transfer after earlier modules changed devices."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    apply_custom_layers_to_model(model)
    first_bytes = sum(param.numel() * param.element_size() for param in model[0].parameters())
    second_bytes = sum(param.numel() * param.element_size() for param in model[1].parameters())
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device("meta"), keep_ram_copy=operation == "unload"
    )
    if operation == "unload":
        # Start from fully loaded state so failure must re-enable custom autocast.
        cached_model.full_load_to_vram()
        assert not model[0].is_device_autocasting_enabled()
        assert not model[1].is_device_autocasting_enabled()
        transfer = cached_model.partial_unload_from_vram
        assert cached_model.cur_vram_bytes() == first_bytes + second_bytes
        expected_devices = ("cpu", "meta")
        expected_bytes = second_bytes
    else:
        transfer = cached_model.partial_load_to_vram
        assert cached_model.cur_vram_bytes() == 0
        expected_devices = ("meta", "cpu")
        expected_bytes = first_bytes

    def fail_second_module(*args, **kwargs):
        raise RuntimeError("second module transfer failed")

    monkeypatch.setattr(model[1], "_load_from_state_dict", fail_second_module)

    with pytest.raises(RuntimeError, match="second module transfer failed"):
        transfer(first_bytes + second_bytes)

    assert tuple(module.weight.device.type for module in model) == expected_devices
    assert cached_model.cur_vram_bytes() == expected_bytes
    assert model[0].is_device_autocasting_enabled()
    assert model[1].is_device_autocasting_enabled()


def test_partial_load_buffer_failure_invalidates_accounting_and_enables_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DummyModule()
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)

    def fail_buffer_move(device: torch.device) -> None:
        raise RuntimeError("simulated buffer transfer failure")

    monkeypatch.setattr(cached_model, "_move_non_persistent_buffers_to_device", fail_buffer_move)

    with pytest.raises(RuntimeError, match="simulated buffer transfer failure"):
        cached_model.full_load_to_vram()

    assert cached_model._cur_vram_bytes is None
    assert model.linear1.is_device_autocasting_enabled()
    assert model.linear2.is_device_autocasting_enabled()


def test_partial_load_retry_after_failure_accounts_actual_residency(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    original_load = model[1]._load_from_state_dict

    def fail_second_module(*args, **kwargs):
        raise RuntimeError("second module transfer failed")

    monkeypatch.setattr(model[1], "_load_from_state_dict", fail_second_module)
    with pytest.raises(RuntimeError, match="second module transfer failed"):
        cached_model.full_load_to_vram()

    monkeypatch.setattr(model[1], "_load_from_state_dict", original_load)
    cached_model.full_load_to_vram()

    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()
