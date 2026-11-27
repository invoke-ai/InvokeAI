"""int8 compute: W8A8 kernels behind `Int8ConvrotLinear` (`backend.quantization.int8_matmul`).

With `int8_compute` off, on a device without the kernels, or for a layer they do not cover, `Int8ConvrotLinear` must
produce exactly what the dequantized path produces. Where the kernels run, they must match an independent W8A8
reference, stay close to the dense weight and within the dequantized path's memory, serve the model cache's wrapper
and LoRA sidecars, and fall back -- never fail -- when they break.

The CPU tests run everywhere: a fake kernels module computing in float32 stands in for the Triton kernels, so they
establish the policy and the routing, not how the kernels compute. `TestOnNvidiaGpu` does, on an NVIDIA GPU (compute
capability 8.0+) with Triton, which CI does not have. It is marked `slow`, so run it with `-m slow`: its first run
JIT-compiles and autotunes the kernels for each test shape (about a minute); later runs hit Triton's on-disk cache.
"""

import logging
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import invokeai.backend.quantization.int8_matmul as int8_matmul
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.int8_convrot import (
    Int8ConvrotLinear,
    build_regular_hadamard,
    dequantize_convrot_weight,
    swap_in_int8_linears,
)
from invokeai.backend.util.triton_kernels import MAX_CACHE_FAILURES, TritonKernelHealth

GROUP = 256
CPU = torch.device("cpu")


def _quantize(weight: torch.Tensor, convrot: bool, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel int8 the way Comfy's convrot builds store it: rotate, then quantize."""
    if convrot:
        out_features, in_features = weight.shape
        rotation = build_regular_hadamard(group)
        weight = (weight.view(out_features, in_features // group, group) @ rotation).view(out_features, in_features)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (weight / scale).round().clamp(-127, 127).to(torch.int8), scale


def _layer(
    out_features: int = 384, in_features: int = 512, convrot: bool = True, bias: bool = True, group: int = GROUP
) -> Int8ConvrotLinear:
    weight = torch.randn(out_features, in_features) * 0.05
    q, scale = _quantize(weight, convrot, group)
    return Int8ConvrotLinear(
        q, scale, convrot=convrot, bias=torch.randn(out_features) if bias else None, group_size=group
    )


def _dequantized(layer: Int8ConvrotLinear, x: torch.Tensor) -> torch.Tensor:
    """The dequantized path's output."""
    return DequantizingLinear.forward(layer, x)


@pytest.fixture
def int8_compute(monkeypatch):
    """int8 compute switched on, with fresh per-device state."""
    monkeypatch.setattr(int8_matmul, "_health", TritonKernelHealth())
    monkeypatch.setattr(int8_matmul, "_available", {})
    int8_matmul.set_int8_compute_enabled(True)
    yield
    int8_matmul.set_int8_compute_enabled(None)


# Added to every output the fake kernels write: small enough to pass the first-call comparison, and a result the
# dequantized fallback cannot produce, so a test can tell which path an output came from.
MARK = 0.01


class FakeKernels:
    """Stands in for `int8_matmul_kernels` on the CPU: records each call and writes the dequantized result plus `MARK`
    into the output it is handed -- or raises `error`, or writes `result(rows)` when a test sets one."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.error: Exception | None = None
        self.result = None

    def w8a8_rows(self, rows, weight, weight_scale, bias, out, m_bucket) -> None:
        self.calls.append(rows.shape[0])
        if self.error is not None:
            raise self.error
        if self.result is not None:
            out.copy_(self.result(rows))
            return
        dense = dequantize_convrot_weight(weight, weight_scale, True, rows.dtype)
        out.copy_(F.linear(rows, dense, bias.to(rows.dtype) if bias is not None else None) + MARK)


@pytest.fixture
def fake_kernels(monkeypatch, int8_compute, caplog) -> FakeKernels:
    """The CPU, in float32 (no CPU bf16 GEMM), treated as a device with the kernels."""
    kernels = FakeKernels()
    monkeypatch.setattr(int8_matmul, "_load_kernels", lambda: kernels)
    monkeypatch.setattr(int8_matmul, "_HALF_PRECISION", (torch.float32,))
    monkeypatch.setattr(int8_matmul, "_available", {CPU: True})
    caplog.set_level(logging.WARNING, logger=int8_matmul.__name__)
    return kernels


def _served_by_the_kernels(caplog) -> bool:
    """The kernel path completed: nothing was retired and nothing warned."""
    warnings = [r for r in caplog.records if r.name == int8_matmul.__name__ and r.levelno >= logging.WARNING]
    return not int8_matmul._health.is_retired(CPU) and not warnings


class CountingFallback:
    def __init__(self, layer: Int8ConvrotLinear) -> None:
        self.layer = layer
        self.rows: list[torch.Tensor] = []
        self.error: Exception | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.rows.append(x)
        if self.error is not None:
            raise self.error
        return _dequantized(self.layer, x)


def _int8_linear(layer: Int8ConvrotLinear, x: torch.Tensor, fallback) -> torch.Tensor:
    return int8_matmul.int8_linear(x, layer.weight, layer.weight_scale, layer.bias, fallback=fallback)


@pytest.mark.parametrize("enabled", [False, True], ids=["setting-off", "setting-on"])
def test_the_setting_decides_on_a_device_with_the_kernels(fake_kernels, enabled):
    torch.manual_seed(0)
    int8_matmul.set_int8_compute_enabled(enabled)
    layer = _layer()
    x = torch.randn(2, 5, 512)

    out = layer(x)

    assert int8_matmul.can_use_int8_matmul(x, 512) is enabled
    assert fake_kernels.calls == ([10] if enabled else [])
    if enabled:
        torch.testing.assert_close(out, _dequantized(layer, x) + MARK)
    else:
        assert torch.equal(out, _dequantized(layer, x))


class TestWhereTheKernelsExist:
    @pytest.mark.parametrize(
        "hip, capability, kernels, available, level",
        [
            (None, (8, 9), SimpleNamespace(), True, None),
            ("6.4", (9, 4), SimpleNamespace(), False, logging.INFO),
            (None, (7, 5), SimpleNamespace(), False, logging.INFO),
            (None, (8, 9), "it needs Triton, which on Windows is the `triton-windows` package", False, logging.WARNING),
        ],
        ids=["nvidia", "rocm", "turing", "no-triton"],
    )
    def test_each_device_is_decided_and_explained_once(
        self, monkeypatch, int8_compute, caplog, hip, capability, kernels, available, level
    ):
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
        monkeypatch.setattr(int8_matmul, "_load_kernels", lambda: kernels)
        caplog.set_level(logging.INFO, logger=int8_matmul.__name__)
        device = torch.device("cuda", 0)

        assert [int8_matmul._kernels_available(device) for _ in range(2)] == [available, available]
        assert [r.levelno for r in caplog.records] == ([] if level is None else [level])
        if level == logging.WARNING:
            assert "triton-windows" in caplog.records[0].message

    def test_other_device_types_fall_back(self, int8_compute):
        assert not int8_matmul._kernels_available(CPU)


class TestTheGate:
    """Branches of `can_use_int8_matmul` and the layer's own conditions, on CPU tensor metadata (no matmul runs)."""

    @pytest.fixture
    def cpu_with_kernels(self, monkeypatch, int8_compute):
        monkeypatch.setattr(int8_matmul, "_available", {CPU: True})

    def test_a_half_precision_input_of_the_right_width_qualifies(self, cpu_with_kernels):
        assert int8_matmul.can_use_int8_matmul(torch.randn(3, 512, dtype=torch.bfloat16), 512)

    @pytest.mark.parametrize("case", ["autocast", "requires-grad", "wrong-width", "strided-columns", "empty"])
    def test_inputs_the_kernels_cannot_take_fall_back(self, cpu_with_kernels, case):
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        if case == "requires-grad":
            x.requires_grad_(True)
        elif case == "wrong-width":
            x = torch.randn(3, 640, dtype=torch.bfloat16)
        elif case == "strided-columns":
            x = torch.randn(512, 3, dtype=torch.bfloat16).t()
        elif case == "empty":
            x = x[:0]

        if case == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert not int8_matmul.can_use_int8_matmul(x, 512)
        else:
            assert not int8_matmul.can_use_int8_matmul(x, 512)

    @pytest.mark.parametrize(
        "case, eligible",
        [("rotated", True), ("unrotated", False), ("group-64", False), ("full-precision", False)],
    )
    def test_only_rotated_layers_in_groups_of_256_take_the_kernels(self, cpu_with_kernels, case, eligible):
        layer = _layer(convrot=case != "unrotated", group=64 if case == "group-64" else GROUP)
        layer.full_precision_matmul = case == "full-precision"

        assert layer._can_use_int8_matmul(torch.randn(3, 512, dtype=torch.bfloat16)) is eligible


class TestTheFailurePolicy:
    def test_the_kernels_result_is_returned_in_the_inputs_shape(self, fake_kernels, caplog):
        torch.manual_seed(1)
        layer = _layer()
        x = torch.randn(2, 7, 512)

        out = layer(x)

        assert fake_kernels.calls == [14] and _served_by_the_kernels(caplog)
        torch.testing.assert_close(out, _dequantized(layer, x) + MARK)

    def test_rows_are_processed_in_chunks_of_twice_the_output_width(self, fake_kernels, caplog):
        """Bounds the int8 activation copy by the dequantized path's weight-sized transient."""
        torch.manual_seed(2)
        layer = _layer(out_features=16)
        x = torch.randn(77, 512)

        out = layer(x)

        assert fake_kernels.calls == [32, 32, 13] and _served_by_the_kernels(caplog)
        torch.testing.assert_close(out, _dequantized(layer, x) + MARK)

    @pytest.mark.parametrize("reference", ["nan", "zero"])
    def test_a_reference_that_says_nothing_leaves_the_key_for_a_later_call(self, fake_kernels, caplog, reference):
        """A non-finite reference (NaN input, fp16 overflow) or an all-zero one cannot tell a broken kernel from a
        working one: neither retire the device nor trust the key yet."""
        torch.manual_seed(5)
        layer = _layer(bias=reference == "nan")
        fallback = CountingFallback(layer)
        unclear = torch.randn(3, 512) if reference == "nan" else torch.zeros(3, 512)
        if reference == "nan":
            unclear[1, 7] = float("nan")
        clear = torch.randn(3, 512)

        validations, outputs = [], []
        for x in (unclear, clear, clear):
            before = len(fallback.rows)
            outputs.append(_int8_linear(layer, x, fallback))
            validations.append(len(fallback.rows) - before)

        assert validations == [1, 1, 0], "the clear call validated the key"
        assert _served_by_the_kernels(caplog)
        if reference == "nan":
            assert outputs[0][1].isnan().all()
        else:
            assert torch.equal(outputs[0], torch.full_like(outputs[0], MARK)), "the kernels' output stands"

    def test_each_autotune_key_is_validated_once_on_its_last_rows(self, fake_kernels):
        torch.manual_seed(3)
        with_bias, without_bias = _layer(), _layer(bias=False)
        fallbacks = {layer: CountingFallback(layer) for layer in (with_bias, without_bias)}
        x = torch.randn(300, 512)

        validations = []
        for layer, rows in [(with_bias, x)] * 3 + [(with_bias, x[:3]), (without_bias, x), (without_bias, x)]:
            before = len(fallbacks[layer].rows)
            _int8_linear(layer, rows, fallbacks[layer])
            validations.append(len(fallbacks[layer].rows) - before)

        assert validations == [1, 0, 0, 1, 1, 0], "once per row bucket and bias"
        assert torch.equal(fallbacks[with_bias].rows[0], x[-int8_matmul._FIRST_CALL_ROWS :])

    def test_running_out_of_memory_propagates_without_retiring(self, fake_kernels):
        """By the memory bound, the dequantized fallback would not fit either."""
        fake_kernels.error = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.50 GiB")
        layer = _layer()

        with pytest.raises(torch.OutOfMemoryError):
            layer(torch.randn(3, 512))
        assert not int8_matmul._health.is_retired(CPU)

    def test_triton_cache_failures_fall_back_and_only_a_streak_retires(self, fake_kernels, caplog):
        caplog.set_level(logging.WARNING, logger=int8_matmul.__name__)
        layer = _layer()
        x = torch.randn(3, 512)
        fake_kernels.error = PermissionError("[WinError 5] Access is denied")

        outputs = [layer(x) for _ in range(MAX_CACHE_FAILURES - 1)]
        assert not caplog.records and not int8_matmul._health.is_retired(CPU)

        outputs.append(layer(x))
        layer(x)

        assert all(torch.equal(out, _dequantized(layer, x)) for out in outputs)
        assert len(fake_kernels.calls) == MAX_CACHE_FAILURES, "the retired device was not tried again"
        assert len(caplog.records) == 1

    @pytest.mark.parametrize("broken", ["raises", "wrong-first-result", "nan-first-result", "inf-first-result"])
    def test_a_broken_kernel_retires_the_device_with_one_warning(self, fake_kernels, caplog, broken):
        """The reference is finite in every case; a non-finite kernel output is a broken kernel, not an unclear one."""
        caplog.set_level(logging.WARNING, logger=int8_matmul.__name__)
        fill = {"wrong-first-result": 0.0, "nan-first-result": float("nan"), "inf-first-result": float("inf")}
        if broken == "raises":
            fake_kernels.error = RuntimeError("Triton Error [CUDA]: invalid argument")
        else:
            fake_kernels.result = lambda rows: torch.full((rows.shape[0], 384), fill[broken])
        layer = _layer()
        x = torch.randn(3, 512)

        outputs = [layer(x), layer(x)]

        assert all(torch.equal(out, _dequantized(layer, x)) for out in outputs)
        assert len(fake_kernels.calls) == 1
        assert len(caplog.records) == 1 and "384x512" in caplog.records[0].message

    def test_a_failing_dequantized_path_is_not_blamed_on_int8_compute(self, fake_kernels, monkeypatch):
        events = []
        monkeypatch.setattr(int8_matmul, "_synchronize", lambda device: events.append("synchronize"))
        layer = _layer()
        fallback = CountingFallback(layer)
        fallback.error = ValueError("the dequantized path failed")

        with pytest.raises(ValueError, match="dequantized path failed"):
            _int8_linear(layer, torch.randn(3, 512), fallback)
        assert events == ["synchronize"], "the kernels were waited for before the reference ran"
        assert not int8_matmul._health.is_retired(CPU)

    def test_a_kernel_fault_surfacing_asynchronously_retires_the_device_and_falls_back(
        self, fake_kernels, monkeypatch, caplog
    ):
        """Triton launches return before the kernel runs; a fault shows up at the next synchronization, which on a key's
        first call is the one before the reference -- it must count as the kernel's, not escape the generation."""

        def faulted(device):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        monkeypatch.setattr(int8_matmul, "_synchronize", faulted)
        layer = _layer()
        x = torch.randn(3, 512)

        outputs = [layer(x), layer(x)]

        assert all(torch.equal(out, _dequantized(layer, x)) for out in outputs)
        assert len(fake_kernels.calls) == 1 and int8_matmul._health.is_retired(CPU)
        assert len(caplog.records) == 1 and "illegal memory access" in caplog.records[0].message

    def test_only_calls_with_a_key_to_validate_wait_for_the_kernels(self, fake_kernels, monkeypatch):
        synchronized = []
        monkeypatch.setattr(int8_matmul, "_synchronize", synchronized.append)
        layer = _layer()
        x = torch.randn(3, 512)

        layer(x)
        layer(x)

        assert synchronized == [CPU], "validated calls stay asynchronous"

    def test_the_failed_output_is_released_before_the_fallback_allocates(self, fake_kernels):
        """Kept validation rows and chunk views must not hold the kernels' output buffer through the fallback."""
        torch.manual_seed(6)
        layer = _layer(out_features=16)
        buffers = []

        def second_chunk_fails(rows, weight, weight_scale, bias, out, m_bucket):
            buffers.append(weakref.ref(out._base))
            if len(buffers) == 2:
                raise RuntimeError("Triton Error [CUDA]: invalid argument")

        fake_kernels.w8a8_rows = second_chunk_fails
        released = []

        def fallback(x):
            released.append(all(buffer() is None for buffer in buffers))
            return _dequantized(layer, x)

        _int8_linear(layer, torch.randn(77, 512), fallback)

        assert released == [True]


class TestTheModelCacheWrapper:
    def test_it_reaches_the_kernels_with_and_without_a_lora_sidecar(self, fake_kernels, caplog):
        torch.manual_seed(4)
        model = torch.nn.Module()
        model.add_module("proj", _layer())
        apply_custom_layers_to_model(model)
        down, up = torch.randn(4, 512) * 0.1, torch.randn(384, 4) * 0.1
        patch = ModelPatchRaw(
            layers={"lora_transformer-proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)}
        )
        x = torch.randn(5, 512)

        plain = model.proj(x)
        with LayerPatcher.apply_smart_model_patches(
            model=model, patches=[(patch, 0.5)], prefix="lora_transformer-", dtype=torch.float32,
            force_sidecar_patching=True,
        ):  # fmt: skip
            patched = model.proj(x)

        assert fake_kernels.calls == [5, 5] and _served_by_the_kernels(caplog)
        torch.testing.assert_close(plain, _dequantized(model.proj, x) + MARK)
        torch.testing.assert_close(patched, plain + 0.5 * F.linear(F.linear(x, down), up))


def test_layers_the_kernels_do_not_cover_are_counted_at_load(caplog, int8_compute):
    """Comfy's quantized-layer config carries `full_precision_matrix_mult` for every format: a marked int8 layer keeps
    dequantizing, and so does an unrotated one; a model that gains little from the setting says why."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(*(torch.nn.Linear(512, 64) for _ in range(3)))
    sd = {}
    for index in ("0", "1", "2"):
        sd[f"{index}.weight"], sd[f"{index}.weight_scale"] = _quantize(torch.randn(64, 512), convrot=index != "2")
        sd[f"{index}.bias"] = torch.zeros(64)
    markers = {
        "0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP},
        "1": {"format": "int8_tensorwise", "convrot": True, "full_precision_matrix_mult": True},
        "2": {"format": "int8_tensorwise", "convrot": False},
    }
    caplog.set_level(logging.INFO, logger="invokeai.backend.quantization.int8_convrot")

    swap_in_int8_linears(model, sd, markers)

    assert [layer.int8_matmul_eligible for layer in model] == [True, False, False]
    messages = [r.message for r in caplog.records if "int8_compute" in r.message]
    assert len(messages) == 1 and "2 of 3" in messages[0] and "1 marked full_precision_matrix_mult" in messages[0]


def _cuda_unavailable_reason() -> str | None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "needs an NVIDIA GPU"
    if torch.cuda.get_device_capability(0) < (8, 0):
        return "needs compute capability 8.0 or newer"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "needs Triton"
    return None


def _gemm_configs() -> list:
    """The autotuner's tiles, where the kernels module can be imported at all."""
    if _cuda_unavailable_reason() is not None:
        return []
    from invokeai.backend.quantization.int8_matmul_kernels import gemm_configs

    return gemm_configs()


def _tile_id(tile) -> str:
    meta = tile.kwargs
    return f"{meta['BM']}x{meta['BN']}x{meta['BK']}-s{tile.num_stages}-w{tile.num_warps}"


def _w8a8_reference(x: torch.Tensor, layer: Int8ConvrotLinear) -> torch.Tensor:
    """W8A8 in plain torch and float64: rotation, round half up per token, an exact integer GEMM (float64 holds every
    int8 dot product of these widths exactly), then the scales and bias."""
    k = x.shape[-1]
    rows = x.reshape(-1, k).double()
    rotation = build_regular_hadamard(GROUP, torch.float64).to(x.device)
    rows = (rows.view(-1, k // GROUP, GROUP) @ rotation).view(-1, k)
    amax = rows.abs().amax(dim=1).clamp_min(1e-12)
    q = torch.floor(rows * (127.0 / amax)[:, None] + 0.5).clamp(-127, 127)
    out = (q @ layer.weight.to(x.device).double().t()) * (amax / 127.0)[:, None]
    out = out * layer.weight_scale.to(x.device).double().reshape(1, -1)
    if layer.bias is not None:
        out = out + layer.bias.to(x.device).double()
    return out.reshape(*x.shape[:-1], -1)


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


@pytest.mark.slow
@pytest.mark.skipif(_cuda_unavailable_reason() is not None, reason=str(_cuda_unavailable_reason()))
class TestOnNvidiaGpu:
    DEVICE = torch.device("cuda:0")

    @pytest.fixture
    def kernel_calls(self, monkeypatch, int8_compute):
        """Counts the calls that reached the real kernels."""
        import invokeai.backend.quantization.int8_matmul_kernels as kernels

        calls = []
        real = kernels.w8a8_rows

        def counting(rows, *args, **kwargs):
            calls.append(rows.shape)
            return real(rows, *args, **kwargs)

        monkeypatch.setattr(kernels, "w8a8_rows", counting)
        return calls

    def _on_gpu(self, layer: Int8ConvrotLinear) -> Int8ConvrotLinear:
        return layer.to(self.DEVICE)

    def _x(self, *shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, device=self.DEVICE, dtype=dtype)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("tokens", [1, 3, 77, 4097])
    def test_rotated_layers_match_the_w8a8_reference(self, kernel_calls, tokens, dtype):
        torch.manual_seed(tokens)
        layer = self._on_gpu(_layer())
        x = self._x(tokens, 512, dtype=dtype)

        out = layer(x)

        assert kernel_calls and out.dtype == dtype and out.shape == (tokens, 384)
        # Only the final rounding to the compute dtype (and a rare activation code one step off, where the rotation
        # sums in another order) separates the kernels from the reference.
        assert _relative(out, _w8a8_reference(x, layer)) < 2e-3

    @pytest.mark.parametrize("tile", _gemm_configs(), ids=_tile_id)
    def test_every_tile_the_autotuner_can_pick_computes_the_same_product(self, int8_compute, tile):
        """The autotuner only ever runs its winner; any of these can win on another GPU or shape. M, N and K are all
        partial tiles of every config."""
        import invokeai.backend.quantization.int8_matmul_kernels as kernels

        torch.manual_seed(5)
        layer = self._on_gpu(_layer(out_features=300, in_features=1024))
        x = self._x(77, 1024)
        q, q_scale = kernels.quantize_activations(x)
        out = torch.empty(77, 300, device=self.DEVICE, dtype=torch.bfloat16)

        kernels.launch_gemm(
            kernels._int8_gemm_kernel, q, q_scale, layer.weight, layer.weight_scale, layer.bias, out, 128,
            **tile.all_kwargs(),
        )  # fmt: skip

        assert _relative(out, _w8a8_reference(x, layer)) < 2e-3

    @pytest.mark.parametrize("layout", ["column", "vector", "scalar"])
    def test_every_accepted_scale_layout_scales_the_output_channels(self, kernel_calls, layout):
        torch.manual_seed(1)
        layer = _layer()
        scale = {"column": layer.weight_scale, "vector": layer.weight_scale[:, 0], "scalar": layer.weight_scale.max()}
        layer.weight_scale = scale[layout].clone()
        layer = self._on_gpu(layer)
        x = self._x(77, 512)

        out = layer(x)

        assert kernel_calls
        assert _relative(out, _w8a8_reference(x, layer)) < 2e-3

    def test_the_result_stays_close_to_the_dense_weight(self, kernel_calls):
        """W8A8 measures ~0.8 % here, the dequantized path ~0.3 %. A bound well above both still catches a lost
        scale, rotation or bias."""
        torch.manual_seed(2)
        dense = torch.randn(384, 512) * 0.05
        q, scale = _quantize(dense, convrot=True)
        layer = self._on_gpu(Int8ConvrotLinear(q, scale, convrot=True))
        x = self._x(4097, 512)

        error = _relative(layer(x), F.linear(x.double(), dense.to(self.DEVICE).double()))

        assert kernel_calls and error < 0.03

    def test_rows_strided_in_memory_quantize_like_contiguous_ones(self, kernel_calls):
        """The FluxControl sidecar path slices `x[..., :in_features]` out of a wider input."""
        torch.manual_seed(6)
        layer = self._on_gpu(_layer())
        wide = self._x(2, 77, 640)

        assert torch.equal(layer(wide[..., :512]), layer(wide[..., :512].contiguous()))
        assert len(kernel_calls) == 2

    def test_a_layer_streamed_from_ram_gets_the_resident_result(self, kernel_calls):
        """Partial loading changes where the int8 weight lives, not what the layer computes."""
        torch.manual_seed(7)
        layer = self._on_gpu(_layer())
        x = self._x(77, 512)
        resident = layer(x)
        layer.weight, layer.weight_scale = layer.weight.cpu(), layer.weight_scale.cpu()

        streamed = layer(x)

        assert len(kernel_calls) == 2 and torch.equal(streamed, resident)

    def test_a_nan_input_row_comes_out_nan_as_on_the_dequantized_path(self, kernel_calls):
        torch.manual_seed(8)
        layer = self._on_gpu(_layer())
        x = self._x(9, 512)
        clean = layer(x)
        x[4, 100] = float("nan")

        out = layer(x)

        assert out[4].isnan().all() and _dequantized(layer, x)[4].isnan().all()
        assert torch.equal(out[torch.arange(9) != 4], clean[torch.arange(9) != 4])

    @pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
    def test_the_kernels_stay_within_the_dequantized_paths_memory(self, kernel_calls, streamed):
        """M far above 2·N, so the output's rows are produced in several chunks."""
        torch.manual_seed(9)
        layer = self._on_gpu(_layer(out_features=256))
        x = self._x(4097, 512)
        layer(x)  # tunes and validates every key first
        if streamed:
            layer.weight, layer.weight_scale = layer.weight.cpu(), layer.weight_scale.cpu()
        output_bytes = 4097 * 256 * x.element_size()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.DEVICE)
        baseline = torch.cuda.memory_allocated(self.DEVICE)

        out = layer(x)

        transient = torch.cuda.max_memory_allocated(self.DEVICE) - baseline - output_bytes
        assert out.shape == (4097, 256) and len(kernel_calls) > 2
        assert transient <= layer.dequant_transient_bytes(x.dtype)

    def test_the_cache_wrapper_with_a_lora_sidecar_runs_the_kernels_and_adds_the_residual(self, kernel_calls):
        torch.manual_seed(3)
        model = torch.nn.Module()
        model.add_module("proj", self._on_gpu(_layer()))
        apply_custom_layers_to_model(model)
        down, up = torch.randn(4, 512) * 0.1, torch.randn(384, 4) * 0.1
        patch = ModelPatchRaw(
            layers={"lora_transformer-proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)}
        )
        x = self._x(77, 512)

        with LayerPatcher.apply_smart_model_patches(
            model=model, patches=[(patch, 0.75)], prefix="lora_transformer-", dtype=torch.bfloat16,
            force_sidecar_patching=True,
        ):  # fmt: skip
            out = model.proj(x)

        residual = 0.75 * F.linear(F.linear(x.double(), down.to(self.DEVICE).double()), up.to(self.DEVICE).double())
        assert len(kernel_calls) == 1
        assert _relative(out, _w8a8_reference(x, model.proj) + residual) < 3e-3

    @pytest.mark.parametrize("case", ["unrotated", "group-64", "full-precision", "fp32-input"])
    def test_layers_the_kernels_do_not_cover_compute_exactly_what_they_did(self, kernel_calls, case):
        torch.manual_seed(4)
        layer = self._on_gpu(_layer(convrot=case != "unrotated", group=64 if case == "group-64" else GROUP))
        layer.full_precision_matmul = case == "full-precision"
        x = self._x(77, 512, dtype=torch.float32 if case == "fp32-input" else torch.bfloat16)

        assert torch.equal(layer(x), _dequantized(layer, x))
        assert not kernel_calls

    def test_without_triton_the_layer_falls_back_and_says_why_once(self, monkeypatch, int8_compute, caplog):
        monkeypatch.setitem(sys.modules, "triton", None)
        monkeypatch.delitem(sys.modules, int8_matmul._KERNELS_MODULE, raising=False)
        int8_matmul._load_kernels.cache_clear()
        caplog.set_level(logging.INFO, logger=int8_matmul.__name__)
        layer = self._on_gpu(_layer())
        x = self._x(77, 512)
        try:
            outputs = [layer(x), layer(x)]
        finally:
            int8_matmul._load_kernels.cache_clear()

        assert all(torch.equal(out, _dequantized(layer, x)) for out in outputs)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "triton-windows" in warnings[0]

    def test_a_kernel_error_on_the_gpu_retires_it_and_falls_back(self, monkeypatch, int8_compute, caplog):
        import invokeai.backend.quantization.int8_matmul_kernels as kernels

        def broken_gemm(*args, **kwargs):
            raise RuntimeError("Triton Error [CUDA]: invalid argument")

        monkeypatch.setattr(kernels, "launch_gemm", broken_gemm)
        caplog.set_level(logging.WARNING, logger=int8_matmul.__name__)
        layer = self._on_gpu(_layer())
        x = self._x(77, 512)

        assert torch.equal(layer(x), _dequantized(layer, x))
        assert int8_matmul._health.is_retired(self.DEVICE)
        assert len([r for r in caplog.records if "cuda:0" in r.message]) == 1
