"""Tests for `ErnieImageDenoiseInvocation`'s initial-latent handling.

ERNIE denoises on the rectified-flow path, so the loop assumes its input already sits at the first
sigma of the schedule. Both directions of that contract are easy to get silently wrong:

- clean init latents passed through *unnoised* tell the model the sample is at `sigma_0` when it is
  at 0 (guarded by `add_noise=True`, with the blend itself done in `denoise()`, which is the only
  layer that knows the post-shift sigma)
- pure noise at a *reduced* `sigma_0` tells the model it is already partly denoised

Neither produces an error at runtime -- just a bad image -- so they are pinned here.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.ernie_image.ernie_image_denoise import ErnieImageDenoiseInvocation
from invokeai.app.invocations.fields import LatentsField


def _invocation(*, denoising_start: float = 0.0, latents: LatentsField | None = None) -> ErnieImageDenoiseInvocation:
    return ErnieImageDenoiseInvocation.model_construct(denoising_start=denoising_start, latents=latents)


def _context(tensor: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(tensors=SimpleNamespace(load=lambda _name: tensor))


def _load(invocation: ErnieImageDenoiseInvocation, context: SimpleNamespace, noise: torch.Tensor):
    return invocation._load_init_latents(context, noise, device=torch.device("cpu"), dtype=torch.float32)  # type: ignore[arg-type]


def test_txt2img_reports_no_init_latents() -> None:
    noise = torch.randn(1, 128, 4, 4)

    assert _load(_invocation(), _context(), noise) is None


def test_denoising_start_without_init_latents_is_rejected() -> None:
    """Starting at sigma 0.5 from pure noise is silently wrong, so it must raise.

    `get_schedule` happily produces a window starting at 0.5, and nothing downstream can tell that
    the sample handed to it is full-magnitude noise rather than a half-denoised latent.
    """
    noise = torch.randn(1, 128, 4, 4)

    with pytest.raises(ValueError, match="denoising_start must be 0"):
        _load(_invocation(denoising_start=0.5), _context(), noise)


def test_init_latents_are_returned_for_the_loop_to_blend() -> None:
    """The invocation validates and hands them over; it must not blend them itself.

    Blending here would have to use the raw schedule sigma, since the scheduler's `shift` is only
    applied later inside `set_timesteps`.
    """
    noise = torch.randn(1, 128, 4, 4)
    init = torch.randn(1, 128, 4, 4)
    invocation = _invocation(denoising_start=0.25, latents=LatentsField(latents_name="init"))

    result = _load(invocation, _context(init), noise)

    assert result is not None
    assert torch.equal(result, init)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 4, 4),  # unpatchified -- 32 latent channels instead of the patched 128
        (1, 128, 8, 8),  # right channels, wrong spatial grid
        (2, 128, 4, 4),  # batched, but the text conditioning is built for batch 1
    ],
)
def test_mismatched_init_latents_are_rejected(shape: tuple[int, ...]) -> None:
    """A mismatch must fail loudly here rather than broadcast or blow up inside the transformer."""
    noise = torch.randn(1, 128, 4, 4)
    invocation = _invocation(latents=LatentsField(latents_name="init"))

    with pytest.raises(ValueError, match="patchified"):
        _load(invocation, _context(torch.randn(*shape)), noise)


def test_add_noise_defaults_to_on() -> None:
    """A clean VAE-encoded latent is the common case, and it has to be noised to be usable."""
    assert ErnieImageDenoiseInvocation.model_fields["add_noise"].default is True


class _StopBeforeLoad(Exception):
    """Raised out of the mocked cache so the node stops before it denoises."""


_GIB = 1024**3


def _gguf_transformer(
    weight: torch.Tensor | None = None, compute_dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    import gguf
    import numpy as np

    from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

    weight = torch.zeros(128, 64) if weight is None else weight
    linear = torch.nn.Linear(64, 128, bias=False)
    raw = gguf.quantize(weight.numpy().astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    linear.weight = torch.nn.Parameter(
        GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, torch.Size((128, 64)), compute_dtype),
        requires_grad=False,
    )
    return torch.nn.Sequential(linear)


def _lora(key: str, up: torch.Tensor | None = None, down: torch.Tensor | None = None):
    """A LoRA field entry and what loading its model returns: one low-rank layer on the Linear at index 0."""
    from contextlib import nullcontext

    from invokeai.backend.patches.layers.lora_layer import LoRALayer
    from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import (
        ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
    )
    from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

    up = torch.zeros(128, 16) if up is None else up
    down = torch.zeros(16, 64) if down is None else down
    layer = LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)
    patch = ModelPatchRaw(layers={f"{ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX}0": layer})
    info = SimpleNamespace(model=patch, model_in_ram=nullcontext)
    return SimpleNamespace(lora=SimpleNamespace(key=key), weight=1.0), info


def _run_node(monkeypatch, model: torch.nn.Module, model_format, loras=(), *, floor_gb=3.0, while_patched=None):
    """Run the node against a mocked cache and return its `model_on_device()` mock.

    Without `while_patched` the node stops at the transformer lock. With it, the lock hands over `model`, and
    `while_patched(model)` runs where the node starts using the transformer -- with its LoRAs applied.
    """
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from invokeai.app.invocations.ernie_image import ernie_image_denoise

    transformer_info = MagicMock()
    transformer_info.model = model
    if while_patched is None:
        transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)
    else:

        @contextmanager
        def on_device(working_mem_bytes: int):
            yield None, model

        def pad_text(*args, **kwargs):
            while_patched(model)
            raise _StopBeforeLoad

        transformer_info.model_on_device = MagicMock(side_effect=on_device)
        model.config = SimpleNamespace(text_in_dim=8, in_channels=128)
        monkeypatch.setattr(ernie_image_denoise, "sampling_utils", SimpleNamespace(pad_text=pad_text))

    loaded = {"transformer": transformer_info, **{field.lora.key: info for field, info in loras}}
    context = MagicMock()
    context.models.load.side_effect = lambda identifier: loaded[identifier.key]
    context.models.get_config.return_value = SimpleNamespace(format=model_format)
    context.config.get.return_value = SimpleNamespace(device_working_mem_gb=floor_gb)
    invocation = ErnieImageDenoiseInvocation.model_construct(
        transformer=SimpleNamespace(transformer=SimpleNamespace(key="transformer"), loras=[f for f, _ in loras]),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=None,
        guidance_scale=1.0,
    )
    monkeypatch.setattr(
        ErnieImageDenoiseInvocation, "_load_conditioning", lambda *args, **kwargs: SimpleNamespace(prompt_embeds=None)
    )

    with pytest.raises(_StopBeforeLoad):
        invocation.invoke(context)
    return transformer_info.model_on_device


def _requested_working_memory(monkeypatch, model: torch.nn.Module, model_format, loras=(), **kwargs) -> int:
    return _run_node(monkeypatch, model, model_format, loras, **kwargs).call_args.kwargs["working_mem_bytes"]


def _dequant_transient(model: torch.nn.Module) -> int:
    from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
    from invokeai.backend.util.devices import TorchDevice

    dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
    return peak_dequant_transient_bytes(model, dtype)


def test_a_gguf_transformer_without_loras_reserves_only_its_dequant_transient(monkeypatch) -> None:
    """The node has no activation estimate, so the GGUF dequantization copy is the whole request; the cache's
    `device_working_mem_gb` floor stands in for the activations."""
    from invokeai.backend.model_manager.taxonomy import ModelFormat

    model = _gguf_transformer()

    requested = _requested_working_memory(monkeypatch, model, ModelFormat.GGUFQuantized)

    # At least the bfloat16 copy of the weight it dequantizes on every forward.
    assert requested == _dequant_transient(model) >= 128 * 64 * 2


@pytest.mark.parametrize("floor_gb", [3.0, 0.0], ids=["default-floor", "floor-lowered"])
def test_sidecar_loras_are_reserved_on_top_of_the_floor(monkeypatch, floor_gb: float) -> None:
    """Patches that stay on the device for the whole run must not eat into the headroom the floor holds for the
    activations, and every LoRA counts. With the floor lowered, they sit on top of the dequant transient."""
    from invokeai.backend.model_manager.taxonomy import ModelFormat

    loras = [_lora("a"), _lora("b", up=torch.zeros(128, 32), down=torch.zeros(32, 64))]
    model = _gguf_transformer()
    lora_bytes = sum(info.model.calc_size() for _, info in loras)

    requested = _requested_working_memory(monkeypatch, model, ModelFormat.GGUFQuantized, loras, floor_gb=floor_gb)

    assert requested == max(_dequant_transient(model), int(floor_gb * _GIB)) + lora_bytes + _GIB


def test_fp8_weights_reserve_the_lora_patches(monkeypatch) -> None:
    """FP8 Storage has no add kernel, so the patcher puts the LoRA on the sidecar by itself."""
    from invokeai.backend.model_manager.taxonomy import ModelFormat

    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False))
    model[0].weight = torch.nn.Parameter(model[0].weight.data.to(torch.float8_e4m3fn), requires_grad=False)
    field, info = _lora("lora")

    requested = _requested_working_memory(monkeypatch, model, ModelFormat.Diffusers, [(field, info)])

    assert requested == 3 * _GIB + info.model.calc_size() + _GIB


def test_a_directly_patched_transformer_reserves_nothing_for_its_loras(monkeypatch) -> None:
    """Direct patching merges the LoRA into the weights and returns its tensors to RAM as it goes."""
    from invokeai.backend.model_manager.taxonomy import ModelFormat

    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False))

    assert _requested_working_memory(monkeypatch, model, ModelFormat.Diffusers, [_lora("lora")]) == 0


@pytest.mark.parametrize("model_format", ["gguf", "diffusers"])
def test_the_node_applies_its_loras_while_denoising_and_removes_them_after(monkeypatch, model_format: str) -> None:
    """The node's own patcher call has to put its LoRAs on the transformer, and take them off again.

    Both a packed GGUF Linear and a plain one must compute with W + delta while denoising and with W afterwards.
    On the CPU the patcher picks the sidecar for every layer by itself, so whether the node forces it for GGUF
    shows only on a GPU; the reservation tests above pin that decision. Computed in float32: a CPU bfloat16 matmul
    aborts some CI runners, and low precision is not what is under test.
    """
    from invokeai.backend.util.devices import TorchDevice

    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32))
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
        apply_custom_layers_to_model,
    )
    from invokeai.backend.model_manager.taxonomy import ModelFormat

    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(128, 64, generator=generator)
    if model_format == "gguf":
        model, fmt = _gguf_transformer(weight, compute_dtype=torch.float32), ModelFormat.GGUFQuantized
        weight = model[0].weight.get_dequantized_tensor().float()
    else:
        model, fmt = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False)), ModelFormat.Diffusers
        model[0].weight.data.copy_(weight)
    apply_custom_layers_to_model(model)
    up = torch.randn(128, 16, generator=generator) * 0.1
    down = torch.randn(16, 64, generator=generator)
    x = torch.randn(2, 64, generator=generator)
    seen: list[torch.Tensor] = []

    _run_node(monkeypatch, model, fmt, [_lora("lora", up, down)], while_patched=lambda m: seen.append(m(x)))

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], x @ (weight + up @ down).T, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(model(x), x @ weight.T, atol=1e-4, rtol=1e-4)
