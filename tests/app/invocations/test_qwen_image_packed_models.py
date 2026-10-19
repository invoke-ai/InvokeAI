"""The Qwen-Image nodes have to treat a packed nvfp4 model like the other nodes treat one.

The denoise node reserves the per-forward dequantization transient of packed Linears and applies LoRA to them as
sidecars, which it used to decide from the format alone -- GGUF only -- so a `checkpoint` build holding packed Linears
would have been patched directly. The text encoder node reserves the same transient.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.qwen_image import qwen_image_denoise
from invokeai.app.invocations.qwen_image.qwen_image_denoise import QwenImageDenoiseInvocation
from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import QwenImageTextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.quantization.nvfp4 import NVFP4Linear

# One packed 128x64 Linear at bf16: the weight (2 bytes), its int32 byte index (4 bytes per packed byte, so 2 per
# element) and the 16-element block scale grid (2 bytes per block).
PACKED_TRANSIENT_BYTES = 128 * 64 * (2 + 2) + (128 * 64 // 16) * 2


class _Stop(Exception):
    """Raised where the node has made the decision a test looks at."""


def _module(packed: bool) -> torch.nn.Module:
    module = torch.nn.Module()
    if packed:
        module.proj = NVFP4Linear(
            torch.zeros(128, 32, dtype=torch.uint8), torch.ones(128, 4).to(torch.float8_e4m3fn), torch.tensor(1.0)
        )
    else:
        module.proj = torch.nn.Linear(64, 128)
    return module


class _TinyQwenImageTransformer(torch.nn.Module):
    """Stands in for `QwenImageTransformer2DModel`: the config the node reads before the lock, and one Linear."""

    def __init__(self, packed: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=2, out_channels=16, zero_cond_t=False)
        self.inner = _module(packed)


def _denoise_node() -> QwenImageDenoiseInvocation:
    return QwenImageDenoiseInvocation.model_construct(
        latents=None,
        reference_latents=None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        transformer=SimpleNamespace(transformer=SimpleNamespace(), loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=None,
        cfg_scale=1.0,
        width=64,
        height=64,
        steps=2,
        seed=0,
        shift=None,
    )


@pytest.mark.parametrize(
    ("packed", "model_format", "transient", "sidecar"),
    [
        (True, ModelFormat.Checkpoint, PACKED_TRANSIENT_BYTES, True),
        (False, ModelFormat.Checkpoint, 0, False),
        (False, ModelFormat.GGUFQuantized, 0, True),
    ],
    ids=["nvfp4_checkpoint", "dense_checkpoint", "gguf"],
)
def test_the_denoise_node_reserves_the_transient_and_patches_quantized_weights_as_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    packed: bool,
    model_format: ModelFormat,
    transient: int,
    sidecar: bool,
) -> None:
    transformer = _TinyQwenImageTransformer(packed=packed)
    monkeypatch.setattr(qwen_image_denoise, "QwenImageTransformer2DModel", _TinyQwenImageTransformer)
    monkeypatch.setattr(qwen_image_denoise.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        QwenImageDenoiseInvocation, "_load_text_conditioning", lambda self, **_kwargs: (torch.zeros(1, 4, 8), None)
    )
    loaded = MagicMock()
    loaded.model = transformer
    loaded.model_on_device = MagicMock(return_value=nullcontext((None, transformer)))
    context = MagicMock()
    context.models.load.return_value = loaded
    context.models.get_config.return_value = SimpleNamespace(format=model_format)
    context.models.get_absolute_path.return_value = tmp_path / "qwen_image.safetensors"
    patching: dict = {}

    def stop(**kwargs):
        patching.update(kwargs)
        raise _Stop

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)

    with pytest.raises(_Stop):
        _denoise_node()._run_diffusion(context)

    assert loaded.model_on_device.call_args.kwargs["working_mem_bytes"] == transient
    assert patching["force_sidecar_patching"] is sidecar


def test_the_text_encoder_reserves_the_dequant_transient_of_a_packed_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    class _TinyQwenVL(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = _module(packed=True)

    monkeypatch.setattr(transformers, "Qwen2_5_VLForConditionalGeneration", _TinyQwenVL, raising=False)
    encoder = _TinyQwenVL()
    reserved: dict = {}

    @contextmanager
    def model_on_device(working_mem_bytes=None):
        reserved["working_mem_bytes"] = working_mem_bytes
        yield (None, encoder)

    loaded = SimpleNamespace(model=encoder, compute_device=torch.device("cpu"), model_on_device=model_on_device)
    context = MagicMock()
    context.models.load.return_value = loaded
    node = QwenImageTextEncoderInvocation.model_construct(
        qwen_vl_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace())
    )

    _, _, cleanup = node._load_cached_encoder(context)
    cleanup()

    assert reserved["working_mem_bytes"] == PACKED_TRANSIENT_BYTES
