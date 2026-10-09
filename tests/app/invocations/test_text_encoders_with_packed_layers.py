"""The text-encoder nodes have to treat a quantized encoder the way the denoise nodes treat a quantized transformer:
reserve working memory for the per-forward dequantization of packed nvfp4 and GGUF Linears, and apply LoRA as a sidecar
wherever a direct patch cannot write the weights -- packed nvfp4 Linears, and GGUF or SDNQ encoders, which the denoise
nodes already patch this way. The three Qwen3 nodes load through the Qwen3 loader and the FLUX.2 [dev] node through
the Mistral loader, and both keep Comfy's fp4_mixed files packed, so each node receives a packed encoder as soon as
someone installs one.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch

from invokeai.app.invocations.text_encoder.anima_text_encoder import AnimaTextEncoderInvocation
from invokeai.app.invocations.text_encoder.ernie_image_text_encoder import ErnieImageTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_dev_text_encoder import Flux2DevTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_klein_text_encoder import Flux2KleinTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux_text_encoder import FluxTextEncoderInvocation
from invokeai.app.invocations.text_encoder.ideogram4_text_encoder import Ideogram4TextEncoderInvocation
from invokeai.app.invocations.text_encoder.sd3_text_encoder import Sd3TextEncoderInvocation
from invokeai.app.invocations.text_encoder.z_image_text_encoder import ZImageTextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.nvfp4 import NVFP4Linear


class _StopAtPatching(Exception):
    """Raised from the patched `apply_smart_model_patches`, so the node stops right where it has decided."""


class _StopOnDevice(Exception):
    """Raised once the encoder is put on the device: the reservation is decided by then."""


class _LoadedModel:
    def __init__(self, model: object, model_format: ModelFormat = ModelFormat.Checkpoint) -> None:
        self.model = model
        self.config = SimpleNamespace(format=model_format)
        self.compute_device = torch.device("cpu")
        self.working_mem_bytes: int | None = None
        self.stop_on_device = False

    @contextmanager
    def model_on_device(self, working_mem_bytes: int | None = None):
        self.working_mem_bytes = working_mem_bytes
        if self.stop_on_device:
            raise _StopOnDevice
        yield (None, self.model)

    def repair_required_tensors_on_device(self) -> int:
        return 0


def _encoder(kind: str) -> torch.nn.Module:
    encoder = torch.nn.Module()
    if kind == "nvfp4":
        weight = torch.zeros(128, 32, dtype=torch.uint8)
        encoder.proj = NVFP4Linear(weight, torch.ones(128, 4).to(torch.float8_e4m3fn), torch.tensor(1.0))
    elif kind == "gguf":
        encoder.proj = torch.nn.Linear(64, 128, bias=False)
        raw = gguf.quantize(np.zeros((128, 64), np.float32), gguf.GGMLQuantizationType.Q8_0)
        encoder.proj.weight = torch.nn.Parameter(
            GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, torch.Size((128, 64)), torch.bfloat16),
            requires_grad=False,
        )
    else:
        encoder.proj = torch.nn.Linear(64, 128)
    return encoder


# (node class, the field naming its encoder, how the node's encode step is called)
NODES = {
    "z_image": (
        ZImageTextEncoderInvocation,
        "qwen3_encoder",
        lambda node, context: node._encode_prompt(context, max_seq_len=512),
    ),
    "flux2_klein": (
        Flux2KleinTextEncoderInvocation,
        "qwen3_encoder",
        lambda node, context: node._encode_prompt(context, ExitStack()),
    ),
    "anima": (AnimaTextEncoderInvocation, "qwen3_encoder", lambda node, context: node._encode_prompt(context)),
    "flux2_dev": (
        Flux2DevTextEncoderInvocation,
        "mistral_encoder",
        lambda node, context: node._encode_prompt(context, ExitStack()),
    ),
}


@pytest.mark.parametrize(
    ("kind", "model_format", "sidecar", "working_memory"),
    [
        ("nvfp4", ModelFormat.Checkpoint, True, True),
        ("dense", ModelFormat.Checkpoint, False, False),
        ("gguf", ModelFormat.GGUFQuantized, True, True),
    ],
    ids=["nvfp4_checkpoint", "dense_checkpoint", "gguf"],
)
@pytest.mark.parametrize("node_name", list(NODES))
def test_the_node_reserves_the_dequant_transient_and_patches_quantized_encoders_as_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    node_name: str,
    kind: str,
    model_format: ModelFormat,
    sidecar: bool,
    working_memory: bool,
) -> None:
    invocation_class, encoder_field, encode = NODES[node_name]
    encoder = _LoadedModel(_encoder(kind))
    context = MagicMock()
    context.models.load.side_effect = [encoder, _LoadedModel(object())]
    context.models.get_config.return_value = SimpleNamespace(format=model_format)
    patching: dict = {}

    def stop(**kwargs):
        patching.update(kwargs)
        raise _StopAtPatching

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)
    node = invocation_class.model_construct(
        prompt="a prompt",
        **{encoder_field: SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[])},
        mask=None,
        max_seq_len=512,
    )

    with pytest.raises(_StopAtPatching):
        encode(node, context)

    assert patching["force_sidecar_patching"] is sidecar
    assert bool(encoder.working_mem_bytes) is working_memory


@pytest.mark.parametrize(
    ("kind", "model_format", "sidecar", "working_memory"),
    [
        ("nvfp4", ModelFormat.Checkpoint, True, True),
        ("dense", ModelFormat.Checkpoint, False, False),
        ("gguf", ModelFormat.GGUFQuantized, True, True),
    ],
    ids=["nvfp4_checkpoint", "dense_checkpoint", "gguf"],
)
def test_the_krea2_node_reserves_the_dequant_transient_and_patches_quantized_encoders_as_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    model_format: ModelFormat,
    sidecar: bool,
    working_memory: bool,
) -> None:
    """Krea-2's Qwen3-VL encoder became loadable from GGUF, which makes this node's quantized path
    reachable for the first time -- and its nvfp4 path was already reachable through the ComfyUI
    single-file encoder.

    A direct LoRA patch cannot write a packed weight: a GGMLTensor reports `dtype=uint8` (so the
    patcher's fp8 check misses it) and a packed `nelement()`, so every encoder LoRA layer is dropped
    over an apparent shape mismatch that blames the LoRA. It is also VRAM-dependent -- with the
    layers left on CPU the patcher picks the sidecar anyway -- so the same graph would otherwise
    produce different images under memory pressure.

    Kept out of the table above because this node loads its tokenizer first and enters that handle
    as a context manager, which the shared harness does not model.
    """
    from invokeai.app.invocations.text_encoder.krea2_text_encoder import Krea2TextEncoderInvocation

    class _LoadedTokenizer:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *_exc):
            return False

    encoder = _LoadedModel(_encoder(kind))
    context = MagicMock()
    # The node loads the tokenizer first, then the encoder.
    context.models.load.side_effect = [_LoadedTokenizer(), encoder]
    context.models.get_config.return_value = SimpleNamespace(format=model_format)
    patching: dict = {}

    def stop(**kwargs):
        patching.update(kwargs)
        raise _StopAtPatching

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", stop)
    node = Krea2TextEncoderInvocation.model_construct(
        prompt="a prompt",
        qwen3_vl_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        mask=None,
    )

    with pytest.raises(_StopAtPatching):
        node._encode(context)

    assert patching["force_sidecar_patching"] is sidecar
    assert bool(encoder.working_mem_bytes) is working_memory


# Nodes that reached a GGUF encoder without asking for its transient at all: (node class, how its
# encoder field is filled, how its encode step is called).
GGUF_ONLY_NODES = {
    "flux_t5": (
        FluxTextEncoderInvocation,
        {"t5_encoder": SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[])},
        lambda node, context: node._t5_encode(context),
    ),
    "sd3_t5": (
        Sd3TextEncoderInvocation,
        {"t5_encoder": SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace())},
        lambda node, context: node._t5_encode(context, 16),
    ),
    "ideogram4": (
        Ideogram4TextEncoderInvocation,
        {"qwen3_encoder": SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace())},
        lambda node, context: node.invoke(context),
    ),
    "ernie_image": (
        ErnieImageTextEncoderInvocation,
        {"text_encoder": SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace())},
        lambda node, context: node._encode_prompt(context, "a prompt"),
    ),
}


@pytest.mark.parametrize("kind", ["gguf", "dense"])
@pytest.mark.parametrize("node_name", list(GGUF_ONLY_NODES))
def test_a_gguf_encoder_reserves_its_dequant_transient(node_name: str, kind: str) -> None:
    """The reservation is decided when the encoder is put on the device, before any encoding."""
    invocation_class, fields, encode = GGUF_ONLY_NODES[node_name]
    model_format = ModelFormat.GGUFQuantized if kind == "gguf" else ModelFormat.Checkpoint
    encoder = _LoadedModel(_encoder(kind), model_format)
    encoder.stop_on_device = True
    context = MagicMock()
    context.models.load.side_effect = [encoder, _LoadedModel(object())]
    node = invocation_class.model_construct(prompt="a prompt", t5_max_seq_len=512, **fields)

    with pytest.raises(_StopOnDevice):
        encode(node, context)

    if kind == "gguf":
        # At least the bfloat16 copy of the one packed 128x64 weight.
        assert encoder.working_mem_bytes is not None and encoder.working_mem_bytes > 128 * 64 * 2
    else:
        assert not encoder.working_mem_bytes
