"""The text-encoder nodes have to treat a quantized encoder the way the denoise nodes treat a quantized transformer:
reserve working memory for the per-forward dequantization of packed nvfp4 Linears, and apply LoRA as a sidecar
wherever a direct patch cannot write the weights -- packed nvfp4 Linears, and GGUF or SDNQ encoders, which the denoise
nodes already patch this way. The three Qwen3 nodes load through the Qwen3 loader and the FLUX.2 [dev] node through
the Mistral loader, and both keep Comfy's fp4_mixed files packed, so each node receives a packed encoder as soon as
someone installs one.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.text_encoder.anima_text_encoder import AnimaTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_dev_text_encoder import Flux2DevTextEncoderInvocation
from invokeai.app.invocations.text_encoder.flux2_klein_text_encoder import Flux2KleinTextEncoderInvocation
from invokeai.app.invocations.text_encoder.z_image_text_encoder import ZImageTextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.quantization.nvfp4 import NVFP4Linear


class _StopAtPatching(Exception):
    """Raised from the patched `apply_smart_model_patches`, so the node stops right where it has decided."""


class _LoadedModel:
    def __init__(self, model: object) -> None:
        self.model = model
        self.compute_device = torch.device("cpu")
        self.working_mem_bytes: int | None = None

    @contextmanager
    def model_on_device(self, working_mem_bytes: int | None = None):
        self.working_mem_bytes = working_mem_bytes
        yield (None, self.model)

    def repair_required_tensors_on_device(self) -> int:
        return 0


def _encoder(packed: bool) -> torch.nn.Module:
    encoder = torch.nn.Module()
    if packed:
        weight = torch.zeros(128, 32, dtype=torch.uint8)
        encoder.proj = NVFP4Linear(weight, torch.ones(128, 4).to(torch.float8_e4m3fn), torch.tensor(1.0))
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
    ("packed", "model_format", "sidecar", "working_memory"),
    [
        (True, ModelFormat.Checkpoint, True, True),
        (False, ModelFormat.Checkpoint, False, False),
        (False, ModelFormat.GGUFQuantized, True, False),
    ],
    ids=["nvfp4_checkpoint", "dense_checkpoint", "gguf"],
)
@pytest.mark.parametrize("node_name", list(NODES))
def test_the_node_reserves_the_dequant_transient_and_patches_quantized_encoders_as_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    node_name: str,
    packed: bool,
    model_format: ModelFormat,
    sidecar: bool,
    working_memory: bool,
) -> None:
    invocation_class, encoder_field, encode = NODES[node_name]
    encoder = _LoadedModel(_encoder(packed))
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
    ("packed", "model_format", "sidecar", "working_memory"),
    [
        (True, ModelFormat.Checkpoint, True, True),
        (False, ModelFormat.Checkpoint, False, False),
        (False, ModelFormat.GGUFQuantized, True, False),
    ],
    ids=["nvfp4_checkpoint", "dense_checkpoint", "gguf"],
)
def test_the_krea2_node_reserves_the_dequant_transient_and_patches_quantized_encoders_as_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    packed: bool,
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

    encoder = _LoadedModel(_encoder(packed))
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
