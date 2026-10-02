"""Keep Qwen3VLVisionModel's pos-embed interpolation on the compute device under partial loading.

``Qwen3VLVisionModel.fast_pos_embed_interpolate`` (transformers 5.5.x,
``modeling_qwen3_vl.py``) derives its working device from ``self.pos_embed.weight.device`` and
builds every tensor it makes — index tensor, interpolation weights, the looked-up embeddings —
on that device. Under InvokeAI's partial loading that is the wrong device to trust:
``nn.Embedding`` is autocast-wrapped (``CustomEmbedding``), so ``pos_embed.weight`` may
legitimately reside on the CPU while the model computes on CUDA. The device-autocast wrappers
cast weights to each *input's* device per-op, which keeps every normal layer on the compute
device — but this function manufactures its own inputs on the weight's device, so the whole
interpolation lands on the CPU and ``forward``'s ``hidden_states + pos_embeds`` fails with
"Expected all tensors to be on the same device, but found at least two devices, cuda:0 and
cpu!".

Observed in the wild when two simultaneous MiniMax H3 video jobs partial-loaded the 27 GB text
encoder and a Ref2VA reference image sent a prompt through the vision tower; any VRAM pressure
that leaves ``visual.pos_embed.weight`` off-device reproduces it. It cannot be triggered on a
fully loaded model (device autocasting is then disabled and the weight is resident), which is
why it appears only intermittently, under memory pressure.

The patch is deliberately minimal: ``forward`` records the true compute device from its
``hidden_states`` input (the tensor ``pos_embeds`` will be added to), and
``fast_pos_embed_interpolate`` runs UNCHANGED — every internal op stays on the weight's device,
preserving upstream behavior exactly when the weight is resident — with only its result moved
to the recorded device. When the weight lives on the CPU the interpolation arithmetic (a few
thousand rows) runs there and one small tensor crosses the bus; correctness is unaffected
either way.

Class-level and idempotent, mirroring ``contiguous_attention``: applied by both text-encoder
load paths (the diffusers-folder and single-file loaders in
``model_loaders/minimax_h3.py``).
"""

from functools import wraps

import torch

from invokeai.backend.util.logging import InvokeAILogger

_SENTINEL = "_invokeai_pos_embed_device_patch"
_DEVICE_ATTR = "_invokeai_vision_input_device"


def apply_qwen3vl_vision_pos_embed_device_patch() -> None:
    """Install the patch on ``Qwen3VLVisionModel`` (idempotent, class-level)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    if getattr(Qwen3VLVisionModel, _SENTINEL, False):
        return

    orig_forward = getattr(Qwen3VLVisionModel, "forward", None)
    orig_interpolate = getattr(Qwen3VLVisionModel, "fast_pos_embed_interpolate", None)
    if orig_forward is None or orig_interpolate is None:
        # A future transformers restructured the class; the residency bug this guards against
        # may be gone too. Skip rather than crash the model load, but say so.
        InvokeAILogger.get_logger(__name__).warning(
            "Qwen3VLVisionModel no longer has forward/fast_pos_embed_interpolate; skipping the "
            "pos-embed device patch. If partially-loaded H3 text encoders start failing with "
            "cross-device errors in the vision tower, this patch needs updating."
        )
        return

    @wraps(orig_forward)
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        # hidden_states is the tensor pos_embeds will be added to — its device is the one
        # authoritative answer to "where is this forward computing", independent of where the
        # partial loader happened to leave any given weight.
        setattr(self, _DEVICE_ATTR, hidden_states.device)
        return orig_forward(self, hidden_states, grid_thw, **kwargs)

    @wraps(orig_interpolate)
    def fast_pos_embed_interpolate(self, grid_thw):
        pos_embeds = orig_interpolate(self, grid_thw)
        target = getattr(self, _DEVICE_ATTR, None)
        if target is not None and pos_embeds.device != target:
            pos_embeds = pos_embeds.to(target)
        return pos_embeds

    Qwen3VLVisionModel.forward = forward
    Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    setattr(Qwen3VLVisionModel, _SENTINEL, True)
