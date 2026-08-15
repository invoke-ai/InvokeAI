"""MiniMax H3 (Hailuo 3.0) model classes, vendored from the in-progress diffusers integration.

Vendored from huggingface/diffusers PR #14355 ("Add MiniMax-H3") at commit
abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc (branch `minimax-h3`), which is not yet
in any tagged diffusers release. The only local changes are rewriting the
package-relative imports to absolute `diffusers.*` imports (all referenced
symbols exist in the pinned diffusers==0.39.0) and ruff import sorting. Keep
these files otherwise
byte-identical to upstream: when a diffusers release ships the H3 classes,
delete this vendoring and import them from diffusers instead.

Apache-2.0, copyright The HuggingFace Team / MiniMax (see file headers).
"""

from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from invokeai.backend.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

__all__ = [
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3Scheduler",
    "MiniMaxH3Transformer3DModel",
]
