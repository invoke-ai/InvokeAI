"""Container holding Ideogram 4's two transformer branches as a single submodel.

Ideogram 4 uses dual-branch asymmetric CFG with two *separate* weight sets
(``transformer/`` and ``unconditional_transformer/`` on disk). InvokeAI's model
cache keys a cached entity by (model, submodel_type) and there is no
"unconditional transformer" submodel type, so we load both branches into one
``nn.Module`` returned for ``SubModelType.Transformer``. This keeps both branches
co-resident through the denoise loop (each step runs both), which is required for
acceptable performance and is what makes the nf4 build fit in 24 GB.
"""

from __future__ import annotations

import torch

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Transformer


class Ideogram4TransformerPair(torch.nn.Module):
    """Holds the conditional and unconditional Ideogram 4 transformers."""

    def __init__(self, conditional: Ideogram4Transformer, unconditional: Ideogram4Transformer) -> None:
        super().__init__()
        self.conditional = conditional
        self.unconditional = unconditional
