"""Base class for LoRA and Textual Inversion models.

The EmbeddingRaw class is the base class of LoRAModelRaw and TextualInversionModelRaw,
and is used for type checking of calls to the model patcher.

The use of "Raw" here is a historical artifact, and carried forward in
order to avoid confusion.
"""


class EmbeddingModelRaw:
    """Base class for LoRA and Textual Inversion models."""
