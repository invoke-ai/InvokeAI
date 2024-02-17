"""Base class for 'Raw' models.

The RawModel class is the base class of LoRAModelRaw and TextualInversionModelRaw,
and is used for type checking of calls to the model patcher. Its main purpose
is to avoid a circular import issues when lora.py tries to import BaseModelType
from invokeai.backend.model_manager.config, and the latter tries to import LoRAModelRaw
from lora.py.

The term 'raw' was introduced to describe a wrapper around a torch.nn.Module
that adds additional methods and attributes.
"""

class RawModel:
    """Base class for 'Raw' model wrappers."""
