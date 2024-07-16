from abc import ABC, abstractmethod
from typing import Optional

import torch


class RawModel(ABC):
    """Base class for 'Raw' models.

    The RawModel class is the base class of LoRAModelRaw, TextualInversionModelRaw, etc.
    and is used for type checking of calls to the model patcher. Its main purpose
    is to avoid a circular import issues when lora.py tries to import BaseModelType
    from invokeai.backend.model_manager.config, and the latter tries to import LoRAModelRaw
    from lora.py.

    The term 'raw' was introduced to describe a wrapper around a torch.nn.Module
    that adds additional methods and attributes.
    """

    @abstractmethod
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        pass
