from typing import Optional, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from diffusers import UNet2DConditionModel

@dataclass
class InjectionInfo:
    type: str
    name: str
    order: Optional[str]
    function: Callable

def modifier(name: str, order: str = "any"):
    def _decorator(func):
        func.__inj_info__ = dict(
            type="modifier",
            name=name,
            order=order,
        )
        return func
    return _decorator

def override(name: str):
    def _decorator(func):
        func.__inj_info__ = dict(
            type="override",
            name=name,
        )
        return func
    return _decorator

class ExtensionBase(ABC):
    def __init__(self, priority: int):
        self.priority = priority
        self.injections: List[InjectionInfo] = []
        for func_name in dir(self):
            func = getattr(self, func_name)
            if not callable(func) or not hasattr(func, "__inj_info__"):
                continue

            self.injections.append(InjectionInfo(**func.__inj_info__, function=func))

    def apply_attention_processor(self, attention_processor_cls: object):
        pass

    def restore_attention_processor(self):
        pass

    def patch_unet(self, unet: UNet2DConditionModel):
        pass

    def unpatch_unet(self, unet: UNet2DConditionModel):
        pass