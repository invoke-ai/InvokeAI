import warnings
import weakref
from collections.abc import MutableMapping
from typing import Optional, Callable

import torch
from accelerate.utils import send_to_device
from torch.utils.hooks import RemovableHandle


class OffloadingDevice:
    _hooks: MutableMapping[torch.nn.Module, RemovableHandle]
    _current_model_ref: Callable[[], Optional[torch.nn.Module]]

    def __init__(self, execution_device: torch.device):
        self.execution_device = execution_device
        self._hooks = weakref.WeakKeyDictionary()
        self._current_model_ref = lambda: None

    def install(self, *models: torch.nn.Module):
        for model in models:
            self._hooks[model] = model.register_forward_pre_hook(self._pre_hook)

    def uninstall(self, *models: torch.nn.Module):
        for model in models:
            hook = self._hooks.pop(model)
            hook.remove()
            if self.is_current_model(model):
                # no longer hooked by this object, so don't claim to manage it
                self.clear_current_model()

    def _pre_hook(self, module: torch.nn.Module, forward_input):
        self.load(module)
        if len(forward_input) == 0:
            warnings.warn(f"Hook for {module.__class__.__name__} got no input. "
                          f"Inputs must be positional, not keywords.", stacklevel=3)
        return send_to_device(forward_input, self.execution_device)

    def load(self, module):
        if not self.is_current_model(module):
            self.offload_current()
            self._load(module)

    def offload_current(self) -> torch.nn.Module:
        # noinspection PyNoneFunctionAssignment
        module: Optional[torch.nn.Module] = self._current_model_ref()
        if module is not None:
            module.cpu()
        self.clear_current_model()
        return module

    def _load(self, module: torch.nn.Module) -> torch.nn.Module:
        assert self.is_empty(), f"A model is already loaded: {self._current_model_ref()}"
        module = module.to(self.execution_device)
        self.set_current_model(module)
        return module

    def is_current_model(self, model: torch.nn.Module) -> bool:
        return self._current_model_ref() is model

    def is_empty(self):
        return self._current_model_ref() is None

    def set_current_model(self, value):
        self._current_model_ref = weakref.ref(value)

    def clear_current_model(self):
        self._current_model_ref = lambda: None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {id(self):x}: " \
               f"current_model={type(self._current_model_ref()).__name__} >"
