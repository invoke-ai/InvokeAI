from __future__ import annotations

import warnings
import weakref
from collections.abc import MutableMapping
from typing import Callable

import torch
from accelerate.utils import send_to_device
from torch.utils.hooks import RemovableHandle

OFFLOAD_DEVICE = torch.device("cpu")

class _NoModel:
    def __bool__(self):
        return False

    def to(self, device: torch.device):
        pass

NO_MODEL = _NoModel()


class HotSeatModelGroup:
    _hooks: MutableMapping[torch.nn.Module, RemovableHandle]
    _current_model_ref: Callable[[], torch.nn.Module | _NoModel]

    def __init__(self, execution_device: torch.device):
        self.execution_device = execution_device
        self._hooks = weakref.WeakKeyDictionary()
        self._current_model_ref = weakref.ref(NO_MODEL)

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

    def uninstall_all(self):
        self.uninstall(*self._hooks.keys())

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

    def offload_current(self):
        module = self._current_model_ref()
        if module is not NO_MODEL:
            module.to(device=OFFLOAD_DEVICE)
        self.clear_current_model()

    def _load(self, module: torch.nn.Module) -> torch.nn.Module:
        assert self.is_empty(), f"A model is already loaded: {self._current_model_ref()}"
        module = module.to(self.execution_device)
        self.set_current_model(module)
        return module

    def is_current_model(self, model: torch.nn.Module) -> bool:
        return self._current_model_ref() is model

    def is_empty(self):
        return self._current_model_ref() is NO_MODEL

    def set_current_model(self, value):
        self._current_model_ref = weakref.ref(value)

    def clear_current_model(self):
        self._current_model_ref = weakref.ref(NO_MODEL)

    def set_device(self, device: torch.device):
        if device == self.execution_device:
            return
        self.execution_device = device
        current = self._current_model_ref()
        if current is not NO_MODEL:
            current.to(device)

    def device_for(self, model):
        if model not in self:
            raise KeyError("This does not manage this model f{type(model).__name__}", model)
        return self.execution_device  # this implementation only dispatches to one device

    def ready(self):
        pass  # always ready to load on-demand

    def __contains__(self, model):
        return model in self._hooks

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {id(self):x}: " \
               f"current_model={type(self._current_model_ref()).__name__} >"


class SimpleModelGroup:
    _models: weakref.WeakSet

    def __init__(self, execution_device: torch.device):
        self.execution_device = execution_device
        self._models = weakref.WeakSet()

    def install(self, *models: torch.nn.Module):
        for model in models:
            self._models.add(model)
            model.to(device=self.execution_device)

    def uninstall(self, *models: torch.nn.Module):
        for model in models:
            self._models.remove(model)

    def uninstall_all(self):
        self.uninstall(*self._models)

    def load(self, model):
        model.to(device=self.execution_device)

    def offload_current(self):
        for model in self._models:
            model.to(device=OFFLOAD_DEVICE)

    def ready(self):
        for model in self._models:
            self.load(model)

    def set_device(self, device: torch.device):
        self.execution_device = device
        for model in self._models:
            if model.device != OFFLOAD_DEVICE:
                model.to(device=device)

    def device_for(self, model):
        if model not in self:
            raise KeyError("This does not manage this model f{type(model).__name__}", model)
        return self.execution_device  # this implementation only dispatches to one device

    def __contains__(self, model):
        return model in self._models
