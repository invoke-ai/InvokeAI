from __future__ import annotations

import warnings
import weakref
from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from typing import Callable, Union

import torch
from accelerate.utils import send_to_device
from torch.utils.hooks import RemovableHandle

OFFLOAD_DEVICE = torch.device("cpu")


class _NoModel:
    """Symbol that indicates no model is loaded.

    (We can't weakref.ref(None), so this was my best idea at the time to come up with something
    type-checkable.)
    """

    def __bool__(self):
        return False

    def to(self, device: torch.device):
        pass

    def __repr__(self):
        return "<NO MODEL>"


NO_MODEL = _NoModel()


class ModelGroup(metaclass=ABCMeta):
    """
    A group of models.

    The use case I had in mind when writing this is the sub-models used by a DiffusionPipeline,
    e.g. its text encoder, U-net, VAE, etc.

    Those models are :py:class:`diffusers.ModelMixin`, but "model" is interchangeable with
    :py:class:`torch.nn.Module` here.
    """

    def __init__(self, execution_device: torch.device):
        self.execution_device = execution_device

    @abstractmethod
    def install(self, *models: torch.nn.Module):
        """Add models to this group."""
        pass

    @abstractmethod
    def uninstall(self, models: torch.nn.Module):
        """Remove models from this group."""
        pass

    @abstractmethod
    def uninstall_all(self):
        """Remove all models from this group."""

    @abstractmethod
    def load(self, model: torch.nn.Module):
        """Load this model to the execution device."""
        pass

    @abstractmethod
    def offload_current(self):
        """Offload the current model(s) from the execution device."""
        pass

    @abstractmethod
    def ready(self):
        """Ready this group for use."""
        pass

    @abstractmethod
    def set_device(self, device: torch.device):
        """Change which device models from this group will execute on."""
        pass

    @abstractmethod
    def device_for(self, model) -> torch.device:
        """Get the device the given model will execute on.

        The model should already be a member of this group.
        """
        pass

    @abstractmethod
    def __contains__(self, model):
        """Check if the model is a member of this group."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {id(self):x}: " f"device={self.execution_device} >"


class LazilyLoadedModelGroup(ModelGroup):
    """
    Only one model from this group is loaded on the GPU at a time.

    Running the forward method of a model will displace the previously-loaded model,
    offloading it to CPU.

    If you call other methods on the model, e.g. ``model.encode(x)`` instead of ``model(x)``,
    you will need to explicitly load it with :py:method:`.load(model)`.

    This implementation relies on pytorch forward-pre-hooks, and it will copy forward arguments
    to the appropriate execution device, as long as they are positional arguments and not keyword
    arguments. (I didn't make the rules; that's the way the pytorch 1.13 API works for hooks.)
    """

    _hooks: MutableMapping[torch.nn.Module, RemovableHandle]
    _current_model_ref: Callable[[], Union[torch.nn.Module, _NoModel]]

    def __init__(self, execution_device: torch.device):
        super().__init__(execution_device)
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
            warnings.warn(
                f"Hook for {module.__class__.__name__} got no input. " f"Inputs must be positional, not keywords.",
                stacklevel=3,
            )
        return send_to_device(forward_input, self.execution_device)

    def load(self, module):
        if not self.is_current_model(module):
            self.offload_current()
            self._load(module)

    def offload_current(self):
        module = self._current_model_ref()
        if module is not NO_MODEL:
            module.to(OFFLOAD_DEVICE)
        self.clear_current_model()

    def _load(self, module: torch.nn.Module) -> torch.nn.Module:
        assert self.is_empty(), f"A model is already loaded: {self._current_model_ref()}"
        module = module.to(self.execution_device)
        self.set_current_model(module)
        return module

    def is_current_model(self, model: torch.nn.Module) -> bool:
        """Is the given model the one currently loaded on the execution device?"""
        return self._current_model_ref() is model

    def is_empty(self):
        """Are none of this group's models loaded on the execution device?"""
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
            raise KeyError(f"This does not manage this model {type(model).__name__}", model)
        return self.execution_device  # this implementation only dispatches to one device

    def ready(self):
        pass  # always ready to load on-demand

    def __contains__(self, model):
        return model in self._hooks

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} object at {id(self):x}: "
            f"current_model={type(self._current_model_ref()).__name__} >"
        )


class FullyLoadedModelGroup(ModelGroup):
    """
    A group of models without any implicit loading or unloading.

    :py:meth:`.ready` loads _all_ the models to the execution device at once.
    """

    _models: weakref.WeakSet

    def __init__(self, execution_device: torch.device):
        super().__init__(execution_device)
        self._models = weakref.WeakSet()

    def install(self, *models: torch.nn.Module):
        for model in models:
            self._models.add(model)
            model.to(self.execution_device)

    def uninstall(self, *models: torch.nn.Module):
        for model in models:
            self._models.remove(model)

    def uninstall_all(self):
        self.uninstall(*self._models)

    def load(self, model):
        model.to(self.execution_device)

    def offload_current(self):
        for model in self._models:
            model.to(OFFLOAD_DEVICE)

    def ready(self):
        for model in self._models:
            self.load(model)

    def set_device(self, device: torch.device):
        self.execution_device = device
        for model in self._models:
            if model.device != OFFLOAD_DEVICE:
                model.to(device)

    def device_for(self, model):
        if model not in self:
            raise KeyError("This does not manage this model f{type(model).__name__}", model)
        return self.execution_device  # this implementation only dispatches to one device

    def __contains__(self, model):
        return model in self._models
