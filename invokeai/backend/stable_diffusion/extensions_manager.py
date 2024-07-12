from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict

import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.util.devices import TorchDevice

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extensions import ExtensionBase


class ExtCallbacksApi(ABC):
    @abstractmethod
    def setup(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def pre_denoise_loop(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def post_denoise_loop(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def pre_step(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def post_step(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def pre_unet(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def post_unet(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass

    @abstractmethod
    def post_apply_cfg(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        pass


class ProxyCallsClass:
    def __init__(self, handler):
        self._handler = handler

    def __getattr__(self, item):
        return partial(self._handler, item)


class CallbackInjectionPoint:
    def __init__(self):
        self.handlers = {}

    def add(self, func: Callable, order: int):
        if order not in self.handlers:
            self.handlers[order] = []
        self.handlers[order].append(func)

    def __call__(self, *args, **kwargs):
        for order in sorted(self.handlers.keys(), reverse=True):
            for handler in self.handlers[order]:
                handler(*args, **kwargs)


class ExtensionsManager:
    def __init__(self):
        self.extensions = []

        self._callbacks = {}
        self.callbacks: ExtCallbacksApi = ProxyCallsClass(self.call_callback)

    def add_extension(self, ext: ExtensionBase):
        self.extensions.append(ext)

        self._callbacks.clear()

        for ext in self.extensions:
            for inj_info in ext.injections:
                if inj_info.type == "callback":
                    if inj_info.name not in self._callbacks:
                        self._callbacks[inj_info.name] = CallbackInjectionPoint()
                    self._callbacks[inj_info.name].add(inj_info.function, inj_info.order)

                else:
                    raise Exception(f"Unsupported injection type: {inj_info.type}")

    def call_callback(self, name: str, *args, **kwargs):
        if name in self._callbacks:
            self._callbacks[name](*args, **kwargs)

    # TODO: is there any need in such high abstarction
    # @contextmanager
    # def patch_extensions(self):
    #    exit_stack = ExitStack()
    #    try:
    #        for ext in self.extensions:
    #            exit_stack.enter_context(ext.patch_extension(self))
    #
    #        yield None
    #
    #    finally:
    #        exit_stack.close()

    @contextmanager
    def patch_attention_processor(self, unet: UNet2DConditionModel, attn_processor_cls: object):
        unet_orig_processors = unet.attn_processors
        exit_stack = ExitStack()
        try:
            # just to be sure that attentions have not same processor instance
            attn_procs = {}
            for name in unet.attn_processors.keys():
                attn_procs[name] = attn_processor_cls()
            unet.set_attn_processor(attn_procs)

            for ext in self.extensions:
                exit_stack.enter_context(ext.patch_attention_processor(attn_processor_cls))

            yield None

        finally:
            unet.set_attn_processor(unet_orig_processors)
            exit_stack.close()

    @contextmanager
    def patch_unet(self, state_dict: Dict[str, torch.Tensor], unet: UNet2DConditionModel):
        exit_stack = ExitStack()
        try:
            changed_keys = set()
            changed_unknown_keys = {}

            for ext in self.extensions:
                patch_result = exit_stack.enter_context(ext.patch_unet(state_dict, unet))
                if patch_result is None:
                    continue
                new_keys, new_unk_keys = patch_result
                changed_keys.update(new_keys)
                # skip already seen keys, as new weight might be changed
                for k, v in new_unk_keys.items():
                    if k in changed_unknown_keys:
                        continue
                    changed_unknown_keys[k] = v

            yield None

        finally:
            exit_stack.close()
            assert hasattr(unet, "get_submodule")  # mypy not picking up fact that torch.nn.Module has get_submodule()
            with torch.no_grad():
                for module_key in changed_keys:
                    weight = state_dict[module_key]
                    unet.get_submodule(module_key).weight.copy_(
                        weight, non_blocking=TorchDevice.get_non_blocking(weight.device)
                    )
                for module_key, weight in changed_unknown_keys.items():
                    unet.get_submodule(module_key).weight.copy_(
                        weight, non_blocking=TorchDevice.get_non_blocking(weight.device)
                    )
