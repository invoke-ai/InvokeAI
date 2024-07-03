from abc import ABC
from contextlib import contextmanager
from typing import Callable
from functools import partial

from diffusers import UNet2DConditionModel
from .denoise_context import DenoiseContext
from .extensions import ExtensionBase

class ExtModifiersApi(ABC):
    def pre_denoise_loop(self, ctx: DenoiseContext):
        pass

    def post_denoise_loop(self, ctx: DenoiseContext):
        pass

    def pre_step(self, ctx: DenoiseContext):
        pass

    def post_step(self, ctx: DenoiseContext):
        pass

    def modify_noise_prediction(self, ctx: DenoiseContext):
        pass

    def pre_unet_forward(self, ctx: DenoiseContext):
        pass

class ExtOverridesApi(ABC):
    def step(self, orig_func, ctx: DenoiseContext):
        pass

    def combine_noise(self, orig_func: Callable, ctx: DenoiseContext):
        pass

class ProxyCallsClass:
    def __init__(self, handler):
        self._handler = handler
    def __getattr__(self, item):
        return partial(self._handler, item)

class ModifierInjectionPoint:
    def __init__(self):
        self.first = []
        self.any = []
        self.last = []

    def add(self, func: Callable, order: str):
        if order == "first":
            self.first.append(func)
        elif order == "last":
            self.last.append(func)
        else: # elif order == "any":
            self.any.append(func)

    def __call__(self, *args, **kwargs):
        for func in self.first:
            func(*args, **kwargs)
        for func in self.any:
            func(*args, **kwargs)
        for func in reversed(self.last):
            func(*args, **kwargs)


class ExtensionsManager:
    def __init__(self):
        self.extensions = []

        self._overrides = dict()
        self._modifiers = dict()

        self.modifiers: ExtModifiersApi = ProxyCallsClass(self.call_modifier)
        self.overrides: ExtOverridesApi = ProxyCallsClass(self.call_override)


    def add_extension(self, ext: ExtensionBase):
        self.extensions.append(ext)
        ordered_extensions = sorted(self.extensions, reverse=True, key=lambda ext: ext.priority)

        self._overrides.clear()
        self._modifiers.clear()

        for ext in ordered_extensions:
            for inj_info in ext.injections:
                if inj_info.type == "modifier":
                    if inj_info.name not in self._modifiers:
                        self._modifiers[inj_info.name] = ModifierInjectionPoint()
                    self._modifiers[inj_info.name].add(inj_info.function, inj_info.order)

                else:
                    if inj_info.name in self._overrides:
                        raise Exception(f"Already overloaded - {inj_info.name}")
                    self._overrides[inj_info.name] = inj_info.function

    def call_modifier(self, name: str, ctx: DenoiseContext):
        if name in self._modifiers:
            self._modifiers[name](ctx)

    def call_override(self, name: str, orig_func: Callable, ctx: DenoiseContext):
        if name in self._overrides:
            return self._overrides[name](orig_func, ctx)
        else:
            return orig_func(ctx, self)

    @contextmanager
    def patch_attention_processor(self, unet: UNet2DConditionModel, attn_processor_cls: object):
        unet_orig_processors = unet.attn_processors
        patched_extensions = []
        try:
            # just to be sure that attentions have not same processor instance
            attn_procs = dict()
            for idx, name in enumerate(unet.attn_processors.keys()):
                attn_procs[name] = attn_processor_cls()
            unet.set_attn_processor(attn_procs)

            for ext in self.extensions:
                ext.apply_attention_processor(attn_processor_cls)
                patched_extensions.append(ext)

            yield None

        finally:
            unet.set_attn_processor(unet_orig_processors)
            for ext in patched_extensions:
                ext.restore_attention_processor()

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel):
        _extensions = []
        try:
            ordered_extensions = sorted(self.extensions, reverse=True, key=lambda ext: ext.priority)
            for ext in ordered_extensions:
                ext.patch_unet(unet)
                _extensions.append(ext)

            yield None

        finally:
            for ext in _extensions:
                ext.unpatch_unet(unet)
