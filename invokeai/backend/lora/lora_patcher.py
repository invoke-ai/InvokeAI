from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.lora_layer_wrappers import (
    LoRAConv1dWrapper,
    LoRAConv2dWrapper,
    LoRALinearWrapper,
    LoRASidecarWrapper,
)
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class LoRAPatcher:
    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_smart_lora_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[LoRAModelRaw, float]],
        prefix: str,
        dtype: torch.dtype,
        cached_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Apply 'smart' LoRA patching that chooses whether to use direct patching or a sidecar wrapper for each module."""

        # original_weights are stored for unpatching layers that are directly patched.
        original_weights = OriginalWeightsStorage(cached_weights)
        # original_modules are stored for unpatching layers that are wrapped in a LoRASidecarWrapper.
        original_modules: dict[str, torch.nn.Module] = {}
        try:
            for patch, patch_weight in patches:
                LoRAPatcher._apply_smart_lora_patch(
                    model=model,
                    prefix=prefix,
                    patch=patch,
                    patch_weight=patch_weight,
                    original_weights=original_weights,
                    original_modules=original_modules,
                    dtype=dtype,
                )

            yield
        finally:
            # Restore directly patched layers.
            for param_key, weight in original_weights.get_changed_weights():
                model.get_parameter(param_key).copy_(weight)

            # Restore LoRASidecarWrapper modules.
            # Note: This logic assumes no nested modules in original_modules.
            for module_key, orig_module in original_modules.items():
                module_parent_key, module_name = LoRAPatcher._split_parent_key(module_key)
                parent_module = model.get_submodule(module_parent_key)
                LoRAPatcher._set_submodule(parent_module, module_name, orig_module)

    @staticmethod
    @torch.no_grad()
    def _apply_smart_lora_patch(
        model: torch.nn.Module,
        prefix: str,
        patch: LoRAModelRaw,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA patch to a model using the 'smart' patching strategy that chooses whether to use direct
        patching or a sidecar wrapper for each module.
        """
        if patch_weight == 0:
            return

        # If the layer keys contain a dot, then they are not flattened, and can be directly used to access model
        # submodules. If the layer keys do not contain a dot, then they are flattened, meaning that all '.' have been
        # replaced with '_'. Non-flattened keys are preferred, because they allow submodules to be accessed directly
        # without searching, but some legacy code still uses flattened keys.
        layer_keys_are_flattened = "." not in next(iter(patch.layers.keys()))

        prefix_len = len(prefix)

        for layer_key, layer in patch.layers.items():
            if not layer_key.startswith(prefix):
                continue

            module_key, module = LoRAPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            # Decide whether to use direct patching or a sidecar wrapper.
            # Direct patching is preferred, because it results in better runtime speed.
            # Reasons to use sidecar patching:
            # - The module is already wrapped in a LoRASidecarWrapper.
            # - The module is quantized.
            # - The module is on the CPU (and we don't want to store a second full copy of the original weights on the
            #   CPU, since this would double the RAM usage)
            # NOTE: For now, we don't check if the layer is quantized here. We assume that this is checked in the caller
            # and that the caller will use the 'apply_lora_wrapper_patches' method if the layer is quantized.
            # TODO(ryand): Handle the case where we are running without a GPU. Should we set a config flag that allows
            # forcing full patching even on the CPU?
            if isinstance(module, LoRASidecarWrapper) or LoRAPatcher._is_any_part_of_layer_on_cpu(module):
                LoRAPatcher._apply_lora_layer_wrapper_patch(
                    model=model,
                    module_to_patch=module,
                    module_to_patch_key=module_key,
                    patch=layer,
                    patch_weight=patch_weight,
                    original_modules=original_modules,
                    dtype=dtype,
                )
            else:
                LoRAPatcher._apply_lora_layer_patch(
                    module_to_patch=module,
                    module_to_patch_key=module_key,
                    patch=layer,
                    patch_weight=patch_weight,
                    original_weights=original_weights,
                )

    @staticmethod
    def _is_any_part_of_layer_on_cpu(layer: torch.nn.Module) -> bool:
        return any(p.device.type == "cpu" for p in layer.parameters())

    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_lora_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[LoRAModelRaw, float]],
        prefix: str,
        cached_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Apply one or more LoRA patches to a model within a context manager.

        Args:
            model (torch.nn.Module): The model to patch.
            patches (Iterable[Tuple[LoRAModelRaw, float]]): An iterator that returns tuples of LoRA patches and
                associated weights. An iterator is used so that the LoRA patches do not need to be loaded into memory
                all at once.
            prefix (str): The keys in the patches will be filtered to only include weights with this prefix.
            cached_weights (Optional[Dict[str, torch.Tensor]], optional): Read-only copy of the model's state dict in
                CPU RAM, for efficient unpatching purposes.
        """
        original_weights = OriginalWeightsStorage(cached_weights)
        try:
            for patch, patch_weight in patches:
                LoRAPatcher._apply_lora_patch(
                    model=model,
                    prefix=prefix,
                    patch=patch,
                    patch_weight=patch_weight,
                    original_weights=original_weights,
                )
                del patch

            yield
        finally:
            for param_key, weight in original_weights.get_changed_weights():
                model.get_parameter(param_key).copy_(weight)

    @staticmethod
    @torch.no_grad()
    def _apply_lora_patch(
        model: torch.nn.Module,
        prefix: str,
        patch: LoRAModelRaw,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
    ):
        """Apply a single LoRA patch to a model.

        Args:
            model (torch.nn.Module): The model to patch.
            prefix (str): A string prefix that precedes keys used in the LoRAs weight layers.
            patch (LoRAModelRaw): The LoRA model to patch in.
            patch_weight (float): The weight of the LoRA patch.
            original_weights (OriginalWeightsStorage): Storage for the original weights of the model, for unpatching.
        """
        if patch_weight == 0:
            return

        # If the layer keys contain a dot, then they are not flattened, and can be directly used to access model
        # submodules. If the layer keys do not contain a dot, then they are flattened, meaning that all '.' have been
        # replaced with '_'. Non-flattened keys are preferred, because they allow submodules to be accessed directly
        # without searching, but some legacy code still uses flattened keys.
        layer_keys_are_flattened = "." not in next(iter(patch.layers.keys()))

        prefix_len = len(prefix)

        for layer_key, layer in patch.layers.items():
            if not layer_key.startswith(prefix):
                continue

            module_key, module = LoRAPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            LoRAPatcher._apply_lora_layer_patch(
                module_to_patch=module,
                module_to_patch_key=module_key,
                patch=layer,
                patch_weight=patch_weight,
                original_weights=original_weights,
            )

    @staticmethod
    @torch.no_grad()
    def _apply_lora_layer_patch(
        module_to_patch: torch.nn.Module,
        module_to_patch_key: str,
        patch: AnyLoRALayer,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
    ):
        # All of the LoRA weight calculations will be done on the same device as the module weight.
        # (Performance will be best if this is a CUDA device.)
        device = module_to_patch.weight.device
        dtype = module_to_patch.weight.dtype

        layer_scale = patch.scale()

        # We intentionally move to the target device first, then cast. Experimentally, this was found to
        # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
        # same thing in a single call to '.to(...)'.
        patch.to(device=device)
        patch.to(dtype=torch.float32)

        # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
        # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
        for param_name, lora_param_weight in patch.get_parameters(module_to_patch).items():
            param_key = module_to_patch_key + "." + param_name
            module_param = module_to_patch.get_parameter(param_name)

            # Save original weight
            original_weights.save(param_key, module_param)

            if module_param.shape != lora_param_weight.shape:
                lora_param_weight = lora_param_weight.reshape(module_param.shape)

            lora_param_weight *= patch_weight * layer_scale
            module_param += lora_param_weight.to(dtype=dtype)

        patch.to(device=TorchDevice.CPU_DEVICE)

    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_lora_wrapper_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[LoRAModelRaw, float]],
        prefix: str,
        dtype: torch.dtype,
    ):
        """Apply one or more LoRA wrapper patches to a model within a context manager. Wrapper patches incur some
        runtime overhead compared to normal LoRA patching, but they enable:
        - LoRA layers to be applied to quantized models
        - LoRA layers to be applied to CPU layers without needing to store a full copy of the original weights (i.e.
          avoid doubling the memory requirements).

        Args:
            model (torch.nn.Module): The model to patch.
            patches (Iterable[Tuple[LoRAModelRaw, float]]): An iterator that returns tuples of LoRA patches and
                associated weights. An iterator is used so that the LoRA patches do not need to be loaded into memory
                all at once.
            prefix (str): The keys in the patches will be filtered to only include weights with this prefix.
        """
        original_modules: dict[str, torch.nn.Module] = {}
        try:
            for patch, patch_weight in patches:
                LoRAPatcher._apply_lora_wrapper_patch(
                    model=model,
                    prefix=prefix,
                    patch=patch,
                    patch_weight=patch_weight,
                    original_modules=original_modules,
                    dtype=dtype,
                )
            yield
        finally:
            # Restore original modules.
            # Note: This logic assumes no nested modules in original_modules.
            for module_key, orig_module in original_modules.items():
                module_parent_key, module_name = LoRAPatcher._split_parent_key(module_key)
                parent_module = model.get_submodule(module_parent_key)
                LoRAPatcher._set_submodule(parent_module, module_name, orig_module)

    @staticmethod
    def _apply_lora_wrapper_patch(
        model: torch.nn.Module,
        patch: LoRAModelRaw,
        patch_weight: float,
        prefix: str,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA wrapper patch to a model."""

        if patch_weight == 0:
            return

        # If the layer keys contain a dot, then they are not flattened, and can be directly used to access model
        # submodules. If the layer keys do not contain a dot, then they are flattened, meaning that all '.' have been
        # replaced with '_'. Non-flattened keys are preferred, because they allow submodules to be accessed directly
        # without searching, but some legacy code still uses flattened keys.
        layer_keys_are_flattened = "." not in next(iter(patch.layers.keys()))

        prefix_len = len(prefix)

        for layer_key, layer in patch.layers.items():
            if not layer_key.startswith(prefix):
                continue

            module_key, module = LoRAPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            LoRAPatcher._apply_lora_layer_wrapper_patch(
                model=model,
                module_to_patch=module,
                module_to_patch_key=module_key,
                patch=layer,
                patch_weight=patch_weight,
                original_modules=original_modules,
                dtype=dtype,
            )

    @staticmethod
    @torch.no_grad()
    def _apply_lora_layer_wrapper_patch(
        model: torch.nn.Module,
        module_to_patch: torch.nn.Module,
        module_to_patch_key: str,
        patch: AnyLoRALayer,
        patch_weight: float,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA wrapper patch to a model."""

        # Replace the original module with a LoRASidecarWrapper if it has not already been done.
        if not isinstance(module_to_patch, LoRASidecarWrapper):
            lora_wrapper_layer = LoRAPatcher._initialize_lora_wrapper_layer(module_to_patch)
            original_modules[module_to_patch_key] = module_to_patch
            module_parent_key, module_name = LoRAPatcher._split_parent_key(module_to_patch_key)
            module_parent = model.get_submodule(module_parent_key)
            LoRAPatcher._set_submodule(module_parent, module_name, lora_wrapper_layer)
            orig_module = module_to_patch
        else:
            assert module_to_patch_key in original_modules
            lora_wrapper_layer = module_to_patch
            orig_module = module_to_patch.orig_module

        # Move the LoRA layer to the same device/dtype as the orig module.
        patch.to(device=orig_module.weight.device, dtype=dtype)

        # Add the LoRA wrapper layer to the LoRASidecarWrapper.
        lora_wrapper_layer.add_lora_layer(patch, patch_weight)

    @staticmethod
    def _split_parent_key(module_key: str) -> tuple[str, str]:
        """Split a module key into its parent key and module name.

        Args:
            module_key (str): The module key to split.

        Returns:
            tuple[str, str]: A tuple containing the parent key and module name.
        """
        split_key = module_key.rsplit(".", 1)
        if len(split_key) == 2:
            return tuple(split_key)
        elif len(split_key) == 1:
            return "", split_key[0]
        else:
            raise ValueError(f"Invalid module key: {module_key}")

    @staticmethod
    def _initialize_lora_wrapper_layer(orig_layer: torch.nn.Module):
        if isinstance(orig_layer, torch.nn.Linear):
            return LoRALinearWrapper(orig_layer, [], [])
        elif isinstance(orig_layer, torch.nn.Conv1d):
            return LoRAConv1dWrapper(orig_layer, [], [])
        elif isinstance(orig_layer, torch.nn.Conv2d):
            return LoRAConv2dWrapper(orig_layer, [], [])
        else:
            raise ValueError(f"Unsupported layer type: {type(orig_layer)}")

    @staticmethod
    def _set_submodule(parent_module: torch.nn.Module, module_name: str, submodule: torch.nn.Module):
        try:
            submodule_index = int(module_name)
            # If the module name is an integer, then we use the __setitem__ method to set the submodule.
            parent_module[submodule_index] = submodule  # type: ignore
        except ValueError:
            # If the module name is not an integer, then we use the setattr method to set the submodule.
            setattr(parent_module, module_name, submodule)

    @staticmethod
    def _get_submodule(
        model: torch.nn.Module, layer_key: str, layer_key_is_flattened: bool
    ) -> tuple[str, torch.nn.Module]:
        """Get the submodule corresponding to the given layer key.

        Args:
            model (torch.nn.Module): The model to search.
            layer_key (str): The layer key to search for.
            layer_key_is_flattened (bool): Whether the layer key is flattened. If flattened, then all '.' have been
                replaced with '_'. Non-flattened keys are preferred, because they allow submodules to be accessed
                directly without searching, but some legacy code still uses flattened keys.

        Returns:
            tuple[str, torch.nn.Module]: A tuple containing the module key and the submodule.
        """
        if not layer_key_is_flattened:
            return layer_key, model.get_submodule(layer_key)

        # Handle flattened keys.
        assert "." not in layer_key

        module = model
        module_key = ""
        key_parts = layer_key.split("_")

        submodule_name = key_parts.pop(0)

        while len(key_parts) > 0:
            try:
                module = module.get_submodule(submodule_name)
                module_key += "." + submodule_name
                submodule_name = key_parts.pop(0)
            except Exception:
                submodule_name += "_" + key_parts.pop(0)

        module = module.get_submodule(submodule_name)
        module_key = (module_key + "." + submodule_name).lstrip(".")

        return module_key, module
