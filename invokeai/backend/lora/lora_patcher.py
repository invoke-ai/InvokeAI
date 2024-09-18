from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.sidecar_layers.concatenated_lora.concatenated_lora_linear_sidecar_layer import (
    ConcatenatedLoRALinearSidecarLayer,
)
from invokeai.backend.lora.sidecar_layers.lora.lora_linear_sidecar_layer import LoRALinearSidecarLayer
from invokeai.backend.lora.sidecar_layers.lora_sidecar_module import LoRASidecarModule
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class LoRAPatcher:
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
                LoRAPatcher.apply_lora_patch(
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
    def apply_lora_patch(
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

            # All of the LoRA weight calculations will be done on the same device as the module weight.
            # (Performance will be best if this is a CUDA device.)
            device = module.weight.device
            dtype = module.weight.dtype

            layer_scale = layer.scale()

            # We intentionally move to the target device first, then cast. Experimentally, this was found to
            # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
            # same thing in a single call to '.to(...)'.
            layer.to(device=device)
            layer.to(dtype=torch.float32)

            # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
            # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
            for param_name, lora_param_weight in layer.get_parameters(module).items():
                param_key = module_key + "." + param_name
                module_param = module.get_parameter(param_name)

                # Save original weight
                original_weights.save(param_key, module_param)

                if module_param.shape != lora_param_weight.shape:
                    lora_param_weight = lora_param_weight.reshape(module_param.shape)

                lora_param_weight *= patch_weight * layer_scale
                module_param += lora_param_weight.to(dtype=dtype)

            layer.to(device=TorchDevice.CPU_DEVICE)

    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_lora_sidecar_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[LoRAModelRaw, float]],
        prefix: str,
        dtype: torch.dtype,
    ):
        """Apply one or more LoRA sidecar patches to a model within a context manager. Sidecar patches incur some
        overhead compared to normal LoRA patching, but they allow for LoRA layers to applied to base layers in any
        quantization format.

        Args:
            model (torch.nn.Module): The model to patch.
            patches (Iterable[Tuple[LoRAModelRaw, float]]): An iterator that returns tuples of LoRA patches and
                associated weights. An iterator is used so that the LoRA patches do not need to be loaded into memory
                all at once.
            prefix (str): The keys in the patches will be filtered to only include weights with this prefix.
            dtype (torch.dtype): The compute dtype of the sidecar layers. This cannot easily be inferred from the model,
                since the sidecar layers are typically applied on top of quantized layers whose weight dtype is
                different from their compute dtype.
        """
        original_modules: dict[str, torch.nn.Module] = {}
        try:
            for patch, patch_weight in patches:
                LoRAPatcher._apply_lora_sidecar_patch(
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
    def _apply_lora_sidecar_patch(
        model: torch.nn.Module,
        patch: LoRAModelRaw,
        patch_weight: float,
        prefix: str,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA sidecar patch to a model."""

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

            # Initialize the LoRA sidecar layer.
            lora_sidecar_layer = LoRAPatcher._initialize_lora_sidecar_layer(module, layer, patch_weight)

            # Replace the original module with a LoRASidecarModule if it has not already been done.
            if module_key in original_modules:
                # The module has already been patched with a LoRASidecarModule. Append to it.
                assert isinstance(module, LoRASidecarModule)
                lora_sidecar_module = module
            else:
                # The module has not yet been patched with a LoRASidecarModule. Create one.
                lora_sidecar_module = LoRASidecarModule(module, [])
                original_modules[module_key] = module
                module_parent_key, module_name = LoRAPatcher._split_parent_key(module_key)
                module_parent = model.get_submodule(module_parent_key)
                LoRAPatcher._set_submodule(module_parent, module_name, lora_sidecar_module)

            # Move the LoRA sidecar layer to the same device/dtype as the orig module.
            # TODO(ryand): Experiment with moving to the device first, then casting. This could be faster.
            lora_sidecar_layer.to(device=lora_sidecar_module.orig_module.weight.device, dtype=dtype)

            # Add the LoRA sidecar layer to the LoRASidecarModule.
            lora_sidecar_module.add_lora_layer(lora_sidecar_layer)

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
    def _initialize_lora_sidecar_layer(orig_layer: torch.nn.Module, lora_layer: AnyLoRALayer, patch_weight: float):
        # TODO(ryand): Add support for more original layer types and LoRA layer types.
        if isinstance(orig_layer, torch.nn.Linear) or (
            isinstance(orig_layer, LoRASidecarModule) and isinstance(orig_layer.orig_module, torch.nn.Linear)
        ):
            if isinstance(lora_layer, LoRALayer):
                return LoRALinearSidecarLayer(lora_layer=lora_layer, weight=patch_weight)
            elif isinstance(lora_layer, ConcatenatedLoRALayer):
                return ConcatenatedLoRALinearSidecarLayer(concatenated_lora_layer=lora_layer, weight=patch_weight)
            else:
                raise ValueError(f"Unsupported Linear LoRA layer type: {type(lora_layer)}")
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
