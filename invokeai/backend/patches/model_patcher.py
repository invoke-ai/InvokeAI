from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.patches.pad_with_zeros import pad_with_zeros
from invokeai.backend.patches.sidecar_wrappers.base_sidecar_wrapper import BaseSidecarWrapper
from invokeai.backend.patches.sidecar_wrappers.utils import wrap_module_with_sidecar_wrapper
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class LayerPatcher:
    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_model_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[ModelPatchRaw, float]],
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
                LayerPatcher.apply_model_patch(
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
                cur_param = model.get_parameter(param_key)
                cur_param.data = weight.to(dtype=cur_param.dtype, device=cur_param.device, copy=True)

    @staticmethod
    @torch.no_grad()
    def apply_model_patch(
        model: torch.nn.Module,
        prefix: str,
        patch: ModelPatchRaw,
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

            module_key, module = LayerPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            LayerPatcher._apply_model_layer_patch(
                module_to_patch=module,
                module_to_patch_key=module_key,
                patch=layer,
                patch_weight=patch_weight,
                original_weights=original_weights,
            )

    @staticmethod
    @torch.no_grad()
    def _apply_model_layer_patch(
        module_to_patch: torch.nn.Module,
        module_to_patch_key: str,
        patch: BaseLayerPatch,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
    ):
        # All of the LoRA weight calculations will be done on the same device as the module weight.
        # (Performance will be best if this is a CUDA device.)
        first_param = next(module_to_patch.parameters())
        device = first_param.device
        dtype = first_param.dtype

        # We intentionally move to the target device first, then cast. Experimentally, this was found to
        # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
        # same thing in a single call to '.to(...)'.
        patch.to(device=device)
        patch.to(dtype=torch.float32)

        # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
        # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
        for param_name, param_weight in patch.get_parameters(module_to_patch, weight=patch_weight).items():
            param_key = module_to_patch_key + "." + param_name
            module_param = module_to_patch.get_parameter(param_name)

            # Save original weight
            original_weights.save(param_key, module_param)

            # HACK(ryand): This condition is only necessary to handle layers in FLUX control LoRAs that change the
            # shape of the original layer.
            if module_param.nelement() != param_weight.nelement():
                assert isinstance(patch, FluxControlLoRALayer)
                expanded_weight = pad_with_zeros(module_param, param_weight.shape)
                setattr(
                    module_to_patch,
                    param_name,
                    torch.nn.Parameter(expanded_weight, requires_grad=module_param.requires_grad),
                )
                module_param = expanded_weight

            module_param += param_weight.to(dtype=dtype)

        patch.to(device=TorchDevice.CPU_DEVICE)

    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_model_sidecar_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[ModelPatchRaw, float]],
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
                LayerPatcher._apply_model_sidecar_patch(
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
                module_parent_key, module_name = LayerPatcher._split_parent_key(module_key)
                parent_module = model.get_submodule(module_parent_key)
                LayerPatcher._set_submodule(parent_module, module_name, orig_module)

    @staticmethod
    def _apply_model_sidecar_patch(
        model: torch.nn.Module,
        patch: ModelPatchRaw,
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

            module_key, module = LayerPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            LayerPatcher._apply_model_layer_wrapper_patch(
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
    def _apply_model_layer_wrapper_patch(
        model: torch.nn.Module,
        module_to_patch: torch.nn.Module,
        module_to_patch_key: str,
        patch: BaseLayerPatch,
        patch_weight: float,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA wrapper patch to a model."""
        # Replace the original module with a BaseSidecarWrapper if it has not already been done.
        if not isinstance(module_to_patch, BaseSidecarWrapper):
            wrapped_module = wrap_module_with_sidecar_wrapper(orig_module=module_to_patch)
            original_modules[module_to_patch_key] = module_to_patch
            module_parent_key, module_name = LayerPatcher._split_parent_key(module_to_patch_key)
            module_parent = model.get_submodule(module_parent_key)
            LayerPatcher._set_submodule(module_parent, module_name, wrapped_module)
        else:
            assert module_to_patch_key in original_modules
            wrapped_module = module_to_patch

        # Move the LoRA layer to the same device/dtype as the orig module.
        first_param = next(module_to_patch.parameters())
        device = first_param.device
        patch.to(device=device, dtype=dtype)

        # Add the patch to the sidecar wrapper.
        wrapped_module.add_patch(patch, patch_weight)

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
