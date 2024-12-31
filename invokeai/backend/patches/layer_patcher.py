from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.patches.pad_with_zeros import pad_with_zeros
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class LayerPatcher:
    @staticmethod
    @torch.no_grad()
    @contextmanager
    def apply_smart_model_patches(
        model: torch.nn.Module,
        patches: Iterable[Tuple[ModelPatchRaw, float]],
        prefix: str,
        dtype: torch.dtype,
        cached_weights: Optional[Dict[str, torch.Tensor]] = None,
        force_direct_patching: bool = False,
        force_sidecar_patching: bool = False,
    ):
        """Apply 'smart' model patching that chooses whether to use direct patching or a sidecar wrapper for each
        module.
        """

        # original_weights are stored for unpatching layers that are directly patched.
        original_weights = OriginalWeightsStorage(cached_weights)
        # original_modules are stored for unpatching layers that are wrapped.
        original_modules: dict[str, torch.nn.Module] = {}
        try:
            for patch, patch_weight in patches:
                LayerPatcher.apply_smart_model_patch(
                    model=model,
                    prefix=prefix,
                    patch=patch,
                    patch_weight=patch_weight,
                    original_weights=original_weights,
                    original_modules=original_modules,
                    dtype=dtype,
                    force_direct_patching=force_direct_patching,
                    force_sidecar_patching=force_sidecar_patching,
                )

            yield
        finally:
            # Restore directly patched layers.
            for param_key, weight in original_weights.get_changed_weights():
                cur_param = model.get_parameter(param_key)
                cur_param.data = weight.to(dtype=cur_param.dtype, device=cur_param.device, copy=True)

            # Clear patches from all patched modules.
            # Note: This logic assumes no nested modules in original_modules.
            for orig_module in original_modules.values():
                orig_module.clear_patches()

    @staticmethod
    @torch.no_grad()
    def apply_smart_model_patch(
        model: torch.nn.Module,
        prefix: str,
        patch: ModelPatchRaw,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
        force_direct_patching: bool,
        force_sidecar_patching: bool,
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

            module_key, module = LayerPatcher._get_submodule(
                model, layer_key[prefix_len:], layer_key_is_flattened=layer_keys_are_flattened
            )

            # Decide whether to use direct patching or a sidecar patch.
            # Direct patching is preferred, because it results in better runtime speed.
            # Reasons to use sidecar patching:
            # - The module is quantized, so the caller passed force_sidecar_patching=True.
            # - The module already has sidecar patches.
            # - The module is on the CPU (and we don't want to store a second full copy of the original weights on the
            #   CPU, since this would double the RAM usage)
            # NOTE: For now, we don't check if the layer is quantized here. We assume that this is checked in the caller
            # and that the caller will set force_sidecar_patching=True if the layer is quantized.
            # TODO(ryand): Handle the case where we are running without a GPU. Should we set a config flag that allows
            # forcing full patching even on the CPU?
            use_sidecar_patching = False
            if force_direct_patching and force_sidecar_patching:
                raise ValueError("Cannot force both direct and sidecar patching.")
            elif force_direct_patching:
                use_sidecar_patching = False
            elif force_sidecar_patching:
                use_sidecar_patching = True
            elif module.get_num_patches() > 0:
                use_sidecar_patching = True
            elif LayerPatcher._is_any_part_of_layer_on_cpu(module):
                use_sidecar_patching = True

            if use_sidecar_patching:
                LayerPatcher._apply_model_layer_wrapper_patch(
                    module_to_patch=module,
                    module_to_patch_key=module_key,
                    patch=layer,
                    patch_weight=patch_weight,
                    original_modules=original_modules,
                    dtype=dtype,
                )
            else:
                LayerPatcher._apply_model_layer_patch(
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
        for param_name, param_weight in patch.get_parameters(
            dict(module_to_patch.named_parameters(recurse=False)), weight=patch_weight
        ).items():
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
    def _apply_model_layer_wrapper_patch(
        module_to_patch: torch.nn.Module,
        module_to_patch_key: str,
        patch: BaseLayerPatch,
        patch_weight: float,
        original_modules: dict[str, torch.nn.Module],
        dtype: torch.dtype,
    ):
        """Apply a single LoRA wrapper patch to a module."""
        # Move the LoRA layer to the same device/dtype as the orig module.
        first_param = next(module_to_patch.parameters())
        device = first_param.device
        patch.to(device=device, dtype=dtype)

        if module_to_patch_key not in original_modules:
            original_modules[module_to_patch_key] = module_to_patch

        module_to_patch.add_patch(patch, patch_weight)

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
