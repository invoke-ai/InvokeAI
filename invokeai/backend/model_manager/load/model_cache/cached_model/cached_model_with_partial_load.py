import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from invokeai.backend.util.logging import InvokeAILogger


class CachedModelWithPartialLoad:
    """A wrapper around a PyTorch model to handle partial loads and unloads between the CPU and the compute device.

    Note: "VRAM" is used throughout this class to refer to the memory on the compute device. It could be CUDA memory,
    MPS memory, etc.
    """

    def __init__(self, model: torch.nn.Module, compute_device: torch.device, keep_ram_copy: bool = False):
        self._model = model
        self._compute_device = compute_device

        model_state_dict = model.state_dict()
        # A CPU read-only copy of the model's state dict. Used for faster model unloads from VRAM, and to speed up LoRA
        # patching. Set to `None` if keep_ram_copy is False.
        self._cpu_state_dict: dict[str, torch.Tensor] | None = model_state_dict if keep_ram_copy else None

        # A dictionary of the size of each tensor in the state dict.
        # HACK(ryand): We use this dictionary any time we are doing byte tracking calculations. We do this for
        # consistency in case the application code has modified the model's size (e.g. by casting to a different
        # precision). Of course, this means that we are making model cache load/unload decisions based on model size
        # data that may not be fully accurate.
        self._state_dict_bytes = {k: calc_tensor_size(v) for k, v in model_state_dict.items()}

        self._total_bytes = sum(self._state_dict_bytes.values())
        self._cur_vram_bytes: int | None = None

        self._modules_that_support_autocast = self._find_modules_that_support_autocast()
        self._keys_in_modules_that_do_not_support_autocast = self._find_keys_in_modules_that_do_not_support_autocast(
            model_state_dict
        )
        self._state_dict_keys_by_module_prefix = self._group_state_dict_keys_by_module_prefix(model_state_dict)

    def _find_modules_that_support_autocast(self) -> dict[str, torch.nn.Module]:
        """Find all modules that support autocasting."""
        return {n: m for n, m in self._model.named_modules() if isinstance(m, CustomModuleMixin)}  # type: ignore

    def _find_keys_in_modules_that_do_not_support_autocast(self, state_dict: dict[str, torch.Tensor]) -> set[str]:
        keys_in_modules_that_do_not_support_autocast: set[str] = set()
        for key in state_dict.keys():
            for module_name in self._modules_that_support_autocast.keys():
                if key.startswith(module_name):
                    break
            else:
                keys_in_modules_that_do_not_support_autocast.add(key)
        return keys_in_modules_that_do_not_support_autocast

    def _group_state_dict_keys_by_module_prefix(self, state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        """A helper function that groups state dict keys by module prefix.

        Example:
        ```
        state_dict = {
            "weight": ...,
            "module.submodule.weight": ...,
            "module.submodule.bias": ...,
            "module.other_submodule.weight": ...,
            "module.other_submodule.bias": ...,
        }

        output = group_state_dict_keys_by_module_prefix(state_dict)

        # The output will be:
        output = {
            "": [
                "weight",
            ],
            "module.submodule": [
                "module.submodule.weight",
                "module.submodule.bias",
            ],
            "module.other_submodule": [
                "module.other_submodule.weight",
                "module.other_submodule.bias",
            ],
        }
        ```
        """
        state_dict_keys_by_module_prefix: dict[str, list[str]] = {}
        for key in state_dict.keys():
            split = key.rsplit(".", 1)
            # `split` will have length 1 if the root module has parameters.
            module_name = split[0] if len(split) > 1 else ""
            if module_name not in state_dict_keys_by_module_prefix:
                state_dict_keys_by_module_prefix[module_name] = []
            state_dict_keys_by_module_prefix[module_name].append(key)
        return state_dict_keys_by_module_prefix

    def _move_non_persistent_buffers_to_device(self, device: torch.device):
        """Move the non-persistent buffers to the target device. These buffers are not included in the state dict,
        so we need to move them manually.
        """
        # HACK(ryand): Typically, non-persistent buffers are moved when calling module.to(device). We don't move entire
        # modules, because we manage the devices of individual tensors using the state dict. Since non-persistent
        # buffers are not included in the state dict, we need to handle them manually. The only way to do this is by
        # using private torch.nn.Module attributes.
        for module in self._model.modules():
            for name, buffer in module.named_buffers():
                if name in module._non_persistent_buffers_set:
                    module._buffers[name] = buffer.to(device, copy=True)

    def _set_autocast_enabled_in_all_modules(self, enabled: bool):
        """Set autocast_enabled flag in all modules that support device autocasting."""
        for module in self._modules_that_support_autocast.values():
            module.set_device_autocasting_enabled(enabled)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def get_cpu_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Get a read-only copy of the model's state dict in RAM."""
        # TODO(ryand): Document this better.
        return self._cpu_state_dict

    def total_bytes(self) -> int:
        """Get the total size (in bytes) of all the weights in the model."""
        return self._total_bytes

    def cur_vram_bytes(self) -> int:
        """Get the size (in bytes) of the weights that are currently in VRAM."""
        if self._cur_vram_bytes is None:
            cur_state_dict = self._model.state_dict()
            self._cur_vram_bytes = sum(
                self._state_dict_bytes[k]
                for k, v in cur_state_dict.items()
                if v.device.type == self._compute_device.type
            )
        return self._cur_vram_bytes

    def full_load_to_vram(self) -> int:
        """Load all weights into VRAM."""
        return self.partial_load_to_vram(self.total_bytes())

    def full_unload_from_vram(self) -> int:
        """Unload all weights from VRAM."""
        return self.partial_unload_from_vram(self.total_bytes())

    def _load_state_dict_with_device_conversion(
        self, state_dict: dict[str, torch.Tensor], keys_to_convert: set[str], target_device: torch.device
    ):
        if self._cpu_state_dict is not None:
            # Run the fast version.
            self._load_state_dict_with_fast_device_conversion(
                state_dict=state_dict,
                keys_to_convert=keys_to_convert,
                target_device=target_device,
                cpu_state_dict=self._cpu_state_dict,
            )
        else:
            # Run the low-virtual-memory version.
            self._load_state_dict_with_jit_device_conversion(
                state_dict=state_dict,
                keys_to_convert=keys_to_convert,
                target_device=target_device,
            )

    def _load_state_dict_with_jit_device_conversion(
        self,
        state_dict: dict[str, torch.Tensor],
        keys_to_convert: set[str],
        target_device: torch.device,
    ):
        """A custom state dict loading implementation with good peak memory properties.

        This implementation has the important property that it copies parameters to the target device one module at a time
        rather than applying all of the device conversions and then calling load_state_dict(). This is done to minimize the
        peak virtual memory usage. Specifically, we want to avoid a case where we hold references to all of the CPU weights
        and CUDA weights simultaneously, because Windows will reserve virtual memory for both.
        """
        for module_name, module in self._model.named_modules():
            module_keys = self._state_dict_keys_by_module_prefix.get(module_name, [])
            # Calculate the length of the module name prefix.
            prefix_len = len(module_name)
            if prefix_len > 0:
                prefix_len += 1

            module_state_dict = {}
            for key in module_keys:
                if key in keys_to_convert:
                    # It is important that we overwrite `state_dict[key]` to avoid keeping two copies of the same
                    # parameter.
                    state_dict[key] = state_dict[key].to(target_device)
                # Note that we keep parameters that have not been moved to a new device in case the module implements
                # weird custom state dict loading logic that requires all parameters to be present.
                module_state_dict[key[prefix_len:]] = state_dict[key]

            if len(module_state_dict) > 0:
                # We set strict=False, because if `module` has both parameters and child modules, then we are loading a
                # state dict that only contains the parameters of `module` (not its children).
                # We assume that it is rare for non-leaf modules to have parameters. Calling load_state_dict() on non-leaf
                # modules will recurse through all of the children, so is a bit wasteful.
                incompatible_keys = module.load_state_dict(module_state_dict, strict=False, assign=True)
                # Missing keys are ok, unexpected keys are not.
                assert len(incompatible_keys.unexpected_keys) == 0

    def _load_state_dict_with_fast_device_conversion(
        self,
        state_dict: dict[str, torch.Tensor],
        keys_to_convert: set[str],
        target_device: torch.device,
        cpu_state_dict: dict[str, torch.Tensor],
    ):
        """Convert parameters to the target device and load them into the model. Leverages the `cpu_state_dict` to speed
        up transfers of weights to the CPU.
        """
        for key in keys_to_convert:
            if target_device.type == "cpu":
                state_dict[key] = cpu_state_dict[key]
            else:
                state_dict[key] = state_dict[key].to(target_device)

        self._model.load_state_dict(state_dict, assign=True)

    @torch.no_grad()
    def partial_load_to_vram(self, vram_bytes_to_load: int) -> int:
        """Load more weights into VRAM without exceeding vram_bytes_to_load.

        Returns:
            The number of bytes loaded into VRAM.
        """
        # TODO(ryand): Handle the case where an exception is thrown while loading or unloading weights. At the very
        # least, we should reset self._cur_vram_bytes to None.

        vram_bytes_loaded = 0

        cur_state_dict = self._model.state_dict()

        # Identify the keys that will be loaded into VRAM.
        keys_to_load: set[str] = set()

        # First, process the keys that *must* be loaded into VRAM.
        for key in self._keys_in_modules_that_do_not_support_autocast:
            param = cur_state_dict[key]
            if param.device.type == self._compute_device.type:
                continue

            keys_to_load.add(key)
            param_size = self._state_dict_bytes[key]
            vram_bytes_loaded += param_size

        if vram_bytes_loaded > vram_bytes_to_load:
            logger = InvokeAILogger.get_logger()
            logger.warning(
                f"Loading {vram_bytes_loaded / 2**20} MB into VRAM, but only {vram_bytes_to_load / 2**20} MB were "
                "requested. This is the minimum set of weights in VRAM required to run the model."
            )

        # Next, process the keys that can optionally be loaded into VRAM.
        fully_loaded = True
        for key, param in cur_state_dict.items():
            # Skip the keys that have already been processed above.
            if key in keys_to_load:
                continue

            if param.device.type == self._compute_device.type:
                continue

            param_size = self._state_dict_bytes[key]
            if vram_bytes_loaded + param_size > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
                fully_loaded = False
                continue

            keys_to_load.add(key)
            vram_bytes_loaded += param_size

        if len(keys_to_load) > 0:
            # We load the entire state dict, not just the parameters that changed, in case there are modules that
            # override _load_from_state_dict() and do some funky stuff that requires the entire state dict.
            # Alternatively, in the future, grouping parameters by module could probably solve this problem.
            self._load_state_dict_with_device_conversion(cur_state_dict, keys_to_load, self._compute_device)

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes += vram_bytes_loaded

        if fully_loaded:
            self._set_autocast_enabled_in_all_modules(False)
        else:
            self._set_autocast_enabled_in_all_modules(True)

        # Move all non-persistent buffers to the compute device. These are a weird edge case and do not participate in
        # the vram_bytes_loaded tracking.
        self._move_non_persistent_buffers_to_device(self._compute_device)

        return vram_bytes_loaded

    @torch.no_grad()
    def partial_unload_from_vram(self, vram_bytes_to_free: int, keep_required_weights_in_vram: bool = False) -> int:
        """Unload weights from VRAM until vram_bytes_to_free bytes are freed. Or the entire model is unloaded.

        :param keep_required_weights_in_vram: If True, any weights that must be kept in VRAM to run the model will be
            kept in VRAM.

        Returns:
            The number of bytes unloaded from VRAM.
        """
        vram_bytes_freed = 0
        required_weights_in_vram = 0

        offload_device = "cpu"
        cur_state_dict = self._model.state_dict()

        # Identify the keys that will be offloaded to CPU.
        keys_to_offload: set[str] = set()

        for key, param in cur_state_dict.items():
            if vram_bytes_freed >= vram_bytes_to_free:
                break

            if param.device.type == offload_device:
                continue

            if keep_required_weights_in_vram and key in self._keys_in_modules_that_do_not_support_autocast:
                required_weights_in_vram += self._state_dict_bytes[key]
                continue

            keys_to_offload.add(key)
            vram_bytes_freed += self._state_dict_bytes[key]

        if len(keys_to_offload) > 0:
            self._load_state_dict_with_device_conversion(cur_state_dict, keys_to_offload, torch.device("cpu"))

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes -= vram_bytes_freed

        # We may have gone from a fully-loaded model to a partially-loaded model, so we need to reapply the custom
        # layers.
        self._set_autocast_enabled_in_all_modules(True)
        return vram_bytes_freed
