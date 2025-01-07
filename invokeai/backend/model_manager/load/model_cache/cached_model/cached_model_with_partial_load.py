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

    def __init__(self, model: torch.nn.Module, compute_device: torch.device):
        self._model = model
        self._compute_device = compute_device

        # A CPU read-only copy of the model's state dict.
        self._cpu_state_dict: dict[str, torch.Tensor] = model.state_dict()

        # A dictionary of the size of each tensor in the state dict.
        # HACK(ryand): We use this dictionary any time we are doing byte tracking calculations. We do this for
        # consistency in case the application code has modified the model's size (e.g. by casting to a different
        # precision). Of course, this means that we are making model cache load/unload decisions based on model size
        # data that may not be fully accurate.
        self._state_dict_bytes = {k: calc_tensor_size(v) for k, v in self._cpu_state_dict.items()}

        self._total_bytes = sum(self._state_dict_bytes.values())
        self._cur_vram_bytes: int | None = None

        self._modules_that_support_autocast = self._find_modules_that_support_autocast()
        self._keys_in_modules_that_do_not_support_autocast = self._find_keys_in_modules_that_do_not_support_autocast()

    def _find_modules_that_support_autocast(self) -> dict[str, torch.nn.Module]:
        """Find all modules that support autocasting."""
        return {n: m for n, m in self._model.named_modules() if isinstance(m, CustomModuleMixin)}  # type: ignore

    def _find_keys_in_modules_that_do_not_support_autocast(self) -> set[str]:
        keys_in_modules_that_do_not_support_autocast: set[str] = set()
        for key in self._cpu_state_dict.keys():
            for module_name in self._modules_that_support_autocast.keys():
                if key.startswith(module_name):
                    break
            else:
                keys_in_modules_that_do_not_support_autocast.add(key)
        return keys_in_modules_that_do_not_support_autocast

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

        # First, process the keys that *must* be loaded into VRAM.
        for key in self._keys_in_modules_that_do_not_support_autocast:
            param = cur_state_dict[key]
            if param.device.type == self._compute_device.type:
                continue

            param_size = self._state_dict_bytes[key]
            cur_state_dict[key] = param.to(self._compute_device, copy=True)
            vram_bytes_loaded += param_size

        if vram_bytes_loaded > vram_bytes_to_load:
            logger = InvokeAILogger.get_logger()
            logger.warning(
                f"Loaded {vram_bytes_loaded / 2**20} MB into VRAM, but only {vram_bytes_to_load / 2**20} MB were "
                "requested. This is the minimum set of weights in VRAM required to run the model."
            )

        # Next, process the keys that can optionally be loaded into VRAM.
        fully_loaded = True
        for key, param in cur_state_dict.items():
            if param.device.type == self._compute_device.type:
                continue

            param_size = self._state_dict_bytes[key]
            if vram_bytes_loaded + param_size > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
                fully_loaded = False
                continue

            cur_state_dict[key] = param.to(self._compute_device, copy=True)
            vram_bytes_loaded += param_size

        if vram_bytes_loaded > 0:
            # We load the entire state dict, not just the parameters that changed, in case there are modules that
            # override _load_from_state_dict() and do some funky stuff that requires the entire state dict.
            # Alternatively, in the future, grouping parameters by module could probably solve this problem.
            self._model.load_state_dict(cur_state_dict, assign=True)

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
        for key, param in cur_state_dict.items():
            if vram_bytes_freed >= vram_bytes_to_free:
                break

            if param.device.type == offload_device:
                continue

            if keep_required_weights_in_vram and key in self._keys_in_modules_that_do_not_support_autocast:
                required_weights_in_vram += self._state_dict_bytes[key]
                continue

            cur_state_dict[key] = self._cpu_state_dict[key]
            vram_bytes_freed += self._state_dict_bytes[key]

        if vram_bytes_freed > 0:
            self._model.load_state_dict(cur_state_dict, assign=True)

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes -= vram_bytes_freed

        # We may have gone from a fully-loaded model to a partially-loaded model, so we need to reapply the custom
        # layers.
        self._set_autocast_enabled_in_all_modules(True)
        return vram_bytes_freed
