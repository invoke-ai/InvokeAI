import torch

from invokeai.backend.model_manager.load.model_cache.torch_function_autocast_context import (
    add_autocast_to_module_forward,
    remove_autocast_from_module_forward,
)
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


def set_nested_attr(obj: object, attr: str, value: object):
    """A helper function that extends setattr() to support nested attributes.

    Example:
        set_nested_attr(model, "module.encoder.conv1.weight", new_conv1_weight)
    """
    attrs = attr.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


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

        # TODO(ryand): Handle the case where the model sizes changes after initial load (e.g. due to dtype casting).
        # Consider how we should handle this for both self._total_bytes and self._cur_vram_bytes.
        self._total_bytes = sum(calc_tensor_size(p) for p in self._cpu_state_dict.values())
        self._cur_vram_bytes: int | None = None

        self._update_model_autocast_context()

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
                calc_tensor_size(p) for p in cur_state_dict.values() if p.device.type == self._compute_device.type
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

        for key, param in cur_state_dict.items():
            if param.device.type == self._compute_device.type:
                continue

            param_size = calc_tensor_size(param)
            if vram_bytes_loaded + param_size > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
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

        if self._cur_vram_bytes == self.total_bytes():
            # HACK(ryand): The model should already be on the compute device, but we have to call this to ensure that
            # all non-persistent buffers are moved (i.e. buffers that are not registered in the state dict).
            self._model.to(self._compute_device)

        self._update_model_autocast_context()
        return vram_bytes_loaded

    @torch.no_grad()
    def partial_unload_from_vram(self, vram_bytes_to_free: int) -> int:
        """Unload weights from VRAM until vram_bytes_to_free bytes are freed. Or the entire model is unloaded.

        Returns:
            The number of bytes unloaded from VRAM.
        """
        vram_bytes_freed = 0

        offload_device = "cpu"
        cur_state_dict = self._model.state_dict()
        for key, param in cur_state_dict.items():
            if vram_bytes_freed >= vram_bytes_to_free:
                break

            if param.device.type == offload_device:
                continue

            cur_state_dict[key] = self._cpu_state_dict[key]
            vram_bytes_freed += calc_tensor_size(param)

        if vram_bytes_freed > 0:
            self._model.load_state_dict(cur_state_dict, assign=True)

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes -= vram_bytes_freed

        self._update_model_autocast_context()
        return vram_bytes_freed

    def _update_model_autocast_context(self):
        """A helper function that should be called whenever the model's VRAM usage changes to add/remove the autocast
        context.
        """
        if self.cur_vram_bytes() == self.total_bytes():
            # We remove the autocast context when the model is fully loaded into VRAM, because the context causes some
            # runtime overhead.
            remove_autocast_from_module_forward(self._model)
        else:
            # Monkey-patch the model to add autocasting to the model's forward method.
            add_autocast_to_module_forward(self._model, self._compute_device)
