import itertools

import torch

from invokeai.backend.model_manager.load.model_cache.torch_function_autocast_context import (
    add_autocast_to_module_forward,
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

        # Monkey-patch the model to add autocasting to the model's forward method.
        add_autocast_to_module_forward(model, compute_device)

        self._total_bytes = sum(
            calc_tensor_size(p) for p in itertools.chain(self._model.parameters(), self._model.buffers())
        )
        self._cur_vram_bytes: int | None = None

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
            self._cur_vram_bytes = sum(
                calc_tensor_size(p)
                for p in itertools.chain(self._model.parameters(), self._model.buffers())
                if p.device.type == self._compute_device.type
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
        vram_bytes_loaded = 0

        for key, param in itertools.chain(self._model.named_parameters(), self._model.named_buffers()):
            # Skip parameters that are already on the compute device.
            if param.device.type == self._compute_device.type:
                continue

            # Check the size of the parameter.
            param_size = calc_tensor_size(param)
            if vram_bytes_loaded + param_size > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
                continue

            # Copy the parameter to the compute device.
            # We use the 'overwrite' strategy from torch.nn.Module._apply().
            # TODO(ryand): For some edge cases (e.g. quantized models?), we may need to support other strategies (e.g.
            # swap).
            if isinstance(param, torch.nn.Parameter):
                assert param.is_leaf
                out_param = torch.nn.Parameter(
                    param.to(self._compute_device, copy=True), requires_grad=param.requires_grad
                )
                set_nested_attr(self._model, key, out_param)
                # We did not port the param.grad handling from torch.nn.Module._apply(), because we do not expect to be
                # handling gradients. We assert that this assumption is true.
                assert param.grad is None
            else:
                # Handle buffers.
                set_nested_attr(self._model, key, param.to(self._compute_device, copy=True))

            vram_bytes_loaded += param_size

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes += vram_bytes_loaded

        return vram_bytes_loaded

    @torch.no_grad()
    def partial_unload_from_vram(self, vram_bytes_to_free: int) -> int:
        """Unload weights from VRAM until vram_bytes_to_free bytes are freed. Or the entire model is unloaded.

        Returns:
            The number of bytes unloaded from VRAM.
        """
        vram_bytes_freed = 0

        # TODO(ryand): Iterate over buffers too?
        for key, param in self._model.named_parameters():
            if vram_bytes_freed >= vram_bytes_to_free:
                break

            if param.device.type != self._compute_device.type:
                continue

            # Create a new parameter, but inject the existing CPU tensor into it.
            out_param = torch.nn.Parameter(self._cpu_state_dict[key], requires_grad=param.requires_grad)
            set_nested_attr(self._model, key, out_param)
            vram_bytes_freed += calc_tensor_size(param)

        if self._cur_vram_bytes is not None:
            self._cur_vram_bytes -= vram_bytes_freed

        return vram_bytes_freed
