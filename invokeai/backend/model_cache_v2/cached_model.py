import torch

from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class CachedModel:
    """A wrapper around a PyTorch model to handle partial loads and unloads between the CPU and the compute device.

    Note: "VRAM" is used throughout this class to refer to the memory on the compute device. It could be CUDA memory,
    MPS memory, etc.
    """

    def __init__(self, model: torch.nn.Module, compute_device: torch.device):
        self._model = model
        self._compute_device = compute_device

        # TODO(ryand): Add memoization for total_bytes and cur_vram_bytes?

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def total_bytes(self) -> int:
        """Get the total size (in bytes) of all the weights in the model."""
        return sum(calc_tensor_size(p) for p in self._model.parameters())

    def cur_vram_bytes(self) -> int:
        """Get the size (in bytes) of the weights that are currently in VRAM."""
        return sum(calc_tensor_size(p) for p in self._model.parameters() if p.device.type == self._compute_device.type)

    def partial_load_to_vram(self, vram_bytes_to_load: int) -> int:
        """Load more weights into VRAM without exceeding vram_bytes_to_load.

        Returns:
            The number of bytes loaded into VRAM.
        """
        vram_bytes_loaded = 0

        # TODO(ryand): Should we use self._model.apply(...) instead and move modules around instead of moving tensors?
        # This way we don't have to use the private _apply() method.
        def to_vram(t: torch.Tensor):
            nonlocal vram_bytes_loaded

            # Skip parameters that are already on the compute device.
            if t.device.type == self._compute_device.type:
                return t

            # Check the size of the parameter.
            param_size = calc_tensor_size(t)
            if vram_bytes_loaded + param_size > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
                return t

            vram_bytes_loaded += param_size
            return t.to(self._compute_device)

        self._model._apply(to_vram)

        return vram_bytes_loaded

    def partial_unload_from_vram(self, vram_bytes_to_free: int) -> int:
        """Unload weights from VRAM until vram_bytes_to_free bytes are freed. Or the entire model is unloaded.

        Returns:
            The number of bytes unloaded from VRAM.
        """
        vram_bytes_freed = 0

        def from_vram(t: torch.Tensor):
            nonlocal vram_bytes_freed

            if vram_bytes_freed >= vram_bytes_to_free:
                return t

            if t.device.type != self._compute_device.type:
                return t

            vram_bytes_freed += calc_tensor_size(t)
            return t.to("cpu")

        self._model._apply(from_vram)

        return vram_bytes_freed
