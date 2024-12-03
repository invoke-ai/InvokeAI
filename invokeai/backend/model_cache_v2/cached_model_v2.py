import torch

from invokeai.backend.model_cache_v2.torch_module_overrides import CustomLinear, inject_custom_layers_into_module


class CachedModelV2:
    """A wrapper around a PyTorch model to handle partial loads and unloads between the CPU and the compute device.

    Note: "VRAM" is used throughout this class to refer to the memory on the compute device. It could be CUDA memory,
    MPS memory, etc.
    """

    def __init__(self, model: torch.nn.Module, compute_device: torch.device):
        print("CachedModelV2.__init__")
        self._model = model
        inject_custom_layers_into_module(self._model)
        self._compute_device = compute_device

        # Memoized values.
        self._total_size_cache = None
        self._cur_vram_bytes_cache = None

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def total_bytes(self) -> int:
        if self._total_size_cache is None:
            self._total_size_cache = sum(p.numel() * p.element_size() for p in self._model.parameters())
        return self._total_size_cache

    def cur_vram_bytes(self) -> int:
        """Return the size (in bytes) of the weights that are currently in VRAM."""
        if self._cur_vram_bytes_cache is None:
            self._cur_vram_bytes_cache = sum(
                p.numel() * p.element_size()
                for p in self._model.parameters()
                if p.device.type == self._compute_device.type
            )
        return self._cur_vram_bytes_cache

    def full_load_to_vram(self):
        """Load all weights into VRAM."""
        raise NotImplementedError("Not implemented")
        self._cur_vram_bytes_cache = self.total_bytes()

    def partial_load_to_vram(self, vram_bytes_to_load: int) -> int:
        """Load more weights into VRAM without exceeding vram_bytes_to_load.

        Returns:
            The number of bytes loaded into VRAM.
        """
        vram_bytes_loaded = 0

        def to_vram(m: torch.nn.Module):
            nonlocal vram_bytes_loaded

            if not isinstance(m, CustomLinear):
                # We don't handle offload of this type of module.
                return

            m_device = m.weight.device
            m_bytes = sum(p.numel() * p.element_size() for p in m.parameters())

            # Skip modules that are already on the compute device.
            if m_device.type == self._compute_device.type:
                return

            # Check the size of the parameter.
            if vram_bytes_loaded + m_bytes > vram_bytes_to_load:
                # TODO(ryand): Should we just break here? If we couldn't fit this parameter into VRAM, is it really
                # worth continuing to search for a smaller parameter that would fit?
                return

            vram_bytes_loaded += m_bytes
            m.to(self._compute_device)

        self._model.apply(to_vram)
        self._cur_vram_bytes_cache = None

        return vram_bytes_loaded

    def partial_unload_from_vram(self, vram_bytes_to_free: int) -> int:
        """Unload weights from VRAM until vram_bytes_to_free bytes are freed. Or the entire model is unloaded."""

        vram_bytes_freed = 0

        def from_vram(m: torch.nn.Module):
            nonlocal vram_bytes_freed

            if vram_bytes_freed >= vram_bytes_to_free:
                return

            m_device = m.weight.device
            m_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
            if m_device.type != self._compute_device.type:
                return

            vram_bytes_freed += m_bytes
            m.to("cpu")

        self._model.apply(from_vram)
        self._cur_vram_bytes_cache = None

        return vram_bytes_freed
