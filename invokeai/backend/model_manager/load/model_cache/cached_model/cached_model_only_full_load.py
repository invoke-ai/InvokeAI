from typing import Any

import torch


class CachedModelOnlyFullLoad:
    """A wrapper around a PyTorch model to handle full loads and unloads between the CPU and the compute device.
    Note: "VRAM" is used throughout this class to refer to the memory on the compute device. It could be CUDA memory,
    MPS memory, etc.
    """

    def __init__(
        self, model: torch.nn.Module | Any, compute_device: torch.device, total_bytes: int, keep_ram_copy: bool = False
    ):
        """Initialize a CachedModelOnlyFullLoad.
        Args:
            model (torch.nn.Module | Any): The model to wrap. Should be on the CPU.
            compute_device (torch.device): The compute device to move the model to.
            total_bytes (int): The total size (in bytes) of all the weights in the model.
            keep_ram_copy (bool): Whether to keep a read-only copy of the model's state dict in RAM. Keeping a RAM copy
                increases RAM usage, but speeds up model offload from VRAM and LoRA patching (assuming there is
                sufficient RAM).
        """
        # model is often a torch.nn.Module, but could be any model type. Throughout this class, we handle both cases.
        self._model = model
        self._compute_device = compute_device
        self._offload_device = torch.device("cpu")

        # A CPU read-only copy of the model's state dict.
        self._cpu_state_dict: dict[str, torch.Tensor] | None = None
        if isinstance(model, torch.nn.Module) and keep_ram_copy:
            self._cpu_state_dict = model.state_dict()

        self._total_bytes = total_bytes
        self._is_in_vram = False

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
        if self._is_in_vram:
            return self._total_bytes
        else:
            return 0

    def is_in_vram(self) -> bool:
        """Return true if the model is currently in VRAM."""
        return self._is_in_vram

    def full_load_to_vram(self) -> int:
        """Load all weights into VRAM (if supported by the model).
        Returns:
            The number of bytes loaded into VRAM.
        """
        if self._is_in_vram:
            # Already in VRAM.
            return 0

        if not hasattr(self._model, "to"):
            # Model doesn't support moving to a device.
            return 0

        if self._cpu_state_dict is not None:
            new_state_dict: dict[str, torch.Tensor] = {}
            for k, v in self._cpu_state_dict.items():
                new_state_dict[k] = v.to(self._compute_device, copy=True)
            self._model.load_state_dict(new_state_dict, assign=True)
        self._model.to(self._compute_device)

        self._is_in_vram = True
        return self._total_bytes

    def full_unload_from_vram(self) -> int:
        """Unload all weights from VRAM.
        Returns:
            The number of bytes unloaded from VRAM.
        """
        if not self._is_in_vram:
            # Already in RAM.
            return 0

        if self._cpu_state_dict is not None:
            self._model.load_state_dict(self._cpu_state_dict, assign=True)
        self._model.to(self._offload_device)

        self._is_in_vram = False
        return self._total_bytes
