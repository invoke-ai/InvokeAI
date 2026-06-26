from typing import Any

import torch

from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class CachedModelOnlyFullLoad:
    """A wrapper around a PyTorch model to handle full loads and unloads between the CPU and the compute device.
    Note: "VRAM" is used throughout this class to refer to the memory on the compute device. It could be CUDA memory,
    MPS memory, etc.
    """

    def __init__(
        self,
        model: torch.nn.Module | Any,
        compute_device: torch.device,
        total_bytes: int,
        keep_ram_copy: bool = False,
        shared_store: SharedCpuWeightsStore | None = None,
        cache_key: str | None = None,
    ):
        """Initialize a CachedModelOnlyFullLoad.
        Args:
            model (torch.nn.Module | Any): The model to wrap. Should be on the CPU.
            compute_device (torch.device): The compute device to move the model to.
            total_bytes (int): The total size (in bytes) of all the weights in the model.
            keep_ram_copy (bool): Whether to keep a read-only copy of the model's state dict in RAM. Keeping a RAM copy
                increases RAM usage, but speeds up model offload from VRAM and LoRA patching (assuming there is
                sufficient RAM).
            shared_store (SharedCpuWeightsStore | None): If provided (along with cache_key), share a single canonical
                CPU copy of the weights across per-device caches instead of one copy per device.
            cache_key (str | None): The model cache key used to identify shared weights in `shared_store`.
        """
        # model is often a torch.nn.Module, but could be any model type. Throughout this class, we handle both cases.
        self._model = model
        self._compute_device = compute_device
        self._offload_device = torch.device("cpu")
        # When set, this model's CPU weights are a shared canonical copy owned by `shared_store`
        # under `cache_key`; `release_shared_weights()` must be called exactly once on eviction.
        self._shared_store: SharedCpuWeightsStore | None = None
        self._shared_key: str | None = None

        # A CPU read-only copy of the model's state dict.
        self._cpu_state_dict: dict[str, torch.Tensor] | None = None
        if isinstance(model, torch.nn.Module) and keep_ram_copy:
            cpu_state_dict = model.state_dict()
            # In multi-GPU mode, share one canonical CPU copy across the per-device caches (see
            # SharedCpuWeightsStore). If another device already registered this key, re-point our
            # module at the shared tensors and drop our duplicate so the weights live once in RAM.
            if shared_store is not None and cache_key is not None:
                canonical = shared_store.acquire(cache_key, cpu_state_dict)
                if canonical is not cpu_state_dict:
                    model.load_state_dict(canonical, assign=True)
                cpu_state_dict = canonical
                self._shared_store = shared_store
                self._shared_key = cache_key
            self._cpu_state_dict = cpu_state_dict

        self._total_bytes = total_bytes
        self._is_in_vram = False

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def get_cpu_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Get a read-only copy of the model's state dict in RAM."""
        # TODO(ryand): Document this better.
        return self._cpu_state_dict

    @property
    def uses_shared_weights(self) -> bool:
        """True if this model's CPU weights are deduplicated in a SharedCpuWeightsStore.

        When True, its RAM is accounted by the store (counted once across devices); when False, its
        RAM is per-instance and must be counted by the RamBudget's non-shared total.
        """
        return self._shared_store is not None

    def release_shared_weights(self) -> None:
        """Release this model's reference to its shared canonical CPU weights, if any.

        Must be called exactly once when the cache entry is evicted. Idempotent: a second call is a
        no-op. After release, the shared store frees the canonical tensors once the last device that
        held this key releases it.
        """
        if self._shared_store is not None and self._shared_key is not None:
            self._shared_store.release(self._shared_key)
            self._shared_store = None
            self._shared_key = None

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

    @property
    def compute_device(self) -> torch.device:
        """Return the compute device for this model."""
        return self._compute_device

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

        check_for_gguf = hasattr(self._model, "state_dict") and self._model.state_dict().get("img_in.weight")
        if isinstance(check_for_gguf, GGMLTensor):
            old_value = torch.__future__.get_overwrite_module_params_on_conversion()
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            self._model.to(self._compute_device)
            torch.__future__.set_overwrite_module_params_on_conversion(old_value)
        else:
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

        check_for_gguf = hasattr(self._model, "state_dict") and self._model.state_dict().get("img_in.weight")
        if isinstance(check_for_gguf, GGMLTensor):
            old_value = torch.__future__.get_overwrite_module_params_on_conversion()
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            self._model.to(self._offload_device)
            torch.__future__.set_overwrite_module_params_on_conversion(old_value)
        else:
            self._model.to(self._offload_device)

        self._is_in_vram = False
        return self._total_bytes
