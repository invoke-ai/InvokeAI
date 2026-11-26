"""The cache makes room for a single-file checkpoint before its payload is read into process memory.

A float8 checkpoint is read with ``pread``, so its tensors are ordinary process memory the OS cannot reclaim, unlike
the page cache an mmap load leaves them in. The room for that read is made by the loader base class, which asks the
cache for the file's size before ``_load_model`` runs; the single-file loaders reserve again later for the state dict
they build, which replaces that request rather than adding to it (``make_room`` holds nothing).
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.quantization.fp8_scaled import load_safetensors_checkpoint
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.safetensors_files import record_reads


def _is_cached(cache: ModelCache, key: str) -> bool:
    try:
        cache.get(key)
    except IndexError:
        return False
    return True


class _ReadsAnFp8File(ModelLoader):
    """What every single-file loader does first: read the checkpoint through the shared helper."""

    def _load_model(self, config: Any, submodel_type: Any = None) -> torch.nn.Module:
        load_safetensors_checkpoint(Path(config.path))
        return torch.nn.Linear(2, 2)


def test_room_for_the_file_is_made_before_the_first_tensor_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "a.weight": torch.ones(64, 64).to(torch.float8_e4m3fn),
            "a.bias": torch.zeros(64, dtype=torch.bfloat16),
            "norm.weight": torch.ones(64, dtype=torch.bfloat16),
        },
        str(path),
    )
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        # Room for about 10 kB: the cached model below leaves less than the file needs, so room has to be made.
        max_ram_cache_size_gb=1e-5,
        max_vram_cache_size_gb=0.0,
        execution_device="cpu",
        logger=InvokeAILogger.get_logger("test"),
        shared_cpu_weights=None,
    )
    cache.put("evictable", torch.nn.Linear(40, 40, bias=False))  # 6.4 kB
    log = record_reads(monkeypatch)
    real_make_room = cache.make_room

    def make_room(bytes_needed: int):
        result = real_make_room(bytes_needed)
        # One ordered list with the reads: what was asked for, and whether the cached model was evicted by then.
        log.events.append(("make_room", f"{bytes_needed} evicted={not _is_cached(cache, 'evictable')}"))
        return result

    monkeypatch.setattr(cache, "make_room", make_room)
    app_config = InvokeAIAppConfig(use_memory_db=True, models_dir=tmp_path)
    loader = _ReadsAnFp8File(app_config, InvokeAILogger.get_logger("test"), cache)
    config = Main_Checkpoint_Krea2_Config.model_construct(
        key="k", name="model", path=str(path), cpu_only=None, repo_variant=None
    )

    loader.load_model(config)

    assert log.backends == ["pread"], "the file holds a float8 weight, so it is read into owned memory"
    kinds = [event[0] for event in log.events]
    first_read = kinds.index("read")
    asked = [event for event in log.events[:first_read] if event[0] == "make_room"]
    assert asked, f"nothing made room before the first read: {log.events}"
    # The room is for at least what the read puts into memory (the header is not payload) and no more than the file,
    # and the cache has evicted what stood in the way by the time the first tensor is read.
    size, evicted = asked[0][1].split(" evicted=")
    assert path.stat().st_size - 1024 <= int(size) <= path.stat().st_size
    assert evicted == "True"
