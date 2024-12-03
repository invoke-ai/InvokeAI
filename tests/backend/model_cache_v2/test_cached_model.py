import torch

from invokeai.backend.model_cache_v2.cached_model import CachedModel


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def test_cached_model_partial_load():
    model = DummyModule()
    cached_model = CachedModel(model=model, compute_device=torch.device("cuda"))
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()


def test_cached_model_partial_unload():
    model = DummyModule()
    model.to("cuda")
    cached_model = CachedModel(model=model, compute_device=torch.device("cuda"))
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == model_total_bytes

    bytes_to_free = int(model_total_bytes * 0.4)
    freed_bytes = cached_model.partial_unload_from_vram(bytes_to_free)
    assert freed_bytes >= bytes_to_free
    assert freed_bytes < model_total_bytes
    assert freed_bytes == model_total_bytes - cached_model.cur_vram_bytes()
