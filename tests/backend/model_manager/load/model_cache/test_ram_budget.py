import torch

from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


def test_total_in_use_sums_store_and_non_shared():
    store = SharedCpuWeightsStore()
    store.acquire("k", {"a": torch.ones(100, dtype=torch.float32)})  # 400 bytes
    budget = RamBudget(max_bytes=10_000, shared_store=store)

    assert budget.total_in_use() == 400  # store only
    budget.add_non_shared(600)
    assert budget.total_in_use() == 1000
    assert budget.available() == 9000
    budget.remove_non_shared(600)
    assert budget.total_in_use() == 400


def test_shared_weights_counted_once_regardless_of_refcount():
    store = SharedCpuWeightsStore()
    sd = {"a": torch.ones(100, dtype=torch.float32)}  # 400 bytes
    store.acquire("k", sd)
    store.acquire("k", sd)  # second device acquires the same key
    budget = RamBudget(max_bytes=10_000, shared_store=store)
    # Two references, one physical copy -> counted once.
    assert budget.total_in_use() == 400


def test_remove_non_shared_floors_at_zero():
    budget = RamBudget(max_bytes=10_000, shared_store=None)
    budget.add_non_shared(100)
    budget.remove_non_shared(500)
    assert budget.total_in_use() == 0


def test_available_can_go_negative_when_over_budget():
    budget = RamBudget(max_bytes=100, shared_store=None)
    budget.add_non_shared(250)
    assert budget.available() == -150


def test_no_store_tracks_only_non_shared():
    budget = RamBudget(max_bytes=1000, shared_store=None)
    assert budget.total_in_use() == 0
    budget.add_non_shared(300)
    assert budget.total_in_use() == 300
    assert budget.max_bytes == 1000
