"""Clearing a worker's VRAM when its sessions switch architecture, and only then."""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import (
    LoRACollectionLoader,
    LoRAField,
    LoRALoaderInvocation,
    MainModelLoaderInvocation,
    ModelIdentifierField,
    VAELoaderInvocation,
)
from invokeai.app.invocations.primitives import IntegerInvocation
from invokeai.app.services.session_processor.architecture_switch import ArchitectureSwitchOffload, models_named_by
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.shared.graph_validation import Graph
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _model(key: str, base: BaseModelType, type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"blake3:{key}", name=key, base=base, type=type)


SD1 = _model("sd1-main", BaseModelType.StableDiffusion1, ModelType.Main)
SD1_LORA = _model("sd1-lora", BaseModelType.StableDiffusion1, ModelType.LoRA)
SD2 = _model("sd2-main", BaseModelType.StableDiffusion2, ModelType.Main)
SHARED_VAE = _model("shared-vae", BaseModelType.StableDiffusion1, ModelType.VAE)


def _graph(main: ModelIdentifierField | None, *extra: ModelIdentifierField) -> Graph:
    nodes: dict = {"seed": IntegerInvocation(id="seed", value=1)}
    if main is not None:
        nodes["main"] = MainModelLoaderInvocation(id="main", model=main)
    for model in extra:
        if model.type is ModelType.LoRA:
            nodes[model.key] = LoRALoaderInvocation(id=model.key, lora=model)
        else:
            nodes[model.key] = VAELoaderInvocation(id=model.key, vae_model=model)
    return Graph(nodes=nodes)


class _Cache:
    execution_device = "cuda:0"

    def __init__(self) -> None:
        self.kept: list[set[str]] = []

    def offload_models_from_vram_except(self, keep_model_keys) -> int:
        self.kept.append(set(keep_model_keys))
        return 3 * 2**30


def test_every_model_a_graph_names_is_found_including_loras() -> None:
    graph = _graph(SD1, SD1_LORA, SHARED_VAE)
    assert {model.key for model in models_named_by(graph)} == {"sd1-main", "sd1-lora", "shared-vae"}


def test_models_nested_in_node_inputs_are_found() -> None:
    """A LoRA collection names its models inside a list of `LoRAField`s."""
    graph = Graph(nodes={"loras": LoRACollectionLoader(id="loras", loras=[LoRAField(lora=SD1_LORA, weight=0.5)])})
    assert [model.key for model in models_named_by(graph)] == ["sd1-lora"]


def test_a_switch_keeps_what_the_new_session_names_and_moves_the_rest() -> None:
    offload, cache = ArchitectureSwitchOffload(), _Cache()
    offload.before_session(_graph(SD1, SHARED_VAE), cache)  # type: ignore[arg-type]

    freed = offload.before_session(_graph(SD2, SHARED_VAE), cache)  # type: ignore[arg-type]

    assert cache.kept == [{"sd2-main", "shared-vae"}]
    assert freed == 3 * 2**30


def test_the_same_architecture_moves_nothing() -> None:
    offload, cache = ArchitectureSwitchOffload(), _Cache()
    offload.before_session(_graph(SD1), cache)  # type: ignore[arg-type]
    offload.before_session(_graph(SD1, SD1_LORA), cache)  # type: ignore[arg-type]
    assert cache.kept == []


def test_the_first_session_has_nothing_to_compare_with() -> None:
    offload, cache = ArchitectureSwitchOffload(), _Cache()
    offload.before_session(_graph(SD2), cache)  # type: ignore[arg-type]
    assert cache.kept == []


@pytest.mark.parametrize(
    ("third", "expected"),
    [(_graph(SD1), []), (_graph(SD2), [{"sd2-main"}])],
    ids=["same-architecture-after", "switch-after"],
)
def test_a_session_without_a_main_model_leaves_the_architecture_as_it_was(third: Graph, expected: list) -> None:
    """An upscale between two sessions neither makes the second one look like a switch nor hides one."""
    offload, cache = ArchitectureSwitchOffload(), _Cache()
    for graph in (_graph(SD1), _graph(None), third):
        offload.before_session(graph, cache)  # type: ignore[arg-type]
    assert cache.kept == expected


# --- the session runner --------------------------------------------------------------------------------------


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, **kwargs) -> None:
        self.warnings.append(message)

    def debug(self, message: str, **kwargs) -> None:
        pass


def _runner(load, enabled: bool = True) -> tuple[DefaultSessionRunner, _Logger]:
    runner = DefaultSessionRunner()
    logger = _Logger()
    runner._services = SimpleNamespace(  # type: ignore[assignment]
        model_manager=SimpleNamespace(load=load),
        logger=logger,
        configuration=SimpleNamespace(offload_on_architecture_switch=enabled),
    )
    runner._profiler = None
    return runner, logger


def _queue_item(graph: Graph) -> SimpleNamespace:
    return SimpleNamespace(item_id=1, session_id="s", session=SimpleNamespace(graph=graph))


def test_the_runner_offloads_through_its_devices_cache_before_a_switched_session() -> None:
    cache = _Cache()
    runner, logger = _runner(SimpleNamespace(ram_cache=cache))

    runner._on_before_run_session(_queue_item(_graph(SD1)))  # type: ignore[arg-type]
    runner._on_before_run_session(_queue_item(_graph(SD2)))  # type: ignore[arg-type]

    assert cache.kept == [{"sd2-main"}]
    assert logger.warnings == []


def test_a_failed_offload_is_logged_and_the_session_runs() -> None:
    class _Broken:
        def offload_models_from_vram_except(self, keep_model_keys) -> int:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    runner, logger = _runner(SimpleNamespace(ram_cache=_Broken()))
    runner._on_before_run_session(_queue_item(_graph(SD1)))  # type: ignore[arg-type]
    runner._on_before_run_session(_queue_item(_graph(SD2)))  # type: ignore[arg-type]

    assert len(logger.warnings) == 1


def test_the_setting_turns_it_off() -> None:
    cache = _Cache()
    runner, _ = _runner(SimpleNamespace(ram_cache=cache), enabled=False)

    runner._on_before_run_session(_queue_item(_graph(SD1)))  # type: ignore[arg-type]
    runner._on_before_run_session(_queue_item(_graph(SD2)))  # type: ignore[arg-type]

    assert cache.kept == []


def test_each_worker_judges_only_its_own_sessions() -> None:
    """Two GPUs running different architectures side by side: neither worker is switching."""
    first, _ = _runner(SimpleNamespace(ram_cache=_Cache()))
    second = DefaultSessionProcessor()._clone_session_runner(first)
    second._services = first._services  # type: ignore[attr-defined]
    second._profiler = None  # type: ignore[attr-defined]
    caches = {first: _Cache(), second: _Cache()}
    for _ in range(3):
        for runner, main in ((first, SD1), (second, SD2)):
            runner._services.model_manager.load.ram_cache = caches[runner]  # type: ignore[attr-defined]
            runner._on_before_run_session(_queue_item(_graph(main)))  # type: ignore[arg-type]
    assert caches[first].kept == [] and caches[second].kept == []
