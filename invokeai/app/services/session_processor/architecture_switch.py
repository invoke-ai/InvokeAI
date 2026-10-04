"""Clear VRAM of the previous architecture's models when a worker's sessions switch architecture.

The model cache otherwise leaves every model where it is until a reservation needs its VRAM, and a reservation
evicts only as much as the next node estimates, so after a run of different architectures the card stays nearly full
of models the current one will not use. Measured on an RTX 4090 after SDXL, FLUX.1, FLUX.2 Klein, Z-Image and Krea-2:
with the offload, warm runs were 1-2 s faster (Krea-2 14.1 -> 12.1 s) and the GPU peak was 14-19 GB instead of
19-23 GB. When two architectures that fit in VRAM together alternate, each switch costs a reload from RAM instead
(SDXL 5.8 -> 7.4 s, Z-Image 7.8 -> 10.9 s), which the `offload_on_architecture_switch` setting turns off.

A session's architecture is the set of bases of the main models its graph names. When it differs from the previous
session's on the same worker, every cached model the new graph does not name moves from VRAM to RAM. Models the new
session names stay, so an encoder or VAE that two architectures share (Qwen3 4B for Z-Image and FLUX.2 Klein, the
FLUX.1 autoencoder for Z-Image) is not reloaded.
"""

from typing import Any, Iterator, Optional

from pydantic import BaseModel

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.graph_validation import Graph
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.logging import InvokeAILogger

GB = 2**30


def models_named_by(graph: Graph) -> list[ModelIdentifierField]:
    """Every model a graph's node inputs name, at any depth (a LoRA field, a loader's model field, ...)."""
    return [model for node in graph.nodes.values() for model in _models_in(node)]


def _models_in(value: Any) -> Iterator[ModelIdentifierField]:
    if isinstance(value, ModelIdentifierField):
        yield value
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _models_in(getattr(value, name, None))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _models_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _models_in(item)


class ArchitectureSwitchOffload:
    """One worker's view of which architecture it last ran, and the offload when that changes."""

    def __init__(self) -> None:
        self._last_bases: Optional[frozenset[BaseModelType]] = None

    def before_session(self, graph: Graph, cache: ModelCache) -> int:
        """Offload what `graph` does not name if its architecture differs from the last session's. Returns bytes freed.

        A session that names no main model (an upscale, a utility workflow) decides nothing and leaves the last
        architecture as it was. The first session after startup has nothing to compare with.
        """
        models = models_named_by(graph)
        bases = frozenset(model.base for model in models if model.type is ModelType.Main)
        if not bases:
            return 0
        previous, self._last_bases = self._last_bases, bases
        if previous is None or bases == previous:
            return 0
        freed = cache.offload_models_from_vram_except({model.key for model in models})
        if freed > 0:
            InvokeAILogger.get_logger(__name__).info(
                f"Switched from {_names(previous)} to {_names(bases)} models on {cache.execution_device}: moved "
                f"{freed / GB:.1f} GB of models this session does not use from VRAM to RAM."
            )
        return freed


def _names(bases: frozenset[BaseModelType]) -> str:
    return "/".join(sorted(base.value for base in bases))
