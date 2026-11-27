"""Whether FP8 Storage does anything for a given kind of model, decided in one place.

The toggle was offered for every `main`, `controlnet` and `t2i_adapter` record, and honoured by
whichever loader happened to call the cast. Those are two different questions, and collapsing them
produced a control that changes nothing for 19 of the 52 loader keys it is shown for -- no crash, no
log line, just the same memory as before. Worse, identification *enables* it on its own for any
checkpoint whose denoiser stores float8 (`configs/factory.py`), so a Wan fp8 install used to record
`fp8_storage: true` and then load at bf16 size anyway.

Three things have to be true, and each is owned by a different piece of the system:

- **the model kind** -- only a model run as its own forward pass has anywhere to put the hooks, and
  only three types carry the setting at all (`FP8_STORAGE_MODEL_TYPES`);
- **the format** -- an already-quantized payload must never be re-encoded (`QUANTIZED_MODEL_FORMATS`);
- **the loader** -- it has to implement the cast, which it declares at registration.

The first two are policy and stay here. The third is a capability, so it is keyed the way loaders are
keyed -- `(base, type, format)`, `ModelLoaderRegistry`'s own key. That key is what a per-architecture
field cannot express: FLUX main casts and FLUX ControlNet does not, and both are `flux`.

`fp8_storage_verdict` composes all three, and everything that needs the answer calls it: the loader
gate (`ModelLoader._should_use_fp8`), identification (so it stops enabling what will not run), and
the API row a client reads. Shipping the three inputs separately would invite the next client to
rebuild the composition, differently.

**Not here: whether the device can do it.** `_device_supports_fp8_storage` probes the torch device
and is deliberately the last thing `_should_use_fp8` asks, so the probe never runs on an API or
install thread. It is an environment fact rather than a fact about the model, and folding it in would
put a device probe behind an HTTP handler.

**Not here: `fp8_compute`.** Keeping a checkpoint's weights packed for `_scaled_mm` is a second axis,
and nothing offers it as a control -- `should_keep_fp8_weights` reads the device. A served field
nobody reads is a contract nobody checks, so it is left until something asks.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from pydantic import BaseModel, Field

from invokeai.backend.model_manager.taxonomy import (
    QUANTIZED_MODEL_FORMATS,
    BaseModelType,
    ModelFormat,
    ModelType,
)


@dataclass(frozen=True)
class NotApplicable:
    """This loader will never implement the cast, and that is the right answer.

    An adapter too small for the saving to matter, or weights that are already one byte each.
    """

    reason: str


@dataclass(frozen=True)
class Unimplemented:
    """This loader could implement the cast and does not yet. A backlog entry, not a decision.

    Kept apart from `NotApplicable` so the gap is queryable -- filter `declared_fp8_storage()` for
    this class and you have the list, which shrinks in the code rather than ageing inside a planning
    document. Both markers hide the control identically; the distinction is for us, not the client.
    """

    reason: str


Fp8StorageDeclaration: TypeAlias = NotApplicable | Unimplemented | None
"""What a loader says about FP8 Storage at registration. `None` means "implemented"."""


FP8_STORAGE_MODEL_TYPES: frozenset[ModelType] = frozenset(
    {
        ModelType.Main,
        ModelType.ControlNet,
        ModelType.T2IAdapter,
    }
)
"""The model types FP8 Storage can reach, which is also the set whose settings carry the field.

Everything else is excluded for one of two reasons, and `test_fp8_capability.py` checks that the set
still matches the config schemas rather than trusting this comment:

- it has no `fp8_storage` in its default settings, so nothing can ask (VAE, LoRA, IP-Adapter, the
  encoders, ...). A VAE would be excluded anyway: fp8 storage degrades its decode visibly, which is
  why `_FP8_EXCLUDED_SUBMODEL_TYPES` also keeps it out of a pipeline's cast;
- it carries the field but is never run as its own forward pass. `ControlLoRa` shares
  `ControlAdapterDefaultSettings` with the control adapters, and is patched into a base model, so
  layerwise-casting hooks on it would never fire.
"""


@dataclass(frozen=True)
class Fp8StorageVerdict:
    """Whether the setting can do anything here, and what to say when it cannot."""

    supported: bool
    reason: str | None = None
    """Why not. `None` exactly when `supported`."""


class Fp8StorageSupport(BaseModel):
    """One row of the served table: a loader key and the finished answer.

    The reason is not served. Both markers hide the control identically, so a client has nothing to
    do with the difference; it stays in the declaration, where the backlog lives.
    """

    base: BaseModelType
    type: ModelType
    format: ModelFormat
    supported: bool = Field(description="Whether FP8 Storage changes anything for a model of this kind.")


_declared: dict[tuple[BaseModelType, ModelType, ModelFormat], Fp8StorageDeclaration] = {}
"""Every registered loader key, with what it declared. Written by `ModelLoaderRegistry.register`.

Holding *every* key, including the implemented ones whose value is `None`, is what lets this module
serve the table without importing the registry back -- and the registry is the one thing that knows
which keys exist.
"""


def declare_fp8_storage(
    base: BaseModelType,
    type: ModelType,
    format: ModelFormat,
    declaration: Fp8StorageDeclaration,
) -> None:
    """Record what a loader declared. Called once per registration, including for `None`."""
    _declared[(base, type, format)] = declaration


def _ensure_loaders_registered() -> None:
    """Import the loader modules, and refuse to answer from an empty table.

    The import is function-local because `model_loader_registry` imports this module and the reverse
    edge at module scope would close a cycle. It is also nearly always a no-op: this module lives
    *inside* the package it imports, so anything that can call this has already run
    `load/__init__.py`, which globs `model_loaders/*.py`. It is kept for the one caller that reaches
    here through `configs.factory` rather than through the package, and because it states the
    dependency.

    The check is the part that earns its place. There is exactly one window where `_declared` is
    incomplete -- a re-entrant read from inside `load/__init__.py`, where the import above is a
    `sys.modules` hit on the half-built package -- and in that window every lookup would answer "no
    loader is registered", which is indistinguishable from the truthful answer for an unregistered
    key. Failing loudly is the only way that stays visible.
    """
    import invokeai.backend.model_manager.load  # noqa: F401  (its __init__ globs model_loaders/*.py)

    if not _declared:
        raise RuntimeError(
            "No model loaders have registered, so FP8 Storage support cannot be decided. This means "
            "`fp8_capability` was read while `invokeai.backend.model_manager.load` was still importing."
        )


def fp8_storage_verdict(base: BaseModelType, type: ModelType, format: ModelFormat) -> Fp8StorageVerdict:
    """Whether FP8 Storage would change anything for a model of this kind.

    Ordered so the reported reason is the most fundamental one: a GGUF ControlLoRa is refused for
    being patched in, not for being packed. An unregistered key answers no -- nothing can load it.
    """
    if type not in FP8_STORAGE_MODEL_TYPES:
        # Deliberately one sentence for the whole set rather than a per-type explanation: the grounds
        # differ (a VAE is excluded because fp8 visibly degrades decode, a LoRA because it is patched
        # into a base model and the hooks would never fire, everything else because it carries no such
        # setting), and a single invented reason would be wrong about most of them. The grounds are in
        # `FP8_STORAGE_MODEL_TYPES`, where they can be read next to the set they justify.
        return Fp8StorageVerdict(
            supported=False,
            reason=f"FP8 Storage is not offered for {type.value} models.",
        )

    # Their weights are packed integer payloads, not values anything may re-encode, and casting them
    # is not a no-op: GGUF raises `Operation changed the dtype of GGMLTensor unexpectedly`, while bnb
    # NF4 corrupts *silently* -- `bnb.nn.LinearNF4` subclasses `nn.Linear`, so the packed uint8
    # payload is cast to float8, inference still returns finite numbers, and the model produces
    # garbage. No quantized-format loader implements the cast, so this stays a rule rather than a
    # per-loader declaration: it holds for the next one too.
    if format in QUANTIZED_MODEL_FORMATS:
        return Fp8StorageVerdict(
            supported=False,
            reason=(
                f"{format.value} weights are already a packed quantized payload, which FP8 Storage must not re-encode."
            ),
        )

    _ensure_loaders_registered()
    # Exact key first, then the `Any`-base one, exactly as `ModelLoaderRegistry.get_implementation`
    # resolves a loader. Without the fallback every T2I adapter answers "no loader registered": they are
    # served by `GenericDiffusersLoader` under `any/t2i_adapter/diffusers`, and no record carries `any`
    # as its own base.
    for key in ((base, type, format), (BaseModelType.Any, type, format)):
        if key in _declared:
            declaration = _declared[key]
            if declaration is None:
                return Fp8StorageVerdict(supported=True)
            return Fp8StorageVerdict(supported=False, reason=declaration.reason)

    return Fp8StorageVerdict(
        supported=False,
        reason=f"No loader is registered for {base.value}/{type.value}/{format.value}.",
    )


def fp8_storage_support() -> list[Fp8StorageSupport]:
    """The table a client joins against its model records, sorted so the response is diffable.

    Only the three types that can carry the setting get a row; for anything else the answer is no
    without a loader having to say so, and a row would be a field nobody reads.

    A row whose base is `any` is a wildcard registration, and a client has to resolve it the way the
    registry does: exact `(base, type, format)` first, then `(any, type, format)`. It is served as the
    wildcard rather than expanded over every base because most bases have no such model, and a row per
    base would invent twenty answers to hold one. A key that matches neither is not supported -- which is
    also what `fp8_storage_verdict` concludes for it.
    """
    _ensure_loaders_registered()
    rows = [
        Fp8StorageSupport(
            base=base,
            type=type,
            format=format,
            supported=fp8_storage_verdict(base, type, format).supported,
        )
        for (base, type, format) in _declared
        if type in FP8_STORAGE_MODEL_TYPES
    ]
    return sorted(rows, key=lambda row: (row.type.value, row.base.value, row.format.value))


def declared_fp8_storage() -> Mapping[tuple[BaseModelType, ModelType, ModelFormat], Fp8StorageDeclaration]:
    """Every registered loader key and what it declared, read-only.

    This is where §6's "5 of 20" lives now: filtering for `Unimplemented` is the backlog, as a query
    rather than a number in a document. Exposed rather than left private because the count is meant
    to be asked for -- by the test that keeps the declarations honest, and by anyone wondering what
    is left.
    """
    _ensure_loaders_registered()
    return MappingProxyType(_declared)
