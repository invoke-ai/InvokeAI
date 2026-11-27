"""Registry mechanics, exercised against throwaway facets rather than the real ones.

Using dummy facets keeps these tests from restating what production declares — otherwise they would
fail on every legitimate change to an architecture, and pass for the wrong reason when a facet is
quietly dropped.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from invokeai.backend.architectures import registry
from invokeai.backend.architectures.facet import Facet
from invokeai.backend.model_manager.taxonomy import BaseModelType


@dataclass(frozen=True)
class _Colour(Facet):
    name: str


@dataclass(frozen=True)
class _Size(Facet):
    value: int


@dataclass(frozen=True)
class _Mandatory(Facet):
    REQUIRED = True


FACETS_MODULE = "invokeai.backend.architectures.facets"


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An empty registry, and an empty facet-type collection.

    Both have to be reset. `Facet.FACET_TYPES` is class state filled at import time by every facet
    in the codebase, so leaving it alone would let a real `REQUIRED` facet leak into `validate()`
    here and fail against architectures this test never registered.

    The doubles above go the other way: `Facet.__init_subclass__` added them to the real collection
    at import time, and nothing here takes them back out, because `validate()` only enforces facets
    declared under `invokeai.backend.architectures.facets` — which is what
    `test_a_required_facet_from_elsewhere_is_not_enforced` below pins.
    """
    monkeypatch.setattr(registry, "_ARCHITECTURES", {})
    monkeypatch.setattr(Facet, "FACET_TYPES", {})
    yield


def _register_every_architecture() -> None:
    for base in BaseModelType:
        if base not in registry._NOT_ARCHITECTURES:
            registry.register(base)


def test_register_then_get_round_trips(isolated_registry: None) -> None:
    registry.register(BaseModelType.Flux, _Colour("blue"), _Size(3))
    assert registry.get(BaseModelType.Flux, _Colour) == _Colour("blue")
    assert registry.get(BaseModelType.Flux, _Size) == _Size(3)


def test_get_returns_none_for_an_undeclared_facet(isolated_registry: None) -> None:
    registry.register(BaseModelType.Flux, _Colour("blue"))
    assert registry.get(BaseModelType.Flux, _Size) is None


def test_get_returns_none_for_an_unregistered_base(isolated_registry: None) -> None:
    assert registry.get(BaseModelType.Wan, _Colour) is None


def test_facets_of_returns_everything_declared(isolated_registry: None) -> None:
    registry.register(BaseModelType.Flux, _Colour("blue"), _Size(3))
    assert set(registry.facets_of(BaseModelType.Flux)) == {_Colour("blue"), _Size(3)}


def test_registering_a_different_declaration_raises(isolated_registry: None) -> None:
    registry.register(BaseModelType.Flux, _Colour("blue"))
    with pytest.raises(registry.ArchitectureError, match="already registered"):
        registry.register(BaseModelType.Flux, _Colour("red"))


def test_re_registering_an_identical_declaration_is_a_no_op(isolated_registry: None) -> None:
    """A `defs/` module re-executed by `importlib.reload`, or by jurigged under `--dev_reload`.

    Without this, a developer's next save on any file under `defs/` crashes the running server with
    a message blaming a file that is correct.
    """
    registry.register(BaseModelType.Flux, _Colour("blue"), _Size(3))
    registry.register(BaseModelType.Flux, _Colour("blue"), _Size(3))

    assert set(registry.facets_of(BaseModelType.Flux)) == {_Colour("blue"), _Size(3)}


def test_re_registering_with_a_facet_dropped_raises(isolated_registry: None) -> None:
    """The subtler half: same base, same remaining facets, one gone. Still a collision."""
    registry.register(BaseModelType.Flux, _Colour("blue"), _Size(3))
    with pytest.raises(registry.ArchitectureError, match="already registered"):
        registry.register(BaseModelType.Flux, _Colour("blue"))

    assert set(registry.facets_of(BaseModelType.Flux)) == {_Colour("blue"), _Size(3)}, "the first one stands"


def test_the_same_facet_type_twice_in_one_call_raises(isolated_registry: None) -> None:
    with pytest.raises(registry.ArchitectureError, match="more than once"):
        registry.register(BaseModelType.Flux, _Colour("blue"), _Colour("red"))


@pytest.mark.parametrize("base", [BaseModelType.Any, BaseModelType.External, BaseModelType.Unknown])
def test_a_sentinel_cannot_be_registered(isolated_registry: None, base: BaseModelType) -> None:
    with pytest.raises(registry.ArchitectureError, match="not a model architecture"):
        registry.register(base, _Colour("blue"))


def test_require_names_the_file_to_edit_when_the_facet_is_missing(isolated_registry: None) -> None:
    registry.register(BaseModelType.ZImage, _Colour("blue"))
    with pytest.raises(registry.ArchitectureError) as exc:
        registry.require(BaseModelType.ZImage, _Size)
    assert "invokeai/backend/architectures/defs/z_image.py" in str(exc.value)
    assert "_Size" in str(exc.value)


def test_require_says_how_to_create_a_missing_architecture(isolated_registry: None) -> None:
    """The other half of the message: there is no file yet, so say what to put in it."""
    with pytest.raises(registry.ArchitectureError) as exc:
        registry.require(BaseModelType.ZImage, _Size)
    message = str(exc.value)
    assert "invokeai/backend/architectures/defs/z_image.py" in message
    assert "register(BaseModelType.ZImage, _Size(...))" in message
    assert "no import list to edit" in message


def test_architecture_error_is_a_value_error() -> None:
    """It replaces `raise ValueError("Unsupported base model: ...")`, so handlers must still catch."""
    assert issubclass(registry.ArchitectureError, ValueError)


def test_validate_reports_an_undeclared_required_facet(
    isolated_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative probe for the boot gate. Without this, `validate()` could be a no-op forever.

    The double has to claim a module inside the facets package, because that is what `validate()`
    enforces; the test below is the other side of that filter.
    """
    _register_every_architecture()
    monkeypatch.setattr(_Mandatory, "__module__", f"{FACETS_MODULE}.mandatory")
    Facet.FACET_TYPES[_Mandatory] = None

    with pytest.raises(registry.ArchitectureError) as exc:
        registry.validate()
    assert "_Mandatory" in str(exc.value)
    assert "invokeai/backend/architectures/defs/flux.py" in str(exc.value)


def test_a_required_facet_from_elsewhere_is_not_enforced(isolated_registry: None) -> None:
    """A custom-node pack that merely *defines* a REQUIRED facet must not take the app down.

    `Facet.FACET_TYPES` collects every subclass created anywhere in the interpreter, and
    `load_custom_nodes` swallows a pack's own exceptions — so such a pack would load
    "successfully" and then fail boot with a wall of errors naming first-party files under `defs/`.
    `_Mandatory` is defined in this test module, which stands in for exactly that third-party facet.
    """
    _register_every_architecture()
    assert _Mandatory.__module__ == __name__ and not __name__.startswith(FACETS_MODULE)
    Facet.FACET_TYPES[_Mandatory] = None

    registry.validate()


def test_validate_reports_an_unregistered_architecture(isolated_registry: None) -> None:
    """Discovery is automatic, so the way to fail is a new enum member with no module under defs/."""
    for base in BaseModelType:
        if base not in registry._NOT_ARCHITECTURES and base is not BaseModelType.Wan:
            registry.register(base)

    with pytest.raises(registry.ArchitectureError) as exc:
        registry.validate()
    assert "'wan'" in str(exc.value)
