"""The conditioning facet, and the `safe_globals` list built from it."""

import ast
from pathlib import Path

from invokeai.backend.architectures import conditioning_infos, conditioning_safe_globals, generative_bases
from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    ConditioningFieldData,
    FLUXConditioningInfo,
    IPAdapterConditioningInfo,
    SDXLConditioningInfo,
)

DEPENDENCIES = Path(__file__).parents[3] / "invokeai" / "app" / "api" / "dependencies.py"


def test_every_architecture_declares_a_conditioning_type() -> None:
    undeclared = sorted(b.value for b in generative_bases() if get(b, ConditioningFacet) is None)
    assert undeclared == []


def test_the_shared_types_are_shared_by_identity() -> None:
    """Three pairs share a class. Asserted by identity, so an equal-looking copy would fail."""
    for bases, info in (
        ((BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2), BasicConditioningInfo),
        ((BaseModelType.StableDiffusionXL, BaseModelType.StableDiffusionXLRefiner), SDXLConditioningInfo),
        ((BaseModelType.Flux, BaseModelType.Flux2), FLUXConditioningInfo),
    ):
        for base in bases:
            facet = get(base, ConditioningFacet)
            assert facet is not None and facet.info is info, base.value


def test_fourteen_types_serve_seventeen_architectures() -> None:
    """Pins the sharing itself. A fourteenth type means a new architecture stopped sharing."""
    assert len(conditioning_infos()) == 14
    assert len(generative_bases()) == 17


def test_the_list_is_deterministic() -> None:
    """`safe_globals` should be diffable, and registry order is only incidentally stable."""
    names = [cls.__name__ for cls in conditioning_infos()]
    assert names == sorted(names)


def test_the_declared_classes_are_the_ones_that_get_serialized() -> None:
    """Each facet holds a class, not a name — so this can check the object, not a string.

    `ConditioningFieldData` itself is added separately by the caller, and
    `IPAdapterConditioningInfo` is deliberately absent: it is built in memory and handed to the
    pipeline, never written through `context.conditioning.save`, so it is not unpickled and does not
    need to be a safe global.
    """
    infos = set(conditioning_infos())
    assert IPAdapterConditioningInfo not in infos
    assert all(isinstance(cls, type) for cls in infos)


def _dependencies_ast() -> ast.Module:
    return ast.parse(DEPENDENCIES.read_text(encoding="utf-8"))


def _conditioning_serializer_call() -> ast.Call:
    """The `ObjectSerializerDisk[ConditioningFieldData](...)` call in `dependencies.py`.

    Found by its type parameter, which is what distinguishes it from the tensor serializer built
    three lines above it.
    """
    calls = [
        node
        for node in ast.walk(_dependencies_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ObjectSerializerDisk"
        and isinstance(node.func.slice, ast.Name)
        and node.func.slice.id == "ConditioningFieldData"
    ]
    assert len(calls) == 1, f"expected exactly one conditioning serializer in {DEPENDENCIES.name}, found {len(calls)}"
    return calls[0]


def test_dependencies_passes_the_assembled_list_as_safe_globals() -> None:
    """Read out of `dependencies.py` itself, because this is the assertion the PR exists for.

    Importing `dependencies` to inspect the built list is not an option — it pulls in the whole
    service graph — and rebuilding the list here would only ever agree with itself: shortening the
    call site to `[ConditioningFieldData]` would leave such a test green and the first FLUX, Wan or
    Qwen generation dying mid-graph on an `UnpicklingError`. So the call site is parsed instead.
    """
    keywords = [kw for kw in _conditioning_serializer_call().keywords if kw.arg == "safe_globals"]
    assert len(keywords) == 1, "safe_globals is not passed by keyword"

    argument = keywords[0].value
    assert isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name), (
        "safe_globals must be the assembled list itself, not an expression built at the call site"
    )
    assert argument.func.id == "conditioning_safe_globals"
    assert not argument.args and not argument.keywords


def test_dependencies_imports_that_list_from_the_architecture_package() -> None:
    """Otherwise the name asserted above could be satisfied by anything defined locally."""
    imported = {
        alias.name
        for node in ast.walk(_dependencies_ast())
        if isinstance(node, ast.ImportFrom) and node.module == "invokeai.backend.architectures"
        for alias in node.names
    }
    assert "conditioning_safe_globals" in imported


def test_the_assembled_list_holds_every_conditioning_class_exactly_once() -> None:
    """The content of what the call site passes: the envelope first, then all thirteen classes."""
    safe_globals = conditioning_safe_globals()

    assert safe_globals[0] is ConditioningFieldData
    assert set(safe_globals[1:]) == set(conditioning_infos())
    assert len(safe_globals) == 15
    assert len(set(safe_globals)) == len(safe_globals), "a class appears twice"


def test_the_node_api_exports_every_declared_conditioning_type() -> None:
    """Custom node authors build these; the public surface must offer all of them.

    Derived from the registry rather than written down, because a hand-kept list is exactly what
    goes stale: the version of this list on the abandoned branch already omitted MiniMax H3 by the
    time it was ported. `invocation_api` still needs the static imports -- `__all__` is a real
    re-export, not a runtime lookup -- so this test is what keeps the two in step.
    """
    import invokeai.invocation_api as node_api

    declared = {info.__name__ for info in conditioning_infos()}
    exported = set(node_api.__all__)

    assert declared <= exported, sorted(declared - exported)
    for name in sorted(declared):
        assert getattr(node_api, name, None) is not None, f"{name} is in __all__ but not importable"
