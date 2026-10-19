"""Registry mechanics, against throwaway specs rather than the shipped ones."""

import pytest

from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.spec import EstimatorSpec, register, registered_specs


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spec_module, "_SPECS", {})


def _module_terms():
    """A fresh `terms` function on every call, with one module and qualified name: what `importlib.reload` of its
    defining module produces."""

    def terms(features, values):
        return ()

    return terms


def _another_estimators_terms(features, values):
    return ()


def _spec(terms, version: int = 1) -> EstimatorSpec:
    return EstimatorSpec(
        id="test.estimator",
        version=version,
        coefficients=frozenset({"k"}),
        platform_keys=("conv_backend",),
        terms=terms,
    )


def test_re_executing_the_defining_module_adopts_its_new_terms() -> None:
    """`importlib.reload`: the same declaration from the re-executed function."""
    before, after = _module_terms(), _module_terms()
    register(_spec(before))
    register(_spec(after))

    assert registered_specs()["test.estimator"].terms is after


def test_a_differing_declaration_under_a_taken_id_raises() -> None:
    register(_spec(_module_terms()))
    with pytest.raises(ValueError, match="already registered"):
        register(_spec(_module_terms(), version=2))

    assert registered_specs()["test.estimator"].version == 1, "the first one stands"


def test_the_same_declaration_from_another_function_raises() -> None:
    """`terms` is not part of a spec's equality, so this is the case equality alone would let through."""
    first = _module_terms()
    register(_spec(first))
    with pytest.raises(ValueError, match="ids are unique"):
        register(_spec(_another_estimators_terms))

    assert registered_specs()["test.estimator"].terms is first


def test_the_same_function_registered_twice_raises() -> None:
    """Two specs of one module sharing a `terms` function -- as SD3 and CogView4 do -- under one id. `importlib.reload`
    makes new function objects, so it never hands over the same one twice; jurigged re-running an edited `register`
    statement does, and raises too (see `register`)."""
    terms = _module_terms()
    register(_spec(terms))
    with pytest.raises(ValueError, match="ids are unique"):
        register(_spec(terms))
