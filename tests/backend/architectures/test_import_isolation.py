"""Importing `dependencies` alone fills the registry.

This is the constraint the module-scope import in `invokeai/app/api/dependencies.py` exists for, and
it cannot be checked in-process: by the time any test runs, half the codebase has been imported and
the registry would be full no matter where the import sat. A fresh interpreter is the only way to
tell a module-scope import from a lazy one inside a function.

The check matters because `ApiDependencies.initialize()` builds `ObjectSerializerDisk`, which mutates
process-global torch state via `add_safe_globals`. Anything the registry is meant to contribute there
has to be registered before that point, not on first use.

Deliberately not marked `slow`: pytest's `addopts` carries `-m "not slow"`, so a slow marker would
mean this never runs in CI.
"""

from tests.dangerously_run_function_in_subprocess import dangerously_run_function_in_subprocess


def _registry_is_full_after_importing_dependencies() -> None:
    # No arguments and no closure: the whole function body is re-executed in a fresh interpreter, so
    # every name it uses has to be imported inside it.
    import sys

    import invokeai.app.api.dependencies  # noqa: F401

    # `sys.modules` first, and this ordering is the whole test. Importing anything under
    # `invokeai.backend.architectures` runs that package's `__init__`, which fills the registry as a
    # side effect — so reading the registry before this check would make the assertion true no
    # matter where `dependencies` puts its import, or whether it has one at all.
    assert "invokeai.backend.architectures.defs" in sys.modules, (
        "importing invokeai.app.api.dependencies did not import the architecture registry. "
        "Its import must stay at module scope; a lazy one inside initialize() is too late."
    )

    from invokeai.backend.architectures.registry import _ARCHITECTURES, _NOT_ARCHITECTURES
    from invokeai.backend.model_manager.taxonomy import BaseModelType

    expected = set(BaseModelType) - _NOT_ARCHITECTURES
    missing = sorted(b.value for b in expected - set(_ARCHITECTURES))
    assert not missing, f"not registered after importing dependencies: {missing}"


def test_importing_dependencies_fills_the_registry() -> None:
    _stdout, stderr, returncode = dangerously_run_function_in_subprocess(_registry_is_full_after_importing_dependencies)
    # Asserted on the return code, not on empty stderr: importing this much of the codebase emits
    # library deprecation warnings and an "InvokeAI" log line, none of which are failures.
    assert returncode == 0, stderr
