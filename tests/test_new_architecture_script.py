"""The scaffolder generates valid stubs, and its residual list stays honest."""

import ast
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from invokeai.backend.architectures import registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    """Load a `scripts/` module by path, as `test_check_pins.py` and `test_docs_json_export.py` do.

    `scripts/` is not a package and is not importable. Putting it on `sys.path` instead would leave
    it there for every other file the xdist worker runs afterwards, where a script's name could
    shadow a real module.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new_architecture = _load_module(REPO_ROOT / "scripts" / "new_architecture.py", "new_architecture")
defs_module = new_architecture.defs_module
derive_residual_edits = new_architecture.derive_residual_edits
invocations_package = new_architecture.invocations_package
planned_files = new_architecture.planned_files
starter_models_module = new_architecture.starter_models_module

REQUIRED_FACETS = {"LatentSpaceFacet", "ConditioningFacet", "DefaultSettingsFacet", "ModalityFacet", "FeaturesFacet"}


@pytest.mark.parametrize("render", [defs_module, invocations_package, starter_models_module], ids=lambda f: f.__name__)
def test_every_stub_is_valid_python(render) -> None:  # type: ignore[no-untyped-def]
    ast.parse(render("new-model", "NewModel"))


def test_the_declaration_carries_every_required_facet() -> None:
    """A stub missing one would fail at boot with a message about that facet rather than about the
    stub, which is a worse first experience than the TODOs."""
    tree = ast.parse(defs_module("new-model", "NewModel"))
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert REQUIRED_FACETS <= called, sorted(REQUIRED_FACETS - called)


def test_the_generated_declaration_refuses_to_be_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scaffolder's central claim, executed.

    Its docstring promises the stub "fails loudly at boot until someone fills it in". `validate()`
    alone does not deliver that — it checks only that the required facets are present, and the stub
    declares all five with values that pass their own constructors. The all-zero latent projection
    is what makes the promise true: it raises while the module is being executed, which for a file
    under `defs/` is during boot.

    The module is executed with `register` replaced, so a stub that ever stopped raising could not
    pass this by reaching the real registry instead.
    """
    registered: list[tuple[Any, ...]] = []
    monkeypatch.setattr(registry, "register", lambda *args: registered.append(args))

    with pytest.raises(registry.ArchitectureError, match="rgb_factors is all zeros"):
        exec(  # noqa: S102 -- executing the generated module *is* the behaviour under test
            compile(defs_module("new-model", "NewModel"), "defs/new_model.py", "exec"),
            {"__name__": "invokeai.backend.architectures.defs.new_model"},
        )

    assert registered == [], "the stub reached register(), so a boot would have accepted it"


def test_the_slug_convention_matches_the_registry() -> None:
    """`-` becomes `_`, the same rule `registry.defs_module_path` computes."""
    paths = set(planned_files("new-model", "NewModel"))
    assert "invokeai/backend/architectures/defs/new_model.py" in paths
    assert "invokeai/app/invocations/new_model/__init__.py" in paths
    assert "invokeai/backend/model_manager/starter_models/new_model.py" in paths


def test_the_residual_list_is_derived_from_the_tree() -> None:
    """Not a written-down list. It has already shrunk twice while this series ran, and a hardcoded
    one would still be naming files that no longer dispatch on base."""
    residual = dict(derive_residual_edits(REPO_ROOT / "invokeai"))

    # Still dispatching: one config class per architecture is inherent to how configs work.
    assert "invokeai/backend/model_manager/configs/main.py" in residual

    # No longer dispatching — each of these was a chain the registry absorbed. If one reappears
    # here, a facet has been bypassed. Checked at threshold=1, not at the reporting threshold: a
    # reintroduced four-branch `if/elif` in step_callback.py is exactly the regression this guards,
    # and it would sit under a five-base floor unnoticed.
    every_module_naming_a_base = dict(derive_residual_edits(REPO_ROOT / "invokeai", threshold=1))
    for absorbed in (
        "invokeai/app/util/step_callback.py",
        "invokeai/app/api/dependencies.py",
        "invokeai/app/invocations/ideal_size.py",
    ):
        assert absorbed not in every_module_naming_a_base, f"{absorbed} dispatches on base again"


def test_the_registry_itself_is_never_listed_as_a_cost() -> None:
    """`architectures/` names every base by construction; listing it would drown the real entries."""
    residual = dict(derive_residual_edits(REPO_ROOT / "invokeai"))
    assert not [path for path in residual if path.startswith("invokeai/backend/architectures/")]
