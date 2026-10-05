"""Every registry entry must be an instantiable loader class.

`ModelLoaderRegistry.register` does no runtime type check and returns its argument unchanged, so it
happily decorates whatever follows it. Insert a helper function between the decorator and the class
it was meant to decorate — a plausible outcome of extracting a loop into a named function — and the
registry silently binds the helper instead. The helper keeps working everywhere it is called
directly, the module imports, ruff is clean and the whole suite passes; the only symptom is that
`ModelLoadService.load_model` calls `implementation(app_config=..., logger=..., ram_cache=...)` and
gets `TypeError: <helper>() got an unexpected keyword argument 'app_config'` — i.e. that model type
can no longer be loaded at all.

This happened to `Qwen3EncoderCheckpointLoader`. One assertion over the whole registry closes the
class, so it cannot happen again to a loader nobody has an end-to-end test for.
"""

import importlib
import inspect
import pkgutil

import invokeai.backend.model_manager.load.model_loaders as model_loaders_pkg
from invokeai.backend.model_manager.load.load_base import ModelLoaderBase
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry


def _import_every_loader_module() -> None:
    """Registration is an import side effect, so nothing is in the registry until this runs."""
    for module in pkgutil.iter_modules(model_loaders_pkg.__path__):
        importlib.import_module(f"{model_loaders_pkg.__name__}.{module.name}")


def test_every_registered_implementation_is_a_loader_class() -> None:
    _import_every_loader_module()
    assert ModelLoaderRegistry._registry, "no loaders registered — the import sweep above is broken"

    not_classes = {key: impl for key, impl in ModelLoaderRegistry._registry.items() if not inspect.isclass(impl)}
    assert not not_classes, (
        "these registry keys are bound to something that is not a class, so instantiating them at "
        f"load time raises TypeError: {not_classes}"
    )


def test_every_registered_implementation_subclasses_the_loader_base() -> None:
    """A class is not enough on its own — `load_model` also needs the base's constructor signature."""
    _import_every_loader_module()

    wrong_base = {
        key: impl for key, impl in ModelLoaderRegistry._registry.items() if not issubclass(impl, ModelLoaderBase)
    }
    assert not wrong_base, f"registered implementations that are not ModelLoaderBase subclasses: {wrong_base}"
