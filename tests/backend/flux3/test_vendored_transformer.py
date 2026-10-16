"""The vendored FLUX.3 transformer is a stopgap until a diffusers release ships it.

Everything in InvokeAI imports `Flux3Transformer2DModel` from `invokeai.backend.flux3`, so moving to the released
class is a change in one place. This test is what makes that move happen: it fails on the diffusers bump that
brings the class, instead of the vendored copy outliving its reason (as MiniMax-H3's did).
"""

import diffusers


def test_diffusers_does_not_ship_flux3_yet() -> None:
    assert not hasattr(diffusers, "Flux3Transformer2DModel"), (
        f"diffusers {diffusers.__version__} ships Flux3Transformer2DModel. Import it from diffusers in "
        "invokeai/backend/flux3/__init__.py, delete invokeai/backend/flux3/transformer_flux3.py and its entry in "
        "pyproject.toml's [tool.ruff.format] exclude, and drop the Flux3 classes from _PINNED_MODEL_CLASS_MARKERS "
        "(configs/factory.py). The key conversion, configs, loaders and nodes do not change."
    )
