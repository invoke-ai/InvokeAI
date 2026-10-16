"""FLUX.3 Image (Klein and Klein Base).

``transformer_flux3.py`` is vendored from huggingface/diffusers at commit
7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941 ("Flux3 image (#15046)"), which is not yet in any tagged diffusers
release. The only local change is rewriting the package-relative imports to absolute ``diffusers.*`` imports
(every referenced symbol exists in the pinned diffusers==0.41.0) and ruff import sorting. Keep it otherwise
byte-identical to upstream: when a diffusers release ships ``Flux3Transformer2DModel``, delete the vendored
file and import the class from diffusers here instead. Everything imports it from this package, so this is the
only import to change; ``tests/backend/flux3/test_vendored_transformer.py`` fails on that bump and lists the rest.

The other modules of this package are first-party. Where they port the reference implementation
(github.com/black-forest-labs/flux-3-image-klein at 45e57c9fb06f9751cdced29a65c5d2171003be05, Apache-2.0),
the module says so.

Apache-2.0, copyright Black Forest Labs and The HuggingFace Team (see the file header).
"""

from invokeai.backend.flux3.transformer_flux3 import Flux3Transformer2DModel

__all__ = ["Flux3Transformer2DModel"]
