"""Vendored TripoSplat inference code — single image to 3D Gaussian splat.

Source: https://github.com/VAST-AI-Research/TripoSplat (MIT License, see LICENSE in this directory).

The only change from upstream is that the two cross-module imports were made package-relative so the
two files import correctly as a package:
  - triposplat.py:  `from model import (...)`            -> `from .model import (...)`
  - model.py:       `from triposplat import _build_gaussians` -> `from .triposplat import _build_gaussians`

Import `TripoSplatPipeline` from `.triposplat` lazily (it pulls in torch/torchvision).
"""
