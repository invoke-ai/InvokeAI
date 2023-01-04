---
title: Python Tools
---

# :fontawesome-brands-python: Python Tools

## [pip-compile](https://pip-tools.readthedocs.io)

Creating requirement files for production has never been that easy!!!

## build

Building the wheel is as simple as:

1. `pip install ".[dev]"` (I recommend doing so in a venv)
2. `python -m build`

Afterwards you will have a `dist` folder in the repository root, containing the wheel.
