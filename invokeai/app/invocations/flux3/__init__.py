"""Flux3 nodes.

Node modules go here, keeping their architecture prefix: `flux3_denoise.py`,
`flux3_model_loader.py`. VAE, text-encoder and PiD nodes belong in
`invocations/vae/`, `invocations/text_encoder/` and `invocations/pid/` instead — they are shared
across architectures rather than owned by one.

This package is discovered automatically; there is no list to add it to. The `__init__.py` is what
makes it a package, and without it every node in here would silently not exist.
"""
