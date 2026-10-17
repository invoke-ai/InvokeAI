"""QwenImage21 nodes.

Node modules go here, keeping their architecture prefix: `qwen_image_2_1_denoise.py`,
`qwen_image_2_1_model_loader.py`. VAE, text-encoder and PiD nodes belong in
`invocations/vae/`, `invocations/text_encoder/` and `invocations/pid/` instead — they are shared
across architectures rather than owned by one.

This package is discovered automatically; there is no list to add it to. The `__init__.py` is what
makes it a package, and without it every node in here would silently not exist.
"""
