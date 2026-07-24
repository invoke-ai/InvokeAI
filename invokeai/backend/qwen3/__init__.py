"""Qwen3 encoder backend module.

Shared assets for the standalone Qwen3 text encoders used by Z-Image (4B/8B) and
Anima (0.6B). The Qwen3 BPE tokenizer is identical across all variants, so a single
vendored copy is bundled here and reused by every Qwen3 encoder loader.
"""
