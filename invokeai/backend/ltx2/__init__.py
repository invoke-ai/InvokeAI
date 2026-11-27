"""LTX-2 (Lightricks) audio-video generation: model-manager glue and the inference recipe.

The model classes come from the installed diffusers (``LTX2VideoTransformer3DModel``,
``AutoencoderKLLTX2Video``, ``AutoencoderKLLTX2Audio``, the text connectors, vocoder and latent
upsamplers); this package owns what diffusers does not ship in a form this app can use: how the
released single files are recognised and mapped onto those classes, and the denoising recipe.
"""
