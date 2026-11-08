# Provenance

Unmodified copies of third-party InvokeAI custom nodes, used to check that code written against
`context.images.get_pil` / `context.images.save` keeps working.

- Source: https://github.com/JPPhoto/cinematic-effects-pack
- Commit: 4a4dfa77ec8c57968281f8162a685d1d89a1aaea
- License: MIT (`LICENSE`, Copyright Jonathan S. Pollack)
- Files, copied byte for byte into `cinematic_effects/` (six node files registering eight invocation types,
  because `film_grain.py` also registers `monochrome_film_grain` and `chroma_noise`, plus one helper module):
  `chromatic_aberration.py`, `lens_vignette.py`, `lens_blur.py`, `curves.py`, `film_grain.py`,
  `spherical_distortion.py`, and `util.py` (imported by lens_vignette, lens_blur and curves).

The pack's `__init__.py` is not copied: it imports nodes this fixture does not carry.
Do not edit these files; bump the commit and re-copy instead.
