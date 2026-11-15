# FLUX.3 reference code

Parts of this package port the FLUX.3 Image Klein reference implementation published by Black
Forest Labs at https://github.com/black-forest-labs/flux-3-image-klein (commit
`45e57c9fb06f9751cdced29a65c5d2171003be05`), which is licensed under the Apache License, Version
2.0. Each module that ports it names the reference file it follows in its docstring.

Copyright 2026 Black Forest Labs, Inc. Licensed under the Apache License, Version 2.0; a copy is
available at http://www.apache.org/licenses/LICENSE-2.0.

`layout.py` ports `split`, `valid_box`, the table check of `detect`, and the constants `ID` and
`SCALE` from `src/flux3/bbox.py`, and writes the rows of an edit the way `scripts/app.py` does. Both
files are unchanged at the repository head `bfc8889a4497f846416f253a3972a1941ac698b9` (2026-10-17)
apart from a default value in `app.py` that `layout.py` does not use. Where it differs from the
reference is listed in its module docstring.

`prompt_upsampler.py` drives the prompt upsampler as `Upsampler.upsample` of `src/flux3/pup.py` does, through
diffusers' port of it (`modular_pipelines/flux3/prompt_upsampler.py`, Apache-2.0), whose system prompts, scaffolds and
box-table helpers it uses. The seeded "FLUX.3 Prompt Upsampler" system prompt
(`invokeai/app/services/shared/sqlite_migrator/migrations/migration_2026_11_09_add_flux3_prompt_upsampler_system_prompt.py`)
is the `t2i` entry of `system_prompts.json` of https://huggingface.co/black-forest-labs/flux-3-image-klein-pup at commit
`3f3edb52b2c55de0d49feab28598b68f7f8e4830`, verbatim, under the same license.

The FLUX.3 model **weights** are not covered by this license; they are released under the FLUX
Kommunity License v1.0.
