# MOSS-Image vendored inference code

`pipeline_moss_image.py` in this package is adapted from the MOSS-Image reference
inference code published with the model at
https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base (file `pipeline_moss_image.py`,
revision `1ceede80badddd6a7812069619edaa5633587192`), which is licensed under the Apache
License, Version 2.0.

Copyright © the OpenMOSS Team. Licensed under the Apache License, Version 2.0; a copy is
available at http://www.apache.org/licenses/LICENSE-2.0. The repository's `LICENSE` file is
the unmodified Apache-2.0 text without a copyright line; the model card credits the OpenMOSS
Team as the author.

Modifications by the InvokeAI project: the `MOSSImagePipeline` class and the imports only it
used were removed, a provenance comment was added at the top of the file, and the transformer's
timestep embedding is computed on the CPU on devices without float64 (MPS); the full list is in
`__init__.py`.

`layout.py` ports the box conversion and serialization of the layout compiler in OpenMOSS'
prompt-enhancer skill, https://github.com/OpenMOSS/MOSS-Image (file
`skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`, commit
`c1355e6b1986dfe8fba3c0455783a05899212342`), Apache-2.0 under the same terms; the repository's
`LICENSE` is likewise the unmodified Apache-2.0 text. InvokeAI builds the layout items from canvas
regions instead of compiling a whole draft; the differences are listed in the module docstring.
`tests/backend/moss_image/test_layout.py` ports expectations from that skill's
`test_compile_prompt.py`.

The remaining modules in this package are original InvokeAI code that drives the vendored model
in InvokeAI invocations.

The MOSS-Image model **weights** are also released under the Apache License, Version 2.0, as
stated on the model card.
