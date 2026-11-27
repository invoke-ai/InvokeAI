# Ideogram 4 vendored inference code

The following modules in this package are adapted from the Ideogram 4 reference
implementation at https://github.com/ideogram-oss/ideogram4 (the `ideogram4`
Python package), which is licensed under the Apache License, Version 2.0:

- `modeling_ideogram4.py`
- `autoencoder.py`
- `latent_norm.py`
- `scheduler.py`
- `sampler_configs.py`
- `constants.py`
- `quantized_loading.py`

Copyright © Ideogram, Inc. Licensed under the Apache License, Version 2.0; a copy
is available at http://www.apache.org/licenses/LICENSE-2.0.

Modifications by the InvokeAI project: intra-package import paths were rewritten
to `invokeai.backend.ideogram4.*`. The remaining modules in this package
(`conditioning.py`, `sampling_utils.py`, `denoise.py`, etc.) are original InvokeAI
code that wraps the vendored model for use in InvokeAI invocations.

The Ideogram 4 model **weights** are NOT covered by this Apache license; they are
distributed under the separate "Ideogram Non-Commercial Model Agreement".
