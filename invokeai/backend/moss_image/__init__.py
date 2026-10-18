"""MOSS-Image backend.

``pipeline_moss_image.py`` is vendored from the model repository
https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base at revision
1ceede80badddd6a7812069619edaa5633587192 (file blob 60ceb4b6962a6b3d3c57e404b4d7d26fa4b49b07),
Apache-2.0 (see ``NOTICE.md``). The local changes, and the only ones:

- ``MOSSImagePipeline`` is removed, together with the imports only it used: ``math``,
  ``pathlib.Path``, ``AutoencoderKLFlux2``, ``DiffusionPipeline``, ``VaeImageProcessor``,
  ``ImagePipelineOutput``, ``randn_tensor``, ``transformers.AutoModel``/``AutoTokenizer``, and
  ``SchedulerMixin`` (imported upstream only for diffusers' remote-code component discovery). The
  ``from diffusers import ...`` line is trimmed to ``FlowMatchEulerDiscreteScheduler``.
- A two-line provenance comment is added above the module docstring.

The module docstring still describes the upstream pipeline API, which this copy does not contain.
Keep the file otherwise byte-identical to upstream: it is excluded from ``ruff format`` and keeps
upstream's import order (``pyproject.toml``), so syncing a new revision is a copy plus exactly the
changes above. InvokeAI drives the transformer with its own loop (``denoise.py``); ``sampling.py``
holds the schedule, the initial noise and the conversion between InvokeAI's FLUX.2 latent layout and
the model's.
"""
