---
title: Invoke
---

<!--
  The docs are generated with using mkdocs: https://www.mkdocs.org/

  To preview the docs locally, first install the requirements:
  ```sh
  pip install -e ".[docs]"
  ```

  Then run the mkdocs server with `mkdocs serve`, or, on Unix systems, `make docs`.
-->

<!-- CSS styling -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.1/css/fontawesome.min.css">

<div align="center" markdown>

[![project logo](https://github.com/invoke-ai/InvokeAI/assets/31807370/6e3728c7-e90e-4711-905c-3b55844ff5be)](https://github.com/invoke-ai/InvokeAI)

[![discord badge]][discord link]
[![latest release badge]][latest release link]
[![github stars badge]][github stars link]
[![github forks badge]][github forks link]
[![latest commit to main badge]][latest commit to main link]
[![github open issues badge]][github open issues link]
[![github open prs badge]][github open prs link]

[discord badge]: https://flat.badgen.net/discord/members/ZmtBAhwWhy?icon=discord
[discord link]: https://discord.gg/ZmtBAhwWhy
[github forks badge]: https://flat.badgen.net/github/forks/invoke-ai/InvokeAI?icon=github
[github forks link]: https://useful-forks.github.io/?repo=lstein%2Fstable-diffusion
[github open issues badge]: https://flat.badgen.net/github/open-issues/invoke-ai/InvokeAI?icon=github
[github open issues link]: https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen
[github open prs badge]: https://flat.badgen.net/github/open-prs/invoke-ai/InvokeAI?icon=github
[github open prs link]: https://github.com/invoke-ai/InvokeAI/pulls?q=is%3Apr+is%3Aopen
[github stars badge]: https://flat.badgen.net/github/stars/invoke-ai/InvokeAI?icon=github
[github stars link]: https://github.com/invoke-ai/InvokeAI/stargazers
[latest commit to main badge]: https://flat.badgen.net/github/last-commit/invoke-ai/InvokeAI/main?icon=github&color=yellow&label=last%20commit&cache=900
[latest commit to main link]: https://github.com/invoke-ai/InvokeAI/commits/main
[latest release badge]: https://flat.badgen.net/github/release/invoke-ai/InvokeAI/development?icon=github
[latest release link]: https://github.com/invoke-ai/InvokeAI/releases

</div>

<a href="https://github.com/invoke-ai/InvokeAI">InvokeAI</a> is an
implementation of Stable Diffusion, the open source text-to-image and
image-to-image generator. It provides a streamlined process with various new
features and options to aid the image generation process. It runs on Windows,
Mac and Linux machines, and runs on GPU cards with as little as 4 GB of RAM.

<div align="center"><img src="assets/invoke-web-server-1.png" width=640></div>

## Installation

The simple [installer script](installation/installer.md) is the suggested way to install and update the application.

You may also install Invoke as python package [via PyPI](installation/manual.md) or [docker](installation/docker.md).

<!-- separator -->

### Model Management

- [Installing](installation/050_INSTALLING_MODELS.md)
- [Model Merging](features/MODEL_MERGING.md)
- [ControlNet Models](features/CONTROLNET.md)
- [Style/Subject Concepts and Embeddings](features/CONCEPTS.md)
- [Watermarking and the Not Safe for Work (NSFW) Checker](features/WATERMARK+NSFW.md)
<!-- seperator -->

### Prompt Engineering

- [Prompt Syntax](features/PROMPTS.md)

### InvokeAI Configuration

- [Guide to InvokeAI Runtime Settings](features/CONFIGURATION.md)
- [Database Maintenance and other Command Line Utilities](features/UTILITIES.md)

## :material-target: Troubleshooting

Please check out our **[:material-frequently-asked-questions:
FAQ](help/FAQ/)** to
get solutions for common installation problems and other issues.

## :octicons-repo-push-24: Contributing

Anyone who wishes to contribute to this project, whether documentation,
features, bug fixes, code cleanup, testing, or code reviews, is very much
encouraged to do so.

[Please take a look at our Contribution documentation to learn more about contributing to InvokeAI.](contributing/CONTRIBUTING.md)

## :octicons-person-24: Contributors

This software is a combined effort of various people from across the world.
[Check out the list of all these amazing people](other/CONTRIBUTORS.md). We
thank them for their time, hard work and effort.

## :octicons-question-24: Support

For support, please use this repository's GitHub Issues tracking service. Feel
free to send me an email if you use and like the script.

Original portions of the software are Copyright (c) 2022-23
by [The InvokeAI Team](https://github.com/invoke-ai).
