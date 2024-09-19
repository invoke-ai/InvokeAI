---
title: Home
---

<!--
  The Docs you find here (/docs/*) are built and deployed via mkdocs. If you want to run a local version to verify your changes, it's as simple as::

  ```bash
  pip install -r docs/requirements-mkdocs.txt
  mkdocs serve
  ```
-->

<!-- CSS styling -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.1/css/fontawesome.min.css">
<style>
    .button {
      width: 100%;
      max-width: 100%;
      height: 50px;
      background-color: #35A4DB;
      color: #fff;
      font-size: 16px;
      border: none;
      cursor: pointer;
      border-radius: 0.2rem;
    }

    .button-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      justify-content: center;
    }

    .button:hover {
      background-color: #526CFE;
    }
</style>



<div align="center" markdown>


[![project logo](https://github.com/invoke-ai/InvokeAI/assets/31807370/6e3728c7-e90e-4711-905c-3b55844ff5be)](https://github.com/invoke-ai/InvokeAI)

[![discord badge]][discord link]

[![latest release badge]][latest release link]
[![github stars badge]][github stars link]
[![github forks badge]][github forks link]

<!-- [![CI checks on main badge]][ci checks on main link]
[![CI checks on dev badge]][ci checks on dev link]
[![latest commit to dev badge]][latest commit to dev link] -->

[![github open issues badge]][github open issues link]
[![github open prs badge]][github open prs link]

[ci checks on dev badge]:
  https://flat.badgen.net/github/checks/invoke-ai/InvokeAI/development?label=CI%20status%20on%20dev&cache=900&icon=github
[ci checks on dev link]:
  https://github.com/invoke-ai/InvokeAI/actions?query=branch%3Adevelopment
[ci checks on main badge]:
  https://flat.badgen.net/github/checks/invoke-ai/InvokeAI/main?label=CI%20status%20on%20main&cache=900&icon=github
[ci checks on main link]:
  https://github.com/invoke-ai/InvokeAI/actions/workflows/test-invoke-conda.yml
[discord badge]: https://flat.badgen.net/discord/members/ZmtBAhwWhy?icon=discord
[discord link]: https://discord.gg/ZmtBAhwWhy
[github forks badge]:
  https://flat.badgen.net/github/forks/invoke-ai/InvokeAI?icon=github
[github forks link]:
  https://useful-forks.github.io/?repo=lstein%2Fstable-diffusion
[github open issues badge]:
  https://flat.badgen.net/github/open-issues/invoke-ai/InvokeAI?icon=github
[github open issues link]:
  https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen
[github open prs badge]:
  https://flat.badgen.net/github/open-prs/invoke-ai/InvokeAI?icon=github
[github open prs link]:
  https://github.com/invoke-ai/InvokeAI/pulls?q=is%3Apr+is%3Aopen
[github stars badge]:
  https://flat.badgen.net/github/stars/invoke-ai/InvokeAI?icon=github
[github stars link]: https://github.com/invoke-ai/InvokeAI/stargazers
<!-- [latest commit to dev badge]:
  https://flat.badgen.net/github/last-commit/invoke-ai/InvokeAI/development?icon=github&color=yellow&label=last%20dev%20commit&cache=900
[latest commit to dev link]:
  https://github.com/invoke-ai/InvokeAI/commits/main -->
[latest release badge]:
  https://flat.badgen.net/github/release/invoke-ai/InvokeAI/development?icon=github
[latest release link]: https://github.com/invoke-ai/InvokeAI/releases

</div>

<a href="https://github.com/invoke-ai/InvokeAI">InvokeAI</a> is an
implementation of Stable Diffusion, the open source text-to-image and
image-to-image generator. It provides a streamlined process with various new
features and options to aid the image generation process. It runs on Windows,
Mac and Linux machines, and runs on GPU cards with as little as 4 GB of RAM.

<div align="center"><img src="assets/invoke-web-server-1.png" width=640></div>

## :octicons-link-24: Quick Links

<div class="button-container">
    <a href="installation/INSTALLATION"> <button class="button">Installation</button> </a>
    <a href="features/"> <button class="button">Features</button> </a>
    <a href="help/gettingStartedWithAI/"> <button class="button">Getting Started</button> </a>
    <a href="help/FAQ/"> <button class="button">FAQ</button> </a>
    <a href="help/VIDEOS/"> <button class="button">Invoke YouTube</button> </a>
    <a href="contributing/CONTRIBUTING/"> <button class="button">Contributing</button> </a>
    <a href="https://github.com/invoke-ai/InvokeAI/"> <button class="button">Code and Downloads</button> </a>
    <a href="https://github.com/invoke-ai/InvokeAI/issues"> <button class="button">Bug Reports </button> </a>
    <a href="https://discord.gg/ZmtBAhwWhy"> <button class="button"> Join the Discord Server!</button> </a>
</div>


## :octicons-gift-24: InvokeAI Features

### Installation
  - [Automated Installer](installation/010_INSTALL_AUTOMATED.md)
  - [Manual Installation](installation/020_INSTALL_MANUAL.md)
  - [Docker Installation](installation/040_INSTALL_DOCKER.md)

### The InvokeAI Web Interface
- [WebUI overview](features/WEB.md)
- [WebUI hotkey reference guide](features/WEBUIHOTKEYS.md)
- [WebUI Unified Canvas for Img2Img, inpainting and outpainting](features/UNIFIED_CANVAS.md)

<!-- separator -->

### Image Management
- [Image2Image](features/IMG2IMG.md)
- [Adding custom styles and subjects](features/CONCEPTS.md)
- [Upscaling and Face Reconstruction](features/POSTPROCESS.md)
- [Other Features](features/OTHER.md)

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

[Please take a look at our Contribution documentation to learn more about contributing to InvokeAI. 
](contributing/CONTRIBUTING.md)

## :octicons-person-24: Contributors

This software is a combined effort of various people from across the world.
[Check out the list of all these amazing people](other/CONTRIBUTORS.md). We
thank them for their time, hard work and effort.

## :octicons-question-24: Support

For support, please use this repository's GitHub Issues tracking service. Feel
free to send me an email if you use and like the script.

Original portions of the software are Copyright (c) 2022-23
by [The InvokeAI Team](https://github.com/invoke-ai).

