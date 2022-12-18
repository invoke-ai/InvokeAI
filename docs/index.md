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

<div align="center" markdown>

[![project logo](assets/invoke_ai_banner.png)](https://github.com/invoke-ai/InvokeAI)

[![discord badge]][discord link]

[![latest release badge]][latest release link]
[![github stars badge]][github stars link]
[![github forks badge]][github forks link]

[![CI checks on main badge]][ci checks on main link]
[![CI checks on dev badge]][ci checks on dev link]
[![latest commit to dev badge]][latest commit to dev link]

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
[latest commit to dev badge]:
  https://flat.badgen.net/github/last-commit/invoke-ai/InvokeAI/development?icon=github&color=yellow&label=last%20dev%20commit&cache=900
[latest commit to dev link]:
  https://github.com/invoke-ai/InvokeAI/commits/development
[latest release badge]:
  https://flat.badgen.net/github/release/invoke-ai/InvokeAI/development?icon=github
[latest release link]: https://github.com/invoke-ai/InvokeAI/releases

</div>

<a href="https://github.com/invoke-ai/InvokeAI">InvokeAI</a> is an
implementation of Stable Diffusion, the open source text-to-image and
image-to-image generator. It provides a streamlined process with various new
features and options to aid the image generation process. It runs on Windows,
Mac and Linux machines, and runs on GPU cards with as little as 4 GB or RAM.

**Quick links**: [<a href="https://discord.gg/ZmtBAhwWhy">Discord Server</a>]
[<a href="https://github.com/invoke-ai/InvokeAI/">Code and Downloads</a>] [<a
href="https://github.com/invoke-ai/InvokeAI/issues">Bug Reports</a>] [<a
href="https://github.com/invoke-ai/InvokeAI/discussions">Discussion, Ideas &
Q&A</a>]

<div align="center"><img src="assets/invoke-web-server-1.png" width=640></div>

!!! note

    This fork is rapidly evolving. Please use the [Issues tab](https://github.com/invoke-ai/InvokeAI/issues) to report bugs and make feature requests. Be sure to use the provided templates. They will help aid diagnose issues faster.

## :octicons-package-dependencies-24: Installation

This fork is supported across Linux, Windows and Macintosh. Linux users can use
either an Nvidia-based card (with CUDA support) or an AMD card (using the ROCm
driver).

First time users, please see
[Automated Installer](installation/INSTALL_AUTOMATED.md) for a walkthrough of
getting InvokeAI up and running on your system. For alternative installation and
upgrade instructions, please see:
[InvokeAI Installation Overview](installation/)

Linux users who wish to make use of the PyPatchMatch inpainting functions will
need to perform a bit of extra work to enable this module. Instructions can be
found at [Installing PyPatchMatch](installation/060_INSTALL_PATCHMATCH.md).

## :fontawesome-solid-computer: Hardware Requirements

### :octicons-cpu-24: System

You wil need one of the following:

- :simple-nvidia: An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- :simple-amd: An AMD-based graphics card with 4 GB or more VRAM memory (Linux
  only)
- :fontawesome-brands-apple: An Apple computer with an M1 chip.

We do **not recommend** the following video cards due to issues with their
running in half-precision mode and having insufficient VRAM to render 512x512
images in full-precision mode:

- NVIDIA 10xx series cards such as the 1080ti
- GTX 1650 series cards
- GTX 1660 series cards

### :fontawesome-solid-memory: Memory

- At least 12 GB Main Memory RAM.

### :fontawesome-regular-hard-drive: Disk

- At least 18 GB of free disk space for the machine learning model, Python, and
  all its dependencies.

!!! info

    Precision is auto configured based on the device. If however you encounter errors like
    `expected type Float but found Half` or `not implemented for Half` you can try starting
    `invoke.py` with the `--precision=float32` flag:

    ```bash
    (invokeai) ~/InvokeAI$ python scripts/invoke.py --full_precision
    ```

## :octicons-gift-24: InvokeAI Features

- [The InvokeAI Web Interface](features/WEB.md) -
[WebGUI hotkey reference guide](features/WEBUIHOTKEYS.md) -
[WebGUI Unified Canvas for Img2Img, inpainting and outpainting](features/UNIFIED_CANVAS.md)
<!-- seperator -->
- [The Command Line Interace](features/CLI.md) -
[Image2Image](features/IMG2IMG.md) - [Inpainting](features/INPAINTING.md) -
[Outpainting](features/OUTPAINTING.md) -
[Adding custom styles and subjects](features/CONCEPTS.md) -
[Upscaling and Face Reconstruction](features/POSTPROCESS.md)
<!-- seperator -->
- [Generating Variations](features/VARIATIONS.md)
<!-- seperator -->
- [Prompt Engineering](features/PROMPTS.md)
<!-- seperator -->
- Miscellaneous
  - [NSFW Checker](features/NSFW.md)
  - [Embiggen upscaling](features/EMBIGGEN.md)
  - [Other](features/OTHER.md)

## :octicons-log-16: Latest Changes

### v2.2.4 <small>(11 December 2022)</small>

#### the `invokeai` directory

Previously there were two directories to worry about, the directory that
contained the InvokeAI source code and the launcher scripts, and the `invokeai`
directory that contained the models files, embeddings, configuration and
outputs. With the 2.2.4 release, this dual system is done away with, and
everything, including the `invoke.bat` and `invoke.sh` launcher scripts, now
live in a directory named `invokeai`. By default this directory is located in
your home directory (e.g. `\Users\yourname` on Windows), but you can select
where it goes at install time.

After installation, you can delete the install directory (the one that the zip
file creates when it unpacks). Do **not** delete or move the `invokeai`
directory!

##### Initialization file `invokeai/invokeai.init`

You can place frequently-used startup options in this file, such as the default
number of steps or your preferred sampler. To keep everything in one place, this
file has now been moved into the `invokeai` directory and is named
`invokeai.init`.

#### To update from Version 2.2.3

The easiest route is to download and unpack one of the 2.2.4 installer files.
When it asks you for the location of the `invokeai` runtime directory, respond
with the path to the directory that contains your 2.2.3 `invokeai`. That is, if
`invokeai` lives at `C:\Users\fred\invokeai`, then answer with `C:\Users\fred`
and answer "Y" when asked if you want to reuse the directory.

The `update.sh` (`update.bat`) script that came with the 2.2.3 source installer
does not know about the new directory layout and won't be fully functional.

#### To update to 2.2.5 (and beyond) there's now an update path.

As they become available, you can update to more recent versions of InvokeAI
using an `update.sh` (`update.bat`) script located in the `invokeai` directory.
Running it without any arguments will install the most recent version of
InvokeAI. Alternatively, you can get set releases by running the `update.sh`
script with an argument in the command shell. This syntax accepts the path to
the desired release's zip file, which you can find by clicking on the green
"Code" button on this repository's home page.

#### Other 2.2.4 Improvements

- Fix InvokeAI GUI initialization by @addianto in #1687
- fix link in documentation by @lstein in #1728
- Fix broken link by @ShawnZhong in #1736
- Remove reference to binary installer by @lstein in #1731
- documentation fixes for 2.2.3 by @lstein in #1740
- Modify installer links to point closer to the source installer by @ebr in
  #1745
- add documentation warning about 1650/60 cards by @lstein in #1753
- Fix Linux source URL in installation docs by @andybearman in #1756
- Make install instructions discoverable in readme by @damian0815 in #1752
- typo fix by @ofirkris in #1755
- Non-interactive model download (support HUGGINGFACE_TOKEN) by @ebr in #1578
- fix(srcinstall): shell installer - cp scripts instead of linking by @tildebyte
  in #1765
- stability and usage improvements to binary & source installers by @lstein in
  #1760
- fix off-by-one bug in cross-attention-control by @damian0815 in #1774
- Eventually update APP_VERSION to 2.2.3 by @spezialspezial in #1768
- invoke script cds to its location before running by @lstein in #1805
- Make PaperCut and VoxelArt models load again by @lstein in #1730
- Fix --embedding_directory / --embedding_path not working by @blessedcoolant in
  #1817
- Clean up readme by @hipsterusername in #1820
- Optimized Docker build with support for external working directory by @ebr in
  #1544
- disable pushing the cloud container by @mauwii in #1831
- Fix docker push github action and expand with additional metadata by @ebr in
  #1837
- Fix Broken Link To Notebook by @VedantMadane in #1821
- Account for flat models by @spezialspezial in #1766
- Update invoke.bat.in isolate environment variables by @lynnewu in #1833
- Arch Linux Specific PatchMatch Instructions & fixing conda install on linux by
  @SammCheese in #1848
- Make force free GPU memory work in img2img by @addianto in #1844
- New installer by @lstein

For older changelogs, please visit the
**[CHANGELOG](CHANGELOG/#v223-2-december-2022)**.

## :material-target: Troubleshooting

Please check out our
**[:material-frequently-asked-questions: Q&A](help/TROUBLESHOOT.md)** to get
solutions for common installation problems and other issues.

## :octicons-repo-push-24: Contributing

Anyone who wishes to contribute to this project, whether documentation,
features, bug fixes, code cleanup, testing, or code reviews, is very much
encouraged to do so. If you are unfamiliar with how to contribute to GitHub
projects, here is a
[Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github).

A full set of contribution guidelines, along with templates, are in progress,
but for now the most important thing is to **make your pull request against the
"development" branch**, and not against "main". This will help keep public
breakage to a minimum and will allow you to propose more radical changes.

## :octicons-person-24: Contributors

This fork is a combined effort of various people from across the world.
[Check out the list of all these amazing people](other/CONTRIBUTORS.md). We
thank them for their time, hard work and effort.

## :octicons-question-24: Support

For support, please use this repository's GitHub Issues tracking service. Feel
free to send me an email if you use and like the script.

Original portions of the software are Copyright (c) 2020
[Lincoln D. Stein](https://github.com/lstein)

## :octicons-book-24: Further Reading

Please see the original README for more information on this software and
underlying algorithm, located in the file
[README-CompViz.md](other/README-CompViz.md).
