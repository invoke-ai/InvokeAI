<div align="center">

![project hero](https://github.com/invoke-ai/InvokeAI/assets/31807370/6e3728c7-e90e-4711-905c-3b55844ff5be)

# Invoke - Professional Creative AI Tools for Visual Media 
##  To learn more about Invoke, or implement our Business solutions, visit [invoke.com](https://www.invoke.com/about)
  


[![discord badge]][discord link]

[![latest release badge]][latest release link] [![github stars badge]][github stars link] [![github forks badge]][github forks link]

[![CI checks on main badge]][CI checks on main link] [![latest commit to main badge]][latest commit to main link]

[![github open issues badge]][github open issues link] [![github open prs badge]][github open prs link] [![translation status badge]][translation status link]

[CI checks on main badge]: https://flat.badgen.net/github/checks/invoke-ai/InvokeAI/main?label=CI%20status%20on%20main&cache=900&icon=github
[CI checks on main link]:https://github.com/invoke-ai/InvokeAI/actions?query=branch%3Amain
[discord badge]: https://flat.badgen.net/discord/members/ZmtBAhwWhy?icon=discord
[discord link]: https://discord.gg/ZmtBAhwWhy
[github forks badge]: https://flat.badgen.net/github/forks/invoke-ai/InvokeAI?icon=github
[github forks link]: https://useful-forks.github.io/?repo=invoke-ai%2FInvokeAI
[github open issues badge]: https://flat.badgen.net/github/open-issues/invoke-ai/InvokeAI?icon=github
[github open issues link]: https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen
[github open prs badge]: https://flat.badgen.net/github/open-prs/invoke-ai/InvokeAI?icon=github
[github open prs link]: https://github.com/invoke-ai/InvokeAI/pulls?q=is%3Apr+is%3Aopen
[github stars badge]: https://flat.badgen.net/github/stars/invoke-ai/InvokeAI?icon=github
[github stars link]: https://github.com/invoke-ai/InvokeAI/stargazers
[latest commit to main badge]: https://flat.badgen.net/github/last-commit/invoke-ai/InvokeAI/main?icon=github&color=yellow&label=last%20dev%20commit&cache=900
[latest commit to main link]: https://github.com/invoke-ai/InvokeAI/commits/main
[latest release badge]: https://flat.badgen.net/github/release/invoke-ai/InvokeAI/development?icon=github
[latest release link]: https://github.com/invoke-ai/InvokeAI/releases
[translation status badge]: https://hosted.weblate.org/widgets/invokeai/-/svg-badge.svg
[translation status link]: https://hosted.weblate.org/engage/invokeai/

</div>

InvokeAI is a leading creative engine built to empower professionals
and enthusiasts alike. Generate and create stunning visual media using
the latest AI-driven technologies. InvokeAI offers an industry leading
Web Interface, interactive Command Line Interface, and also serves as
the foundation for multiple commercial products.

**Quick links**: [[How to
  Install](https://invoke-ai.github.io/InvokeAI/installation/INSTALLATION/)] [<a
  href="https://discord.gg/ZmtBAhwWhy">Discord Server</a>] [<a
  href="https://invoke-ai.github.io/InvokeAI/">Documentation and
  Tutorials</a>]
  [<a href="https://github.com/invoke-ai/InvokeAI/issues">Bug Reports</a>]
  [<a
  href="https://github.com/invoke-ai/InvokeAI/discussions">Discussion,
  Ideas & Q&A</a>] 
   [<a
  href="https://invoke-ai.github.io/InvokeAI/contributing/CONTRIBUTING/">Contributing</a>] 

<div align="center">


![Highlighted Features - Canvas and Workflows](https://github.com/invoke-ai/InvokeAI/assets/31807370/708f7a82-084f-4860-bfbe-e2588c53548d)


</div>

## Table of Contents

Table of Contents 📝

**Getting Started**
1. 🏁 [Quick Start](#quick-start) 
3. 🖥️ [Hardware Requirements](#hardware-requirements) 

**More About Invoke**
1. 🌟 [Features](#features) 
2. 📣 [Latest Changes](#latest-changes) 
3. 🛠️ [Troubleshooting](#troubleshooting) 

**Supporting the Project**
1. 🤝 [Contributing](#contributing) 
2. 👥 [Contributors](#contributors) 
3. 💕 [Support](#support) 

## Quick Start

For full installation and upgrade instructions, please see:
[InvokeAI Installation Overview](https://invoke-ai.github.io/InvokeAI/installation/INSTALLATION/)

If upgrading from version 2.3, please read [Migrating a 2.3 root
directory to 3.0](#migrating-to-3) first.

### Automatic Installer (suggested for 1st time users)

1. Go to the bottom of the [Latest Release Page](https://github.com/invoke-ai/InvokeAI/releases/latest)

2. Download the .zip file for your OS (Windows/macOS/Linux).

3. Unzip the file.

4. **Windows:** double-click on the `install.bat` script. **macOS:** Open a Terminal window, drag the file `install.sh` from Finder
into the Terminal, and press return. **Linux:** run `install.sh`.

5. You'll be asked to confirm the location of the folder in which
to install InvokeAI and its image generation model files. Pick a
location with at least 15 GB of free memory. More if you plan on
installing lots of models.

6. Wait while the installer does its thing. After installing the software,
the installer will launch a script that lets you configure InvokeAI and
select a set of starting image generation models.

7. Find the folder that InvokeAI was installed into (it is not the
same as the unpacked zip file directory!) The default location of this
folder (if you didn't change it in step 5) is `~/invokeai` on
Linux/Mac systems, and `C:\Users\YourName\invokeai` on Windows. This directory will contain launcher scripts named `invoke.sh` and `invoke.bat`.

8. On Windows systems, double-click on the `invoke.bat` file. On
macOS, open a Terminal window, drag `invoke.sh` from the folder into
the Terminal, and press return. On Linux, run `invoke.sh`

9. Press 1 to open the "browser-based UI", press enter/return, wait a
minute or two for Stable Diffusion to start up, then open your browser
and go to http://localhost:9090.

10. Type `banana sushi` in the box on the top left and click `Invoke`

### Command-Line Installation (for developers and users familiar with Terminals)

You must have Python 3.10 through 3.11 installed on your machine. Earlier or
later versions are not supported.
Node.js also needs to be installed along with `pnpm` (can be installed with
the command `npm install -g pnpm` if needed)

1. Open a command-line window on your machine. The PowerShell is recommended for Windows.
2. Create a directory to install InvokeAI into. You'll need at least 15 GB of free space:

    ```terminal
    mkdir invokeai
    ````

3. Create a virtual environment named `.venv` inside this directory and activate it:

    ```terminal
    cd invokeai
    python -m venv .venv --prompt InvokeAI
    ```

4. Activate the virtual environment (do it every time you run InvokeAI)

    _For Linux/Mac users:_

    ```sh
    source .venv/bin/activate
    ```

    _For Windows users:_

    ```ps
    .venv\Scripts\activate
    ```

5. Install the InvokeAI module and its dependencies. Choose the command suited for your platform & GPU.

    _For Windows/Linux with an NVIDIA GPU:_

    ```terminal
    pip install "InvokeAI[xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu121
    ```

    _For Linux with an AMD GPU:_

    ```sh
    pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/rocm5.6
    ```

    _For non-GPU systems:_
    ```terminal
    pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/cpu
    ``` 

    _For Macintoshes, either Intel or M1/M2/M3:_

    ```sh
    pip install InvokeAI --use-pep517
    ```

6. Launch the web server (do it every time you run InvokeAI):

    ```terminal
    invokeai-web
    ```

7. Point your browser to http://localhost:9090 to bring up the web interface.

8. Type `banana sushi` in the box on the top left and click `Invoke`.

Be sure to activate the virtual environment each time before re-launching InvokeAI,
using `source .venv/bin/activate` or `.venv\Scripts\activate`.

## Detailed Installation Instructions

This fork is supported across Linux, Windows and Macintosh. Linux
users can use either an Nvidia-based card (with CUDA support) or an
AMD card (using the ROCm driver). For full installation and upgrade
instructions, please see:
[InvokeAI Installation Overview](https://invoke-ai.github.io/InvokeAI/installation/INSTALL_SOURCE/)

## Hardware Requirements

InvokeAI is supported across Linux, Windows and macOS. Linux
users can use either an Nvidia-based card (with CUDA support) or an
AMD card (using the ROCm driver).

### System

You will need one of the following:

- An NVIDIA-based graphics card with 4 GB or more VRAM memory. 6-8 GB
  of VRAM is highly recommended for rendering using the Stable
  Diffusion XL models
- An Apple computer with an M1 chip.
- An AMD-based graphics card with 4GB or more VRAM memory (Linux
  only), 6-8 GB for XL rendering.

We do not recommend the GTX 1650 or 1660 series video cards. They are
unable to run in half-precision mode and do not have sufficient VRAM
to render 512x512 images.

**Memory** - At least 12 GB Main Memory RAM.

**Disk** - At least 12 GB of free disk space for the machine learning model, Python, and all its dependencies.

## Features

Feature documentation can be reviewed by navigating to [the InvokeAI Documentation page](https://invoke-ai.github.io/InvokeAI/features/)

### *Web Server & UI*

InvokeAI offers a locally hosted Web Server & React Frontend, with an industry leading user experience. The Web-based UI allows for simple and intuitive workflows, and is responsive for use on mobile devices and tablets accessing the web server.

### *Unified Canvas*

The Unified Canvas is a fully integrated canvas implementation with support for all core generation capabilities, in/outpainting, brush tools, and more. This creative tool unlocks the capability for artists to create with AI as a creative collaborator, and can be used to augment AI-generated imagery, sketches, photography, renders, and more.

### *Workflows & Nodes*

InvokeAI offers a fully featured workflow management solution, enabling users to combine the power of nodes based workflows with the easy of a UI. This allows for customizable generation pipelines to be developed and shared by users looking to create specific workflows to support their production use-cases.

### *Board & Gallery Management*

Invoke AI provides an organized gallery system for easily storing, accessing, and remixing your content in the Invoke workspace. Images can be dragged/dropped onto any Image-base UI element in the application, and rich metadata within the Image allows for easy recall of key prompts or settings used in your workflow. 

### Other features

- *Support for both ckpt and diffusers models*
- *SD1.5, SD2.0, and SDXL support*
- *Upscaling Tools*
- *Embedding Manager & Support*
- *Model Manager & Support*
- *Workflow creation & management*
- *Node-Based Architecture*


### Latest Changes

For our latest changes, view our [Release
Notes](https://github.com/invoke-ai/InvokeAI/releases) and the
[CHANGELOG](docs/CHANGELOG.md).

### Troubleshooting / FAQ

Please check out our **[FAQ](https://invoke-ai.github.io/InvokeAI/help/FAQ/)** to get solutions for common installation
problems and other issues. For more help, please join our [Discord][discord link]

## Contributing

Anyone who wishes to contribute to this project, whether documentation, features, bug fixes, code
cleanup, testing, or code reviews, is very much encouraged to do so.

Get started with contributing by reading our [Contribution documentation](https://invoke-ai.github.io/InvokeAI/contributing/CONTRIBUTING/), joining the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) or the GitHub discussion board.

If you are unfamiliar with how
to contribute to GitHub projects, we have a new contributor checklist you can follow to get started contributing: 
[New Contributor Checklist](https://invoke-ai.github.io/InvokeAI/contributing/contribution_guides/newContributorChecklist/).

We hope you enjoy using our software as much as we enjoy creating it,
and we hope that some of those of you who are reading this will elect
to become part of our community.

Welcome to InvokeAI!

### Contributors

This fork is a combined effort of various people from across the world.
[Check out the list of all these amazing people](https://invoke-ai.github.io/InvokeAI/other/CONTRIBUTORS/). We thank them for
their time, hard work and effort.

### Support

For support, please use this repository's GitHub Issues tracking service, or join the [Discord][discord link].

Original portions of the software are Copyright (c) 2024 by respective contributors.

