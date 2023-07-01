<div align="center">

![project hero](https://github.com/invoke-ai/InvokeAI/assets/31807370/1a917d94-e099-4fa1-a70f-7dd8d0691018)

# Invoke AI - Generative AI for Professional Creatives
## Image Generation for Stable Diffusion, Custom-Trained Models, and more. 
  Learn more about us and get started instantly at [invoke.ai](https://invoke.ai)


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

_**Note: This is an alpha release. Bugs are expected and not all
features are fully implemented. Please use the GitHub [Issues
pages](https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen)
to report unexpected problems. Also note that InvokeAI root directory
which contains models, outputs and configuration files, has changed
between the 2.x and 3.x release. If you wish to use your v2.3 root
directory with v3.0, please follow the directions in [Migrating a 2.3
root directory to 3.0](#migrating-to-3).**_

InvokeAI is a leading creative engine built to empower professionals
and enthusiasts alike. Generate and create stunning visual media using
the latest AI-driven technologies. InvokeAI offers an industry leading
Web Interface, interactive Command Line Interface, and also serves as
the foundation for multiple commercial products.

**Quick links**: [[How to
  Install](https://invoke-ai.github.io/InvokeAI/#installation)] [<a
  href="https://discord.gg/ZmtBAhwWhy">Discord Server</a>] [<a
  href="https://invoke-ai.github.io/InvokeAI/">Documentation and
  Tutorials</a>] [<a
  href="https://github.com/invoke-ai/InvokeAI/">Code and
  Downloads</a>] [<a
  href="https://github.com/invoke-ai/InvokeAI/issues">Bug Reports</a>]
  [<a
  href="https://github.com/invoke-ai/InvokeAI/discussions">Discussion,
  Ideas & Q&A</a>]

<div align="center">

![canvas preview](https://github.com/invoke-ai/InvokeAI/raw/main/docs/assets/canvas_preview.png)

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
[InvokeAI Installation Overview](https://invoke-ai.github.io/InvokeAI/installation/)

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

9. Press 2 to open the "browser-based UI", press enter/return, wait a
minute or two for Stable Diffusion to start up, then open your browser
and go to http://localhost:9090.

10. Type `banana sushi` in the box on the top left and click `Invoke`

### Command-Line Installation (for developers and users familiar with Terminals)

You must have Python 3.9 or 3.10 installed on your machine. Earlier or later versions are
not supported.

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
    pip install "InvokeAI[xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

    _For Linux with an AMD GPU:_

    ```sh
    pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/rocm5.4.2
    ```

    _For non-GPU systems:_
    ```terminal
    pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/cpu
    ``` 

    _For Macintoshes, either Intel or M1/M2:_

    ```sh
    pip install InvokeAI --use-pep517
    ```

6. Configure InvokeAI and install a starting set of image generation models (you only need to do this once):

    ```terminal
    invokeai-configure
    ```

7. Launch the web server (do it every time you run InvokeAI):

    ```terminal
    invokeai --web
    ```

8. Point your browser to http://localhost:9090 to bring up the web interface.
9. Type `banana sushi` in the box on the top left and click `Invoke`.

Be sure to activate the virtual environment each time before re-launching InvokeAI,
using `source .venv/bin/activate` or `.venv\Scripts\activate`.

## Detailed Installation Instructions

This fork is supported across Linux, Windows and Macintosh. Linux
users can use either an Nvidia-based card (with CUDA support) or an
AMD card (using the ROCm driver). For full installation and upgrade
instructions, please see:
[InvokeAI Installation Overview](https://invoke-ai.github.io/InvokeAI/installation/INSTALL_SOURCE/)

<a name="migrating-to-3"></a>
### Migrating a v2.3 InvokeAI root directory

The InvokeAI root directory is where the InvokeAI startup file,
installed models, and generated images are stored. It is ordinarily
named `invokeai` and located in your home directory. The contents and
layout of this directory has changed between versions 2.3 and 3.0 and
cannot be used directly.

We currently recommend that you use the installer to create a new root
directory named differently from the 2.3 one, e.g. `invokeai-3` and
then use a migration script to copy your 2.3 models into the new
location. However, if you choose, you can upgrade this directory in
place.  This section gives both recipes.

#### Creating a new root directory and migrating old models

This is the safer recipe because it leaves your old root directory in
place to fall back on.

1. Follow the instructions above to create and install InvokeAI in a
directory that has a different name from the 2.3 invokeai directory.
In this example, we will use "invokeai-3"

2. When you are prompted to select models to install, select a minimal
set of models, such as stable-diffusion-v1.5 only.

3. After installation is complete launch `invokeai.sh` (Linux/Mac) or
`invokeai.bat` and select option 8 "Open the developers console". This
will take you to the command line.

4. Issue the command `invokeai-migrate3 --from /path/to/v2.3-root --to
/path/to/invokeai-3-root`. Provide the correct `--from` and `--to`
paths for your v2.3 and v3.0 root directories respectively.

This will copy and convert your old models from 2.3 format to 3.0
format and create a new `models` directory in the 3.0 directory. The
old models directory (which contains the models selected at install
time) will be renamed `models.orig` and can be deleted once you have
confirmed that the migration was successful.

#### Migrating in place

For the adventurous, you may do an in-place upgrade from 2.3 to 3.0
without touching the command line. The recipe is as follows>

1. Launch the InvokeAI launcher script in your current v2.3 root directory.

2. Select option [9] "Update InvokeAI" to bring up the updater dialog.

3a. During the alpha release phase, select option [3] and manually
enter the tag name `v3.0.0+a2`.

3b. Once 3.0 is released, select option [1] to upgrade to the latest release.

4. Once the upgrade is finished you will be returned to the launcher
menu. Select option [7] "Re-run the configure script to fix a broken
install or to complete a major upgrade".

This will run the configure script against the v2.3 directory and
update it to the 3.0 format. The following files will be replaced:

  - The invokeai.init file, replaced by invokeai.yaml
  - The models directory
  - The configs/models.yaml model index
  
The original versions of these files will be saved with the suffix
".orig" appended to the end. Once you have confirmed that the upgrade
worked, you can safely remove these files. Alternatively you can
restore a working v2.3 directory by removing the new files and
restoring the ".orig" files' original names.

#### Migration Caveats

The migration script will migrate your invokeai settings and models,
including textual inversion models, LoRAs and merges that you may have
installed previously. However it does **not** migrate the generated
images stored in your 2.3-format outputs directory. The released
version of 3.0 is expected to have an interface for importing an
entire directory of image files as a batch.

## Hardware Requirements

InvokeAI is supported across Linux, Windows and macOS. Linux
users can use either an Nvidia-based card (with CUDA support) or an
AMD card (using the ROCm driver).

### System

You will need one of the following:

- An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- An Apple computer with an M1 chip.
- An AMD-based graphics card with 4GB or more VRAM memory. (Linux only)

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

### *Advanced Prompt Syntax*

Invoke AI's advanced prompt syntax allows for token weighting, cross-attention control, and prompt blending, allowing for fine-tuned tweaking of your invocations and exploration of the latent space. 

### *Command Line Interface*

For users utilizing a terminal-based environment, or who want to take advantage of CLI features, InvokeAI offers an extensive and actively supported command-line interface that provides the full suite of generation functionality available in the tool.

### Other features

- *Support for both ckpt and diffusers models*
- *SD 2.0, 2.1 support*
- *Upscaling & Face Restoration Tools*
- *Embedding Manager & Support*
- *Model Manager & Support*
- *Node-Based Architecture*
- *Node-Based Plug-&-Play UI (Beta)*
- *Boards & Gallery Management

### Latest Changes

For our latest changes, view our [Release
Notes](https://github.com/invoke-ai/InvokeAI/releases) and the
[CHANGELOG](docs/CHANGELOG.md).

### Troubleshooting

Please check out our **[Q&A](https://invoke-ai.github.io/InvokeAI/help/TROUBLESHOOT/#faq)** to get solutions for common installation
problems and other issues.

## 🤝 Contributing

Anyone who wishes to contribute to this project, whether documentation, features, bug fixes, code
cleanup, testing, or code reviews, is very much encouraged to do so.

To join, just raise your hand on the InvokeAI Discord server (#dev-chat) or the GitHub discussion board.

If you'd like to help with translation, please see our [translation guide](docs/other/TRANSLATION.md).

If you are unfamiliar with how
to contribute to GitHub projects, here is a
[Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github). A full set of contribution guidelines, along with templates, are in progress. You can **make your pull request against the "main" branch**.

We hope you enjoy using our software as much as we enjoy creating it,
and we hope that some of those of you who are reading this will elect
to become part of our community.

Welcome to InvokeAI!

### 👥 Contributors

This fork is a combined effort of various people from across the world.
[Check out the list of all these amazing people](https://invoke-ai.github.io/InvokeAI/other/CONTRIBUTORS/). We thank them for
their time, hard work and effort.

### Support

For support, please use this repository's GitHub Issues tracking service, or join the Discord.

Original portions of the software are Copyright (c) 2023 by respective contributors.

