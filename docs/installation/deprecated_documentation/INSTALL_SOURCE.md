---
title: Source Installer
---

# The InvokeAI Source Installer

## Introduction

The source installer is a shell script that attempts to automate every step
needed to install and run InvokeAI on a stock computer running recent versions
of Linux, MacOS or Windows. It will leave you with a version that runs a stable
version of InvokeAI with the option to upgrade to experimental versions later.

Before you begin, make sure that you meet the
[hardware requirements](../../index.md#hardware-requirements) and has the appropriate
GPU drivers installed. In particular, if you are a Linux user with an AMD GPU
installed, you may need to install the
[ROCm driver](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

Installation requires roughly 18G of free disk space to load the libraries and
recommended model weights files.

## Walk through

Though there are multiple steps, there really is only one click involved to kick
off the process.

1.  The source installer is distributed in ZIP files. Go to the
    [latest release](https://github.com/invoke-ai/InvokeAI/releases/latest), and
    look for a series of files named:

    - [invokeAI-src-installer-2.2.3-mac.zip](https://github.com/invoke-ai/InvokeAI/releases/latest/download/invokeAI-src-installer-2.2.3-mac.zip)
    - [invokeAI-src-installer-2.2.3-windows.zip](https://github.com/invoke-ai/InvokeAI/releases/latest/download/invokeAI-src-installer-2.2.3-windows.zip)
    - [invokeAI-src-installer-2.2.3-linux.zip](https://github.com/invoke-ai/InvokeAI/releases/latest/download/invokeAI-src-installer-2.2.3-linux.zip)

    Download the one that is appropriate for your operating system.

2.  Unpack the zip file into a directory that has at least 18G of free space. Do
    _not_ unpack into a directory that has an earlier version of InvokeAI.

    This will create a new directory named "InvokeAI". This example shows how
    this would look using the `unzip` command-line tool, but you may use any
    graphical or command-line Zip extractor:

    ```cmd
    C:\Documents\Linco> unzip invokeAI-windows.zip
    Archive:  C: \Linco\Downloads\invokeAI-linux.zip
    creating: invokeAI\
    inflating: invokeAI\install.bat
    inflating: invokeAI\readme.txt
    ```

3. If you are a macOS user, you may need to install the Xcode command line tools.
   These are a set of tools that are needed to run certain applications in a Terminal,
   including InvokeAI. This package is provided directly by Apple.

   To install, open a terminal window and run `xcode-select --install`. You will get
   a macOS system popup guiding you through the install. If you already have them
   installed, you will instead see some output in the Terminal advising you that the
   tools are already installed.

   More information can be found here:
   https://www.freecodecamp.org/news/install-xcode-command-line-tools/

4.  If you are using a desktop GUI, double-click the installer file. It will be
    named `install.bat` on Windows systems and `install.sh` on Linux and
    Macintosh systems.

5.  Alternatively, from the command line, run the shell script or .bat file:

    ```cmd
    C:\Documents\Linco> cd invokeAI
    C:\Documents\Linco\invokeAI> install.bat
    ```

6.  Sit back and let the install script work. It will install various binary
    requirements including Conda, Git and Python, then download the current
    InvokeAI code and install it along with its dependencies.

    Be aware that some of the library download and install steps take a long time.
    In particular, the `pytorch` package is quite large and often appears to get
    "stuck" at 99.9%. Similarly, the `pip installing requirements` step may
    appear to hang. Have patience and the installation step will eventually
    resume. However, there are occasions when the library install does
    legitimately get stuck. If you have been waiting for more than ten minutes
    and nothing is happening, you can interrupt the script with ^C. You may restart
    it and it will pick up where it left off.

7.  After installation completes, the installer will launch a script called
    `configure_invokeai.py`, which will guide you through the first-time process of
    selecting one or more Stable Diffusion model weights files, downloading and
    configuring them.

    Note that the main Stable Diffusion weights file is protected by a license
    agreement that you must agree to in order to use. The script will list the
    steps you need to take to create an account on the official site that hosts
    the weights files, accept the agreement, and provide an access token that
    allows InvokeAI to legally download and install the weights files.

    If you have already downloaded the weights file(s) for another Stable
    Diffusion distribution, you may skip this step (by selecting "skip" when
    prompted) and configure InvokeAI to use the previously-downloaded files. The
    process for this is described in [Installing Models](../050_INSTALLING_MODELS.md).

8.  The script will now exit and you'll be ready to generate some images. The
    invokeAI directory will contain numerous files. Look for a shell script
    named `invoke.sh` (Linux/Mac) or `invoke.bat` (Windows). Launch the script
    by double-clicking it or typing its name at the command-line:

    ```cmd
    C:\Documents\Linco> cd invokeAI
    C:\Documents\Linco\invokeAI> invoke.bat
    ```

The `invoke.bat` (`invoke.sh`) script will give you the choice of starting (1)
the command-line interface, or (2) the web GUI. If you start the latter, you can
load the user interface by pointing your browser at http://localhost:9090.

The `invoke` script also offers you a third option labeled "open the developer
console". If you choose this option, you will be dropped into a command-line
interface in which you can run python commands directly, access developer tools,
and launch InvokeAI with customized options. To do the latter, you would launch
the script `scripts/invoke.py` as shown in this example:

```cmd
python scripts/invoke.py --web --max_load_models=3 \
    --model=waifu-1.3 --steps=30 --outdir=C:/Documents/AIPhotos
```

These options are described in detail in the
[Command-Line Interface](../../features/CLI.md) documentation.

## Troubleshooting

_Package dependency conflicts_ If you have previously installed
InvokeAI or another Stable Diffusion package, the installer may
occasionally pick up outdated libraries and either the installer or
`invoke` will fail with complaints out library conflicts. There are
two steps you can take to clear this problem. Both of these are done
from within the "developer's console", which you can get to by
launching `invoke.sh` (or `invoke.bat`) and selecting launch option
#3:

1. Remove the previous `invokeai` environment completely. From within
   the developer's console, give the command `conda env remove -n
   invokeai`. This will delete previous files installed by `invoke`.

   Then exit from the developer's console and launch the script
   `update.sh` (or `update.bat`). This will download the most recent
   InvokeAI (including bug fixes) and reinstall the environment.
   You should then be able to run `invoke.sh`/`invoke.bat`.

2. If this doesn't work, you can try cleaning your system's conda
   cache. This is slightly more extreme, but won't interfere with
   any other python-based programs installed on your computer.
   From the developer's console, run the command `conda clean -a`
   and answer "yes" to all prompts.

   After this is done, run `update.sh` and try again as before.

_"Corrupted configuration file."__ Everything seems to install ok, but
`invoke` complains of a corrupted configuration file and goes calls
`configure_invokeai.py` to fix, but this doesn't fix the problem.

This issue is often caused by a misconfigured configuration directive
in the `.invokeai` initialization file that contains startup settings.
This can be corrected by fixing the offending line.

First find `.invokeai`. It is a small text file located in your home
directory, `~/.invokeai` on Mac and Linux systems, and `C:\Users\*your
name*\.invokeai` on Windows systems. Open it with a text editor
(e.g. Notepad on Windows, TextEdit on Macs, or `nano` on Linux)
and look for the lines starting with `--root` and `--outdir`.

An example is here:

```cmd
--root="/home/lstein/invokeai"
--outdir="/home/lstein/invokeai/outputs"
```

There should not be whitespace before or after the directory paths,
and the paths should not end with slashes:

```cmd
--root="/home/lstein/invokeai "     # wrong! no whitespace here
--root="/home\lstein\invokeai\"     # wrong! shouldn't end in a slash
```

Fix the problem with your text editor and save as a **plain text**
file. This should clear the issue.

_If none of these maneuvers fixes the problem_ then please report the
problem to the [InvokeAI
Issues](https://github.com/invoke-ai/InvokeAI/issues) section, or
visit our [Discord Server](https://discord.gg/ZmtBAhwWhy) for interactive assistance.

## Updating to newer versions

This section describes how to update InvokeAI to new versions of the software.

### Updating the stable version

This distribution is changing rapidly, and we add new features on a daily basis.
To update to the latest released version (recommended), run the `update.sh`
(Linux/Mac) or `update.bat` (Windows) scripts. This will fetch the latest
release and re-run the `configure_invokeai` script to download any updated models
files that may be needed. You can also use this to add additional models that
you did not select at installation time.

You can now close the developer console and run `invoke` as before. If you get
complaints about missing models, then you may need to do the additional step of
running `configure_invokeai.py`. This happens relatively infrequently. To do this,
simply open up the developer's console again and type
`python scripts/configure_invokeai.py`.

## Troubleshooting

If you run into problems during or after installation, the InvokeAI team is
available to help you. Either create an
[Issue](https://github.com/invoke-ai/InvokeAI/issues) at our GitHub site, or
make a request for help on the "bugs-and-support" channel of our
[Discord server](https://discord.gg/ZmtBAhwWhy). We are a 100% volunteer
organization, but typically somebody will be available to help you within 24
hours, and often much sooner.
