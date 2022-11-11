---
title: Source Installer
---

# The InvokeAI Source Installer

## Introduction

The source installer is a shell script that attempts to automate every step
needed to install and run InvokeAI on a stock computer running recent versions
of Linux, MacOS or Windows. It will leave you with a version that runs a stable
version of InvokeAI with the option to upgrade to experimental versions later.
It is not as foolproof as the [InvokeAI installer](INSTALL_INVOKE.md)

Before you begin, make sure that you meet the
[hardware requirements](index.md#Hardware_Requirements) and has the appropriate
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

    - invokeAI-src-installer-mac.zip
    - invokeAI-src-installer-windows.zip
    - invokeAI-src-installer-linux.zip

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

3.  If you are using a desktop GUI, double-click the installer file. It will be
    named `install.bat` on Windows systems and `install.sh` on Linux and
    Macintosh systems.

4.  Alternatively, form the command line, run the shell script or .bat file:

    ```cmd
    C:\Documents\Linco> cd invokeAI
    C:\Documents\Linco\invokeAI> install.bat
    ```

5.  Sit back and let the install script work. It will install various binary
    requirements including Conda, Git and Python, then download the current
    InvokeAI code and install it along with its dependencies.

6.  After installation completes, the installer will launch a script called
    `preload_models.py`, which will guide you through the first-time process of
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
    process for this is described in [Installing Models](INSTALLING_MODELS.md).

7.  The script will now exit and you'll be ready to generate some images. The
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
[Command-Line Interface](../features/CLI.md) documentation.

## Updating to newer versions

This section describes how to update InvokeAI to new versions of the software.

### Updating the stable version

This distribution is changing rapidly, and we add new features on a daily basis.
To update to the latest released version (recommended), run the `update.sh`
(Linux/Mac) or `update.bat` (Windows) scripts. This will fetch the latest
release and re-run the `preload_models` script to download any updated models
files that may be needed. You can also use this to add additional models that
you did not select at installation time.

### Updating to the development version

There may be times that there is a feature in the `development` branch of
InvokeAI that you'd like to take advantage of. Or perhaps there is a branch that
corrects an annoying bug. To do this, you will use the developer's console.

From within the invokeAI directory, run the command `invoke.sh` (Linux/Mac) or
`invoke.bat` (Windows) and selection option (3) to open the developers console.
Then run the following command to get the `development branch`:

```bash
git checkout development
git pull
conda env update
```

You can now close the developer console and run `invoke` as before. If you get
complaints about missing models, then you may need to do the additional step of
running `preload_models.py`. This happens relatively infrequently. To do this,
simply open up the developer's console again and type
`python scripts/preload_models.py`.

## Troubleshooting

If you run into problems during or after installation, the InvokeAI team is
available to help you. Either create an
[Issue](https://github.com/invoke-ai/InvokeAI/issues) at our GitHub site, or
make a request for help on the "bugs-and-support" channel of our
[Discord server](https://discord.gg/ZmtBAhwWhy). We are a 100% volunteer
organization, but typically somebody will be available to help you within 24
hours, and often much sooner.
