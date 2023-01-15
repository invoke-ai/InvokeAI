---
title: Installing with the Automated Installer
---

# InvokeAI Automated Installation

## Introduction

The automated installer is a shell script that attempts to automate every step
needed to install and run InvokeAI on a stock computer running recent versions
of Linux, MacOS or Windows. It will leave you with a version that runs a stable
version of InvokeAI with the option to upgrade to experimental versions later.

## Walk through

1.  Make sure that your system meets the
    [hardware requirements](../index.md#hardware-requirements) and has the
    appropriate GPU drivers installed. In particular, if you are a Linux user
    with an AMD GPU installed, you may need to install the
    [ROCm driver](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

    !!! info "Required Space"

        Installation requires roughly 18G of free disk space to load the libraries and
        recommended model weights files.

        Regardless of your destination disk, your *system drive* (`C:\` on Windows, `/` on macOS/Linux) requires at least 6GB of free disk space to download and cache python dependencies. NOTE for Linux users: if your temporary directory is mounted as a `tmpfs`, ensure it has sufficient space.

2.  Check that your system has an up-to-date Python installed. To do this, open
    up a command-line window ("Terminal" on Linux and Macintosh, "Command" or
    "Powershell" on Windows) and type `python --version`. If Python is
    installed, it will print out the version number. If it is version `3.9.1` or
    higher, you meet requirements.

    !!! warning "If you see an older version, or get a command not found error"

        Go to [Python Downloads](https://www.python.org/downloads/) and
        download the appropriate installer package for your platform. We recommend
        [Version 3.10.9](https://www.python.org/downloads/release/python-3109/),
        which has been extensively tested with InvokeAI.

    !!! warning "At this time we do not recommend Python 3.11"

    _Please select your platform in the section below for platform-specific
    setup requirements._

    === "Windows users"

        - During the Python configuration process,
        look out for a checkbox to add Python to your PATH
        and select it. If the install script complains that it can't
        find python, then open the Python installer again and choose
        "Modify" existing installation.

        - Installation requires an up to date version of the Microsoft Visual C libraries. Please install the 2015-2022 libraries available here: https://learn.microsoft.com/en-us/cpp/windows/deploying-native-desktop-applications-visual-cpp?view=msvc-170

    === "Mac users"

        - After installing Python, you may need to run the
        following command from the Terminal in order to install the Web
        certificates needed to download model data from https sites. If
        you see lots of CERTIFICATE ERRORS during the last part of the
        install, this is the problem, and you can fix it with this command:

            `/Applications/Python\ 3.10/Install\ Certificates.command`

        -  You may need to install the Xcode command line tools. These
        are a set of tools that are needed to run certain applications in a
        Terminal, including InvokeAI. This package is provided directly by Apple.

              - To install, open a terminal window and run `xcode-select
              --install`. You will get a macOS system popup guiding you through the
              install. If you already have them installed, you will instead see some
              output in the Terminal advising you that the tools are already installed.

              - More information can be found here:
                https://www.freecodecamp.org/news/install-xcode-command-line-tools/

    === "Linux users"

        For reasons that are not entirely clear, installing the correct version of Python can be a bit of a challenge on Ubuntu, Linux Mint, Pop!_OS, and other Debian-derived distributions.

        On Ubuntu 22.04 and higher, run the following:

        ```
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv
        sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.10 3
        ```

        On Ubuntu 20.04, the process is slightly different:

        ```
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt install python3.10 python3-pip python3.10-venv
        sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.10 3
        ```

        Both `python` and `python3` commands are now pointing at Python3.10. You can still access older versions of Python by calling `python2`, `python3.8`, etc.

        Linux systems require a couple of additional graphics libraries to be installed for proper functioning of `python3-opencv`. Please run the following:

        `sudo apt update && sudo apt install -y libglib2.0-0 libgl1-mesa-glx`

3.  The source installer is distributed in ZIP files. Go to the
    [latest release](https://github.com/invoke-ai/InvokeAI/releases/latest), and
    look for a series of files named:

    - InvokeAI-installer-2.X.X-mac.zip
    - InvokeAI-installer-2.X.X-windows.zip
    - InvokeAI-installer-2.X.X-linux.zip

    (Where 2.X.X is the current release number).
    Download the one that is appropriate for your operating system.

4.  Unpack the zip file into a convenient directory. This will create a new
    directory named "InvokeAI-Installer". This example shows how this would look
    using the `unzip` command-line tool, but you may use any graphical or
    command-line Zip extractor:

    ```cmd
    C:\Documents\Linco> unzip InvokeAI-installer-2.X.X-windows.zip
    Archive:  C: \Linco\Downloads\InvokeAI-installer-2.X.X-windows.zip
    creating: InvokeAI-Installer\
    inflating: InvokeAI-Installer\install.bat
    inflating: InvokeAI-Installer\readme.txt
    ...
    ```

    After successful installation, you can delete the `InvokeAI-Installer`
    directory.

5.  **Windows only** Please double-click on the file WinLongPathsEnabled.reg and
    accept the dialog box that asks you if you wish to modify your registry.
    This activates long filename support on your system and will prevent
    mysterious errors during installation.

6.  If you are using a desktop GUI, double-click the installer file. It will be
    named `install.bat` on Windows systems and `install.sh` on Linux and
    Macintosh systems.

    On Windows systems you will probably get an "Untrusted Publisher" warning.
    Click on "More Info" and select "Run Anyway." You trust us, right?

7.  Alternatively, from the command line, run the shell script or .bat file:

    ```cmd
    C:\Documents\Linco> cd InvokeAI-Installer
    C:\Documents\Linco\invokeAI> install.bat
    ```

8.  The script will ask you to choose where to install InvokeAI. Select a
    directory with at least 18G of free space for a full install. InvokeAI and
    all its support files will be installed into a new directory named
    `invokeai` located at the location you specify.

    - The default is to install the `invokeai` directory in your home directory,
      usually `C:\Users\YourName\invokeai` on Windows systems,
      `/home/YourName/invokeai` on Linux systems, and `/Users/YourName/invokeai`
      on Macintoshes, where "YourName" is your login name.

    - The script uses tab autocompletion to suggest directory path completions.
      Type part of the path (e.g. "C:\Users") and press ++tab++ repeatedly
      to suggest completions.

9.  Sit back and let the install script work. It will install the third-party
    libraries needed by InvokeAI, then download the current InvokeAI release and
    install it.

    Be aware that some of the library download and install steps take a long
    time. In particular, the `pytorch` package is quite large and often appears
    to get "stuck" at 99.9%. Have patience and the installation step will
    eventually resume. However, there are occasions when the library install
    does legitimately get stuck. If you have been waiting for more than ten
    minutes and nothing is happening, you can interrupt the script with ^C. You
    may restart it and it will pick up where it left off.

10. After installation completes, the installer will launch a script called
    `configure_invokeai.py`, which will guide you through the first-time process
    of selecting one or more Stable Diffusion model weights files, downloading
    and configuring them. We provide a list of popular models that InvokeAI
    performs well with. However, you can add more weight files later on using
    the command-line client or the Web UI. See
    [Installing Models](050_INSTALLING_MODELS.md) for details.

    Note that the main Stable Diffusion weights file is protected by a license
    agreement that you must agree to in order to use. The script will list the
    steps you need to take to create an account on the official site that hosts
    the weights files, accept the agreement, and provide an access token that
    allows InvokeAI to legally download and install the weights files.

    If you have already downloaded the weights file(s) for another Stable
    Diffusion distribution, you may skip this step (by selecting "skip" when
    prompted) and configure InvokeAI to use the previously-downloaded files. The
    process for this is described in [Installing Models](050_INSTALLING_MODELS.md).

11. The script will now exit and you'll be ready to generate some images. Look
    for the directory `invokeai` installed in the location you chose at the
    beginning of the install session. Look for a shell script named `invoke.sh`
    (Linux/Mac) or `invoke.bat` (Windows). Launch the script by double-clicking
    it or typing its name at the command-line:

    ```cmd
    C:\Documents\Linco> cd invokeai
    C:\Documents\Linco\invokeAI> invoke.bat
    ```

    - The `invoke.bat` (`invoke.sh`) script will give you the choice of starting
      (1) the command-line interface, or (2) the web GUI. If you start the
      latter, you can load the user interface by pointing your browser at
      http://localhost:9090.

    - The script also offers you a third option labeled "open the developer
      console". If you choose this option, you will be dropped into a
      command-line interface in which you can run python commands directly,
      access developer tools, and launch InvokeAI with customized options.

12. You can launch InvokeAI with several different command-line arguments that
    customize its behavior. For example, you can change the location of the
    image output directory, or select your favorite sampler. See the
    [Command-Line Interface](../features/CLI.md) for a full list of the options.

        - To set defaults that will take effect every time you launch InvokeAI,
        use a text editor (e.g. Notepad) to exit the file
        `invokeai\invokeai.init`. It contains a variety of examples that you can
        follow to add and modify launch options.

!!! warning "The `invokeai` directory contains the `invoke` application, its
configuration files, the model weight files, and outputs of image generation.
Once InvokeAI is installed, do not move or remove this directory."

## Troubleshooting

### _Package dependency conflicts_

If you have previously installed InvokeAI or another Stable Diffusion package,
the installer may occasionally pick up outdated libraries and either the
installer or `invoke` will fail with complaints about library conflicts. You can
address this by entering the `invokeai` directory and running `update.sh`, which
will bring InvokeAI up to date with the latest libraries.

### ldm from pypi

!!! warning

    Some users have tried to correct dependency problems by installing
    the `ldm` package from PyPi.org. Unfortunately this is an unrelated package that
    has nothing to do with the 'latent diffusion model' used by InvokeAI. Installing
    ldm will make matters worse. If you've installed ldm, uninstall it with
    `pip uninstall ldm`.

### Corrupted configuration file

Everything seems to install ok, but `invoke` complains of a corrupted
configuration file and goes back into the configuration process (asking you to
download models, etc), but this doesn't fix the problem.

This issue is often caused by a misconfigured configuration directive in the
`invokeai\invokeai.init` initialization file that contains startup settings. The
easiest way to fix the problem is to move the file out of the way and re-run
`configure_invokeai.py`. Enter the developer's console (option 3 of the launcher
script) and run this command:

```cmd
configure_invokeai.py --root=.
```

Note the dot (.) after `--root`. It is part of the command.

_If none of these maneuvers fixes the problem_ then please report the problem to
the [InvokeAI Issues](https://github.com/invoke-ai/InvokeAI/issues) section, or
visit our [Discord Server](https://discord.gg/ZmtBAhwWhy) for interactive
assistance.

### other problems

If you run into problems during or after installation, the InvokeAI team is
available to help you. Either create an
[Issue](https://github.com/invoke-ai/InvokeAI/issues) at our GitHub site, or
make a request for help on the "bugs-and-support" channel of our
[Discord server](https://discord.gg/ZmtBAhwWhy). We are a 100% volunteer
organization, but typically somebody will be available to help you within 24
hours, and often much sooner.

## Updating to newer versions

This distribution is changing rapidly, and we add new features on a daily basis.
To update to the latest released version (recommended), run the `update.sh`
(Linux/Mac) or `update.bat` (Windows) scripts. This will fetch the latest
release and re-run the `configure_invokeai` script to download any updated
models files that may be needed. You can also use this to add additional models
that you did not select at installation time.

You can now close the developer console and run `invoke` as before. If you get
complaints about missing models, then you may need to do the additional step of
running `configure_invokeai.py`. This happens relatively infrequently. To do
this, simply open up the developer's console again and type
`python scripts/configure_invokeai.py`.

You may also use the `update` script to install any selected version of
InvokeAI. From https://github.com/invoke-ai/InvokeAI, navigate to the zip file
link of the version you wish to install. You can find the zip links by going to
the one of the release pages and looking for the **Assets** section at the
bottom. Alternatively, you can browse "branches" and "tags" at the top of the
big code directory on the InvokeAI welcome page. When you find the version you
want to install, go to the green "&lt;&gt; Code" button at the top, and copy the
"Download ZIP" link.

Now run `update.sh` (or `update.bat`) with the version number of the desired InvokeAI
version as its argument. For example, this will install the old 2.2.0 release.

```cmd
update.sh v2.2.0
```

You can get the list of version numbers by going to the [releases
page](https://github.com/invoke-ai/InvokeAI/releases) or by browsing
the (Tags)[https://github.com/invoke-ai/InvokeAI/tags] list from the
Code section of the main github page.
