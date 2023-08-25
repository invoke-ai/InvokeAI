---
title: Installing Manually
---

<figure markdown>

# :fontawesome-brands-linux: Linux | :fontawesome-brands-apple: macOS | :fontawesome-brands-windows: Windows

</figure>

!!! warning "This is for Advanced Users"

    **Python experience is mandatory**

## Introduction

!!! tip "Conda"
    As of InvokeAI v2.3.0 installation using the `conda` package manager is no longer being supported. It will likely still work, but we are not testing this installation method.

On Windows systems, you are encouraged to install and use the
[PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.3),
which provides compatibility with Linux and Mac shells and nice
features such as command-line completion.

### Prerequisites

Before you start, make sure you have the following preqrequisites
installed.  These are described in more detail in [Automated
Installation](010_INSTALL_AUTOMATED.md), and in many cases will
already be installed (if, for example, you have used your system for
gaming):

* **Python**

    version 3.9 through 3.11

* **CUDA Tools**

    For those with _NVidia GPUs_, you will need to
    install the [CUDA toolkit and optionally the XFormers library](070_INSTALL_XFORMERS.md).

* **ROCm Tools**

    For _Linux users with AMD GPUs_, you will need
    to install the [ROCm toolkit](./030_INSTALL_CUDA_AND_ROCM.md). Note that
    InvokeAI does not support AMD GPUs on Windows systems due to
    lack of a Windows ROCm library.

* **Visual C++ Libraries**

    _Windows users_ must install the free
    [Visual C++ libraries from Microsoft](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170)

* **The Xcode command line tools**

    for _Macintosh users_. Instructions are available at
    [Free Code Camp](https://www.freecodecamp.org/news/install-xcode-command-line-tools/)

    * _Macintosh users_ may also need to run the `Install Certificates` command
      if model downloads give lots of certificate errors. Run:
      `/Applications/Python\ 3.10/Install\ Certificates.command`

### Installation Walkthrough

To install InvokeAI with virtual environments and the PIP package
manager, please follow these steps:

1.  Please make sure you are using Python 3.9 through 3.11. The rest of the install
    procedure depends on this and will not work with other versions:

    ```bash
    python -V
    ```

2.  Create a directory to contain your InvokeAI library, configuration
    files, and models. This is known as the "runtime" or "root"
    directory, and often lives in your home directory under the name `invokeai`.

    Please keep in mind the disk space requirements - you will need at
    least 20GB for the models and the virtual environment.  From now
    on we will refer to this directory as `INVOKEAI_ROOT`. For convenience,
    the steps below create a shell variable of that name which contains the
    path to `HOME/invokeai`.

    === "Linux/Mac"

        ```bash
        export INVOKEAI_ROOT=~/invokeai
        mkdir $INVOKEAI_ROOT
        ```

    === "Windows (Powershell)"

        ```bash
        Set-Variable -Name INVOKEAI_ROOT -Value $Home/invokeai
        mkdir $INVOKEAI_ROOT
        ```

3. Enter the root (invokeai) directory and create a virtual Python
   environment within it named `.venv`. If the command `python`
   doesn't work, try `python3`. Note that while you may create the
   virtual environment anywhere in the file system, we recommend that
   you create it within the root directory as shown here. This makes
   it possible for the InvokeAI applications to find the model data
   and configuration. If you do not choose to install the virtual
   environment inside the root directory, then you **must** set the
   `INVOKEAI_ROOT` environment variable in your shell environment, for
   example, by editing `~/.bashrc` or `~/.zshrc` files, or setting the
   Windows environment variable using the Advanced System Settings dialogue.
   Refer to your operating system documentation for details.

    ```terminal
    cd $INVOKEAI_ROOT
    python -m venv .venv --prompt InvokeAI
    ```

4.  Activate the new environment:

    === "Linux/Mac"

        ```bash
        source .venv/bin/activate
        ```

    === "Windows"

        ```ps
        .venv\Scripts\activate
        ```

        If you get a permissions error at this point, run this command and try again

        `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

    The command-line prompt should change to to show `(InvokeAI)` at the
    beginning of the prompt. Note that all the following steps should be
    run while inside the INVOKEAI_ROOT directory

5.  Make sure that pip is installed in your virtual environment and up to date:

    ```bash
    python -m pip install --upgrade pip
    ```

6. Install the InvokeAI Package. The `--extra-index-url` option is used to select among
   CUDA, ROCm and CPU/MPS drivers as shown below:

    === "CUDA (NVidia)"

        ```bash
        pip install "InvokeAI[xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu118
        ```

    === "ROCm (AMD)"

        ```bash
        pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/rocm5.4.2
        ```

    === "CPU (Intel Macs & non-GPU systems)"

        ```bash
        pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/cpu
        ```

    === "MPS (M1 and M2 Macs)"

        ```bash
        pip install InvokeAI --use-pep517
        ```

7.  Deactivate and reactivate your runtime directory so that the invokeai-specific commands
    become available in the environment

    === "Linux/Macintosh"

        ```bash
        deactivate && source .venv/bin/activate
        ```

    === "Windows"

        ```ps
        deactivate
        .venv\Scripts\activate
        ```

8.  Set up the runtime directory

    In this step you will initialize your runtime directory with the downloaded
    models, model config files, directory for textual inversion embeddings, and
    your outputs.

    ```terminal
    invokeai-configure --root .
    ```
	
	Don't miss the dot at the end of the command!

    The script `invokeai-configure` will interactively guide you through the
    process of downloading and installing the weights files needed for InvokeAI.
    Note that the main Stable Diffusion weights file is protected by a license
    agreement that you have to agree to. The script will list the steps you need
    to take to create an account on the site that hosts the weights files,
    accept the agreement, and provide an access token that allows InvokeAI to
    legally download and install the weights files.

    If you get an error message about a module not being installed, check that
    the `invokeai` environment is active and if not, repeat step 5.

    !!! tip

        If you have already downloaded the weights file(s) for another Stable
        Diffusion distribution, you may skip this step (by selecting "skip" when
        prompted) and configure InvokeAI to use the previously-downloaded files. The
        process for this is described in [Installing Models](050_INSTALLING_MODELS.md).

9.  Run the command-line- or the web- interface:

    From within INVOKEAI_ROOT, activate the environment
    (with `source .venv/bin/activate` or `.venv\scripts\activate`), and then run
    the script `invokeai`. If the virtual environment you selected is NOT inside
    INVOKEAI_ROOT, then you must specify the path to the root directory by adding
    `--root_dir \path\to\invokeai` to the commands below:

    !!! example ""

        !!! warning "Make sure that the virtual environment is activated, which should create `(.venv)` in front of your prompt!"

        === "local Webserver"

            ```bash
            invokeai --web
            ```

        === "Public Webserver"

            ```bash
            invokeai --web --host 0.0.0.0
            ```

        === "CLI"

            ```bash
            invokeai
            ```

        If you choose the run the web interface, point your browser at
        http://localhost:9090 in order to load the GUI.

    !!! tip

        You can permanently set the location of the runtime directory
        by setting the environment variable `INVOKEAI_ROOT` to the
        path of the directory. As mentioned previously, this is
        *highly recommended** if your virtual environment is located outside of
        your runtime directory.

10.  Render away!

    Browse the [features](../features/index.md) section to learn about all the
    things you can do with InvokeAI.


11.  Subsequently, to relaunch the script, activate the virtual environment, and
    then launch `invokeai` command. If you forget to activate the virtual
    environment you will most likeley receive a `command not found` error.

    !!! warning

        Do not move the runtime directory after installation. The virtual environment will get confused if the directory is moved.

12. Other scripts

    The [Textual Inversion](../features/TRAINING.md) script can be launched with the command:

    ```bash
    invokeai-ti --gui
    ```

    Similarly, the [Model Merging](../features/MODEL_MERGING.md) script can be launched with the command:

    ```bash
    invokeai-merge --gui
    ```

    Leave off the `--gui` option to run the script using command-line arguments. Pass the `--help` argument
    to get usage instructions.

### Developer Install

If you have an interest in how InvokeAI works, or you would like to
add features or bugfixes, you are encouraged to install the source
code for InvokeAI. For this to work, you will need to install the
`git` source code management program. If it is not already installed
on your system, please see the [Git Installation
Guide](https://github.com/git-guides/install-git)

1. From the command line, run this command:
   ```bash
   git clone https://github.com/invoke-ai/InvokeAI.git
   ```

    This will create a directory named `InvokeAI` and populate it with the
    full source code from the InvokeAI repository.

2. Activate the InvokeAI virtual environment as per step (4) of the manual
installation protocol (important!)

3. Enter the InvokeAI repository directory and run one of these
   commands, based on your GPU:

    === "CUDA (NVidia)"
        ```bash
        pip install -e .[xformers] --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu118
        ```

    === "ROCm (AMD)"
        ```bash
        pip install -e . --use-pep517 --extra-index-url https://download.pytorch.org/whl/rocm5.4.2
        ```

    === "CPU (Intel Macs & non-GPU systems)"
        ```bash
        pip install -e . --use-pep517 --extra-index-url https://download.pytorch.org/whl/cpu
        ```

    === "MPS (M1 and M2 Macs)"
        ```bash
        pip install -e . --use-pep517
        ```

    Be sure to pass `-e` (for an editable install) and don't forget the
    dot ("."). It is part of the command.

    You can now run `invokeai` and its related commands. The code will be
    read from the repository, so that you can edit the .py source files
    and watch the code's behavior change.

4.  If you wish to contribute to the InvokeAI project, you are
    encouraged to establish a GitHub account and "fork"
    https://github.com/invoke-ai/InvokeAI into your own copy of the
    repository. You can then use GitHub functions to create and submit
    pull requests to contribute improvements to the project.

    Please see [Contributing](../index.md#contributing) for hints
    on getting started.

### Unsupported Conda Install

Congratulations, you found the "secret" Conda installation
instructions. If you really **really** want to use Conda with InvokeAI
you can do so using this unsupported recipe:

```
mkdir ~/invokeai
conda create -n invokeai python=3.10
conda activate invokeai
pip install InvokeAI[xformers] --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu118
invokeai-configure --root ~/invokeai
invokeai --root ~/invokeai --web
```

The `pip install` command shown in this recipe is for Linux/Windows
systems with an NVIDIA GPU. See step (6) above for the command to use
with other platforms/GPU combinations. If you don't wish to pass the
`--root` argument to `invokeai` with each launch, you may set the
environment variable INVOKEAI_ROOT to point to the installation directory.

Note that if you run into problems with the Conda installation, the InvokeAI
staff will **not** be able to help you out. Caveat Emptor!
