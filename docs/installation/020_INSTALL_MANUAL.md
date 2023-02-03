---
title: Installing Manually
---

<figure markdown>

# :fontawesome-brands-linux: Linux | :fontawesome-brands-apple: macOS | :fontawesome-brands-windows: Windows

</figure>

!!! warning "This is for advanced Users"

    **python experience is mandatory**

## Introduction

!!! tip As of InvokeAI v2.3.0 installation using the `conda` package manager
is no longer being supported. It will likely still work, but we are not testing
this installation method.

On Windows systems, you are encouraged to install and use the
[PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.3),
which provides compatibility with Linux and Mac shells and nice features such as
command-line completion.

To install InvokeAI with virtual environments and the PIP package manager,
please follow these steps:

1.  Please make sure you are using Python 3.9 or 3.10. The rest of the install
    procedure depends on this and will not work with other versions:

    ```bash
    python -V
    ```

2.  Clone the [InvokeAI](https://github.com/invoke-ai/InvokeAI) source code from
    GitHub:

    ```bash
    git clone https://github.com/invoke-ai/InvokeAI.git
    ```

    This will create InvokeAI folder where you will follow the rest of the
    steps.

3.  Create a directory of to contain your InvokeAI installation (known as the "runtime"
    or "root" directory). This is where your models, configs, and outputs will live
    by default. Please keep in mind the disk space requirements - you will need at
    least 18GB (as of this writing) for the models and the virtual environment.
    From now on we will refer to this directory as `INVOKEAI_ROOT`. This keeps the
    runtime directory separate from the source code and aids in updating.

    ```bash
    export INVOKEAI_ROOT="~/invokeai"
    mkdir ${INVOKEAI_ROOT}
    ```

4.  From within the InvokeAI top-level directory, create and activate a virtual
    environment named `.venv` and prompt displaying `InvokeAI`:

    ```bash
    python -m venv ${INVOKEAI_ROOT}/.venv \
        --prompt invokeai \
        --upgrade-deps \
        --copies
    source ${INVOKEAI_ROOT}/.venv/bin/activate
    ```

    !!! warning

        You **may** create your virtual environment anywhere on the filesystem.
        But IF you choose a location that is *not* inside the `$INVOKEAI_ROOT` directory,
        then you must set the `INVOKEAI_ROOT` environment variable in your shell environment,
        for example, by editing `~/.bashrc` or `~/.zshrc` files, or setting the Windows environment
        variable. Refer to your operating system / shell documentation for the correct way of doing so.

5.  Make sure that pip is installed in your virtual environment an up to date:

    ```bash
    python -m pip install --upgrade pip
    ```

6.  Install Package

    ```bash
    pip install --use-pep517 .
    ```

    Deactivate and reactivate your runtime directory so that the invokeai-specific commands
    become available in the environment

    ```
    deactivate && source ${INVOKEAI_ROOT}/.venv/bin/activate
    ```

7.  Set up the runtime directory

    In this step you will initialize your runtime directory with the downloaded
    models, model config files, directory for textual inversion embeddings, and
    your outputs.

    ```bash
    invokeai-configure --root ${INVOKEAI_ROOT}
    ```

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
        process for this is described in [here](050_INSTALLING_MODELS.md).

7.  Run the command-line- or the web- interface:

    Activate the environment (with `source .venv/bin/activate`), and then run
    the script `invokeai`. If you selected a non-default location for the
    runtime directory, please specify the path with the `--root_dir` option
    (abbreviated below as `--root`):

    !!! example ""

        !!! warning "Make sure that the virtual environment is activated, which should create `(invokeai)` in front of your prompt!"

        === "CLI"

            ```bash
            invokeai --root ~/invokeai
            ```

        === "local Webserver"

            ```bash
            invokeai --web --root ~/invokeai
            ```

        === "Public Webserver"

            ```bash
            invokeai --web --host 0.0.0.0 --root ~/invokeai
            ```

        If you choose the run the web interface, point your browser at
        http://localhost:9090 in order to load the GUI.

    !!! tip

        You can permanently set the location of the runtime directory by setting the environment variable `INVOKEAI_ROOT` to the path of the directory. As mentioned previously, this is
        **required** if your virtual environment is located outside of your runtime directory.

8.  Render away!

    Browse the [features](../features/CLI.md) section to learn about all the
    things you can do with InvokeAI.

    Note that some GPUs are slow to warm up. In particular, when using an AMD
    card with the ROCm driver, you may have to wait for over a minute the first
    time you try to generate an image. Fortunately, after the warm-up period
    rendering will be fast.

9.  Subsequently, to relaunch the script, activate the virtual environment, and
    then launch `invokeai` command. If you forget to activate the virtual
    environment you will most likeley receive a `command not found` error.

    !!! warning

        Do not move the runtime directory after installation. The virtual environment has absolute paths in it that get confused if the directory is moved.
