---
title: Installing Manually
---

<figure markdown>

# :fontawesome-brands-linux: Linux | :fontawesome-brands-apple: macOS | :fontawesome-brands-windows: Windows

</figure>

!!! warning "This is for advanced Users"

    **python experience is mandatory**

## Introduction

You have two choices for manual installation. The [first one](#pip-Install) uses
basic Python virtual environment (`venv`) command and `pip` package manager. The
[second one](#Conda-method) uses Anaconda3 package manager (`conda`). Both
methods require you to enter commands on the terminal, also known as the
"console".

Note that the `conda` installation method is currently deprecated and will not
be supported at some point in the future.

On Windows systems, you are encouraged to install and use the
[PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.3),
which provides compatibility with Linux and Mac shells and nice features such as
command-line completion.

## pip Install

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

3.  From within the InvokeAI top-level directory, create and activate a virtual
    environment named `.venv` and prompt displaying `InvokeAI`:

      ```bash
      python -m venv .venv \
          --prompt InvokeAI \
          --upgrade-deps
      source .venv/bin/activate
      ```

4.  Make sure that pip is installed in your virtual environment an up to date:

      ```bash
      python -m ensurepip \
          --upgrade
      python -m pip install \
          --upgrade pip
      ```

5.  Install Package

      ```bash
      pip install --use-pep517 .
      ```

6.  Set up the runtime directory

    In this step you will initialize a runtime directory that will contain the
    models, model config files, directory for textual inversion embeddings, and
    your outputs. This keeps the runtime directory separate from the source code
    and aids in updating.

    You may pick any location for this directory using the `--root_dir` option
    (abbreviated --root). If you don't pass this option, it will default to
    `~/invokeai`.

    ```bash
    invokeai-configure --root_dir ~/Programs/invokeai
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

    Note that `invokeai-configure` and `invokeai` should be installed under your
    virtual environment directory and the system should find them on the PATH.
    If this isn't working on your system, you can call the scripts directory
    using `python scripts/configure_invokeai.py` and `python scripts/invoke.py`.

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
            invoke.py --root ~/Programs/invokeai
            ```

        === "local Webserver"

            ```bash
            invoke.py --web --root ~/Programs/invokeai
            ```

        === "Public Webserver"

            ```bash
            invoke.py --web --host 0.0.0.0 --root ~/Programs/invokeai
            ```

        If you choose the run the web interface, point your browser at
        http://localhost:9090 in order to load the GUI.

    !!! tip

        You can permanently set the location of the runtime directory by setting the environment variable INVOKEAI_ROOT to the path of the directory.

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

    !!! tip

        Do not move the source code repository after installation. The virtual environment directory has absolute paths in it that get confused if the directory is moved.

## Creating an "install" version of InvokeAI

If you wish you can install InvokeAI and all its dependencies in the runtime
directory. This allows you to delete the source code repository and eliminates
the need to provide `--root_dir` at startup time. Note that this method only
works with the PIP method.

1. Follow the instructions for the PIP install, but in step #2 put the virtual
   environment into the runtime directory. For example, assuming the runtime
   directory lives in `~/Programs/invokeai`, you'd run:

    ```bash
    python -m venv ~/Programs/invokeai
    ```

2. Now follow steps 3 to 5 in the PIP recipe, ending with the `pip install`
   step.

3. Run one additional step while you are in the source code repository directory

    ```
    pip install --use-pep517 . # note the dot in the end!!!
    ```

4. That's all! Now, whenever you activate the virtual environment, `invokeai`
   will know where to look for the runtime directory without needing a
   `--root_dir` argument. In addition, you can now move or delete the source
   code repository entirely.

   (Don't move the runtime directory!)

