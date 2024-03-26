# Manual Install

!!! warning "This is for Advanced Users"

    **Python experience is mandatory.**

## Introduction

!!! tip "Conda"

    As of InvokeAI v2.3.0 installation using the `conda` package manager is no longer being supported. It will likely still work, but we are not testing this installation method.

InvokeAI is distributed as a python package on PyPI, installable with `pip`. There are a few things that are handled by the installer that you'll need to manage manually, described in this guide.

### Requirements

Before you start, go through the [installation requirements].

### Installation Walkthrough

1. Create a directory to contain your InvokeAI library, configuration
    files, and models. This is known as the "runtime" or "root"
    directory, and often lives in your home directory under the name `invokeai`.

    We will refer to this directory as `INVOKEAI_ROOT`. For convenience, create an environment variable pointing to the directory.

    === "Linux/macOS"

        ```bash
        export INVOKEAI_ROOT=~/invokeai
        mkdir $INVOKEAI_ROOT
        ```

    === "Windows (PowerShell)"

        ```bash
        Set-Variable -Name INVOKEAI_ROOT -Value $Home/invokeai
        mkdir $INVOKEAI_ROOT
        ```

1. Enter the root (invokeai) directory and create a virtual Python environment within it named `.venv`.

    !!! info "Virtual Environment Location"

        While you may create the virtual environment anywhere in the file system, we recommend that you create it within the root directory as shown here. This allows the application to automatically detect its data directories.

        If you choose a different location for the venv, then you must set the `INVOKEAI_ROOT` environment variable or pass the directory using the `--root` CLI arg.

    ```terminal
    cd $INVOKEAI_ROOT
    python3 -m venv .venv --prompt InvokeAI
    ```

1. Activate the new environment:

    === "Linux/macOS"

        ```bash
        source .venv/bin/activate
        ```

    === "Windows"

        ```ps
        .venv\Scripts\activate
        ```

    !!! info "Permissions Error (Windows)"

        If you get a permissions error at this point, run this command and try again

        `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

    The command-line prompt should change to to show `(InvokeAI)` at the beginning of the prompt.

    The following steps should be run while inside the `INVOKEAI_ROOT` directory.

1. Make sure that pip is installed in your virtual environment and up to date:

    ```bash
    python3 -m pip install --upgrade pip
    ```

1. Install the InvokeAI Package. The `--extra-index-url` option is used to select the correct `torch` backend:

    === "CUDA (NVidia)"

         ```bash
         pip install "InvokeAI[xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu121
         ```

    === "ROCm (AMD)"

         ```bash
         pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/rocm5.6
         ```

    === "CPU (Intel Macs & non-GPU systems)"

         ```bash
         pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/cpu
         ```

    === "MPS (Apple Silicon)"

         ```bash
         pip install InvokeAI --use-pep517
         ```

1. Deactivate and reactivate your runtime directory so that the invokeai-specific commands become available in the environment:

    === "Linux/macOS"

        ```bash
        deactivate && source .venv/bin/activate
        ```

    === "Windows"

        ```ps
        deactivate
        .venv\Scripts\activate
        ```

1. Run the application:

    Run `invokeai-web` to start the UI. You must activate the virtual environment before running the app.

    If the virtual environment you selected is NOT inside `INVOKEAI_ROOT`, then you must specify the path to the root directory by adding
    `--root_dir \path\to\invokeai`.

    !!! tip

        You can permanently set the location of the runtime directory
        by setting the environment variable `INVOKEAI_ROOT` to the
        path of the directory. As mentioned previously, this is
        recommended if your virtual environment is located outside of
        your runtime directory.

## Unsupported Conda Install

Congratulations, you found the "secret" Conda installation instructions. If you really **really** want to use Conda with InvokeAI, you can do so using this unsupported recipe:

```sh
mkdir ~/invokeai
conda create -n invokeai python=3.11
conda activate invokeai
# Adjust this as described above for the appropriate torch backend
pip install InvokeAI[xformers] --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu121
invokeai-web --root ~/invokeai
```

The `pip install` command shown in this recipe is for Linux/Windows
systems with an NVIDIA GPU. See step (6) above for the command to use
with other platforms/GPU combinations. If you don't wish to pass the
`--root` argument to `invokeai` with each launch, you may set the
environment variable `INVOKEAI_ROOT` to point to the installation directory.

Note that if you run into problems with the Conda installation, the InvokeAI
staff will **not** be able to help you out. Caveat Emptor!

[installation requirements]: INSTALL_REQUIREMENTS.md
