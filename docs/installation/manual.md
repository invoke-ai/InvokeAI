# Manual Install

!!! warning

    **Python experience is mandatory.**

    If you want to use Invoke locally, you should probably use the [launcher](./quick_start.md).

    If you want to contribute to Invoke or run the app on the latest dev branch, instead follow the [dev environment](../contributing/dev-environment.md) guide.

InvokeAI is distributed as a python package on PyPI, installable with `pip`. There are a few things that are handled by the launcher that you'll need to manage manually, described in this guide.

## Requirements

Before you start, go through the [installation requirements](./requirements.md).

## Walkthrough

We'll use [`uv`](https://github.com/astral-sh/uv) to install python and create a virtual environment, then install the `invokeai` package. `uv` is a modern, very fast alternative to `pip`.

The following commands vary depending on the version of Invoke being installed and the system onto which it is being installed.

1. Install `uv` as described in its [docs](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). We suggest using the standalone installer method.

    Run `uv --version` to confirm that `uv` is installed and working. After installation, you may need to restart your terminal to get access to `uv`.

2. Create a directory for your installation, typically in your home directory (e.g. `~/invokeai` or `$Home/invokeai`):

    === "Linux/macOS"

        ```bash
        mkdir ~/invokeai
        cd ~/invokeai
        ```

    === "Windows (PowerShell)"

        ```bash
        mkdir $Home/invokeai
        cd $Home/invokeai
        ```

3. Create a virtual environment in that directory:

    ```sh
    uv venv --relocatable --prompt invoke --python 3.11 --python-preference only-managed .venv
    ```

    This command creates a portable virtual environment at `.venv` complete with a portable python 3.11. It doesn't matter if your system has no python installed, or has a different version - `uv` will handle everything.

4. Activate the virtual environment:

    === "Linux/macOS"

        ```bash
        source .venv/bin/activate
        ```

    === "Windows (PowerShell)"

        ```ps
        .venv\Scripts\activate
        ```

5. Choose a version to install. Review the [GitHub releases page](https://github.com/invoke-ai/InvokeAI/releases).

6. Determine the package package specifier to use when installing. This is a performance optimization.

    - If you have an Nvidia 20xx series GPU or older, use `invokeai[xformers]`.
    - If you have an Nvidia 30xx series GPU or newer, or do not have an Nvidia GPU, use `invokeai`.

7. Determine the `PyPI` index URL to use for installation, if any. This is necessary to get the right version of torch installed.

    === "Invoke v5 or later"

        - If you are on Windows with an Nvidia GPU, use `https://download.pytorch.org/whl/cu124`.
        - If you are on Linux with no GPU, use `https://download.pytorch.org/whl/cpu`.
        - If you are on Linux with an AMD GPU, use `https://download.pytorch.org/whl/rocm6.1`.
        - **In all other cases, do not use an index.**

    === "Invoke v4"

        - If you are on Windows with an Nvidia GPU, use `https://download.pytorch.org/whl/cu124`.
        - If you are on Linux with no GPU, use `https://download.pytorch.org/whl/cpu`.
        - If you are on Linux with an AMD GPU, use `https://download.pytorch.org/whl/rocm5.2`.
        - **In all other cases, do not use an index.**

8. Install the `invokeai` package. Substitute the package specifier and version.

    ```sh
    uv pip install <PACKAGE_SPECIFIER>==<VERSION> --python 3.11 --python-preference only-managed --force-reinstall
    ```

    If you determined you needed to use a `PyPI` index URL in the previous step, you'll need to add `--index=<INDEX_URL>` like this:

    ```sh
    uv pip install <PACKAGE_SPECIFIER>==<VERSION> --python 3.11 --python-preference only-managed --index=<INDEX_URL> --force-reinstall
    ```

9. Deactivate and reactivate your venv so that the invokeai-specific commands become available in the environment:

    === "Linux/macOS"

        ```bash
        deactivate && source .venv/bin/activate
        ```

    === "Windows (PowerShell)"

        ```ps
        deactivate
        .venv\Scripts\activate
        ```

10. Run the application, specifying the directory you created earlier as the root directory:

    === "Linux/macOS"

        ```bash
        invokeai-web --root ~/invokeai
        ```

    === "Windows (PowerShell)"

        ```bash
        invokeai-web --root $Home/invokeai
        ```

## Headless Install and Launch Scripts

If you run Invoke on a headless server, you might want to install and run Invoke on the command line.

We do not plan to maintain scripts to do this moving forward, instead focusing our dev resources on the GUI [launcher](../installation/quick_start.md).

You can create your own scripts for this by copying the handful of commands in this guide. `uv`'s [`pip` interface docs](https://docs.astral.sh/uv/reference/cli/#uv-pip-install) may be useful.
