---
title: Manual Installation
---

<figure markdown>
# :fontawesome-brands-linux: Linux | :fontawesome-brands-apple: macOS | :fontawesome-brands-windows: Windows
</figure>

!!! warning "This is for advanced Users"

    who are already experienced with using conda or pip

## Introduction

You have two choices for manual installation, the [first one](#Conda_method)
based on the Anaconda3 package manager (`conda`), and
[a second one](#PIP_method) which uses basic Python virtual environment (`venv`)
commands and the PIP package manager. Both methods require you to enter commands
on the terminal, also known as the "console".

On Windows systems you are encouraged to install and use the
[Powershell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.3),
which provides compatibility with Linux and Mac shells and nice features such as
command-line completion.

### Conda method

1.  Check that your system meets the
    [hardware requirements](index.md#Hardware_Requirements) and has the
    appropriate GPU drivers installed. In particular, if you are a Linux user
    with an AMD GPU installed, you may need to install the
    [ROCm driver](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

    InvokeAI does not yet support Windows machines with AMD GPUs due to the lack
    of ROCm driver support on this platform.

    To confirm that the appropriate drivers are installed, run `nvidia-smi` on
    NVIDIA/CUDA systems, and `rocm-smi` on AMD systems. These should return
    information about the installed video card.

    Macintosh users with MPS acceleration, or anybody with a CPU-only system,
    can skip this step.

2.  You will need to install Anaconda3 and Git if they are not already
    available. Use your operating system's preferred package manager, or
    download the installers manually. You can find them here:

    - [Anaconda3](https://www.anaconda.com/)
    - [git](https://git-scm.com/downloads)

3.  Clone the [InvokeAI](https://github.com/invoke-ai/InvokeAI) source code from
    GitHub:

    ```bash
    git clone https://github.com/invoke-ai/InvokeAI.git
    ```

    This will create InvokeAI folder where you will follow the rest of the
    steps.

4.  Enter the newly-created InvokeAI folder:

    ```bash
    cd InvokeAI
    ```

    From this step forward make sure that you are working in the InvokeAI
    directory!

5.  Select the appropriate environment file:

    We have created a series of environment files suited for different operating
    systems and GPU hardware. They are located in the
    `environments-and-requirements` directory:

    <figure markdown>

    |         filename         |               OS               |
    | :----------------------: | :----------------------------: |
    | environment-lin-amd.yml  |  Linux with an AMD (ROCm) GPU  |
    | environment-lin-cuda.yml | Linux with an NVIDIA CUDA GPU  |
    | environment-mac.yml    |   Macintosh            |
    | environment-win-cuda.yml | Windows with an NVIDA CUDA GPU |

    </figure>

    Choose the appropriate environment file for your system and link or copy it
    to `environment.yml` in InvokeAI's top-level directory. To do so, run
    following command from the repository-root:

    !!! Example ""

        === "Macintosh and Linux"

            !!! todo "Replace `xxx` and `yyy` with the appropriate OS and GPU codes as seen in the table above"

            ```bash
            ln -sf environments-and-requirements/environment-xxx-yyy.yml environment.yml
            ```

            When this is done, confirm that a file `environment.yml` has been linked in
            the InvokeAI root directory and that it points to the correct file in the
            `environments-and-requirements`.

            ```bash
            ls -la
            ```

        === "Windows"

            !!! todo " Since it requires admin privileges to create links, we will use the copy command to create your `environment.yml`"

            ```cmd
            copy environments-and-requirements\environment-win-cuda.yml environment.yml
            ```

            Afterwards verify that the file `environment.yml` has been created, either via the
            explorer or by using the command `dir` from the terminal

            ```cmd
            dir
            ```

    !!! warning "Do not try to run conda on directly on the subdirectory environments file. This won't work. Instead, copy or link it to the top-level directory as shown."

6.  Create the conda environment:

    ```bash
    conda env update
    ```

    This will create a new environment named `invokeai` and install all InvokeAI
    dependencies into it. If something goes wrong you should take a look at
    [troubleshooting](#troubleshooting).

7.  Activate the `invokeai` environment:

    In order to use the newly created environment you will first need to
    activate it

    ```bash
    conda activate invokeai
    ```

    Your command-line prompt should change to indicate that `invokeai` is active
    by prepending `(invokeai)`.

8.  Pre-Load the model weights files:

    !!! tip

        If you have already downloaded the weights file(s) for another Stable
        Diffusion distribution, you may skip this step (by selecting "skip" when
        prompted) and configure InvokeAI to use the previously-downloaded files. The
        process for this is described in [here](050_INSTALLING_MODELS.md).

    ```bash
    python scripts/configure_invokeai.py
    ```

    The script `configure_invokeai.py` will interactively guide you through the
    process of downloading and installing the weights files needed for InvokeAI.
    Note that the main Stable Diffusion weights file is protected by a license
    agreement that you have to agree to. The script will list the steps you need
    to take to create an account on the site that hosts the weights files,
    accept the agreement, and provide an access token that allows InvokeAI to
    legally download and install the weights files.

    If you get an error message about a module not being installed, check that
    the `invokeai` environment is active and if not, repeat step 5.

9.  Run the command-line- or the web- interface:

    !!! example ""

        !!! warning "Make sure that the conda environment is activated, which should create `(invokeai)` in front of your prompt!"

        === "CLI"

            ```bash
            python scripts/invoke.py
            ```

        === "local Webserver"

            ```bash
            python scripts/invoke.py --web
            ```

        === "Public Webserver"

            ```bash
            python scripts/invoke.py --web --host 0.0.0.0
            ```

        If you choose the run the web interface, point your browser at
        http://localhost:9090 in order to load the GUI.

10. Render away!

    Browse the [features](../features/CLI.md) section to learn about all the things you
    can do with InvokeAI.

    Note that some GPUs are slow to warm up. In particular, when using an AMD
    card with the ROCm driver, you may have to wait for over a minute the first
    time you try to generate an image. Fortunately, after the warm up period
    rendering will be fast.

11. Subsequently, to relaunch the script, be sure to run "conda activate
    invokeai", enter the `InvokeAI` directory, and then launch the invoke
    script. If you forget to activate the 'invokeai' environment, the script
    will fail with multiple `ModuleNotFound` errors.

## Updating to newer versions of the script

This distribution is changing rapidly. If you used the `git clone` method
(step 5) to download the InvokeAI directory, then to update to the latest and
greatest version, launch the Anaconda window, enter `InvokeAI` and type:

```bash
git pull
conda env update
python scripts/configure_invokeai.py --no-interactive  #optional
```

This will bring your local copy into sync with the remote one. The last step may
be needed to take advantage of new features or released models. The
`--no-interactive` flag will prevent the script from prompting you to download
the big Stable Diffusion weights files.

## pip Install

To install InvokeAI with only the PIP package manager, please follow these
steps:

1.  Make sure you are using Python 3.9 or higher. The rest of the install
    procedure depends on this:

    ```bash
    python -V
    ```

2.  Install the `virtualenv` tool if you don't have it already:

    ```bash
    pip install virtualenv
    ```

3.  From within the InvokeAI top-level directory, create and activate a virtual
    environment named `invokeai`:

    ```bash
    virtualenv invokeai
    source invokeai/bin/activate
    ```

4.  Pick the correct `requirements*.txt` file for your hardware and operating
    system.

    We have created a series of environment files suited for different operating
    systems and GPU hardware. They are located in the
    `environments-and-requirements` directory:

    <figure markdown>

    |              filename               |                               OS                                |
    | :---------------------------------: | :-------------------------------------------------------------: |
    |      requirements-lin-amd.txt       |                  Linux with an AMD (ROCm) GPU                   |
    |     requirements-lin-arm64.txt      |                 Linux running on arm64 systems                  |
    |      requirements-lin-cuda.txt      |                 Linux with an NVIDIA (CUDA) GPU                 |
    |    requirements-mac-mps-cpu.txt     |                Macintoshes with MPS acceleration                |
    | requirements-lin-win-colab-cuda.txt | Windows with an NVIDA (CUDA) GPU<br>(supports Google Colab too) |

    </figure>

    Select the appropriate requirements file, and make a link to it from
    `requirements.txt` in the top-level InvokeAI directory. The command to do
    this from the top-level directory is:

    !!! example ""

        === "Macintosh and Linux"

            !!! info "Replace `xxx` and `yyy` with the appropriate OS and GPU codes."

            ```bash
            ln -sf environments-and-requirements/requirements-xxx-yyy.txt requirements.txt
            ```

        === "Windows"

            !!! info "on Windows, admin privileges are required to make links, so we use the copy command instead"

            ```cmd
            copy environments-and-requirements\requirements-lin-win-colab-cuda.txt requirements.txt
            ```

        !!! warning

            Please do not link or copy `environments-and-requirements/requirements-base.txt`.
            This is a base requirements file that does not have the platform-specific
            libraries. Also, be sure to link or copy the platform-specific file to
	    a top-level file named `requirements.txt` as shown here. Running pip on
	    a requirements file in a subdirectory will not work as expected.

    When this is done, confirm that a file named `requirements.txt` has been
    created in the InvokeAI root directory and that it points to the correct
    file in `environments-and-requirements`.

5.  Run PIP

    Be sure that the `invokeai` environment is active before doing this:

    ```bash
    pip install --prefer-binary -r requirements.txt
    ```

---

## Troubleshooting

Here are some common issues and their suggested solutions.

### Conda

#### Conda fails before completing `conda update`

The usual source of these errors is a package incompatibility. While we have
tried to minimize these, over time packages get updated and sometimes introduce
incompatibilities.

We suggest that you search
[Issues](https://github.com/invoke-ai/InvokeAI/issues) or the "bugs-and-support"
channel of the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy).

You may also try to install the broken packages manually using PIP. To do this,
activate the `invokeai` environment, and run `pip install` with the name and
version of the package that is causing the incompatibility. For example:

```bash
pip install test-tube==0.7.5
```

You can keep doing this until all requirements are satisfied and the `invoke.py`
script runs without errors. Please report to
[Issues](https://github.com/invoke-ai/InvokeAI/issues) what you were able to do
to work around the problem so that others can benefit from your investigation.

### Create Conda Environment fails on MacOS

If conda create environment fails with lmdb error, this is most likely caused by Clang.
Run brew config to see which Clang is installed on your Mac. If Clang isn't installed, that's causing the error.
Start by installing additional XCode command line tools, followed by brew install llvm.

```bash
xcode-select --install
brew install llvm
```

If brew config has Clang installed, update to the latest llvm and try creating the environment again.

#### `configure_invokeai.py` or `invoke.py` crashes at an early stage

This is usually due to an incomplete or corrupted Conda install. Make sure you
have linked to the correct environment file and run `conda update` again.

If the problem persists, a more extreme measure is to clear Conda's caches and
remove the `invokeai` environment:

```bash
conda deactivate
conda env remove -n invokeai
conda clean -a
conda update
```

This removes all cached library files, including ones that may have been
corrupted somehow. (This is not supposed to happen, but does anyway).

#### `invoke.py` crashes at a later stage

If the CLI or web site had been working ok, but something unexpected happens
later on during the session, you've encountered a code bug that is probably
unrelated to an install issue. Please search
[Issues](https://github.com/invoke-ai/InvokeAI/issues), file a bug report, or
ask for help on [Discord](https://discord.gg/ZmtBAhwWhy)

#### My renders are running very slowly

You may have installed the wrong torch (machine learning) package, and the
system is running on CPU rather than the GPU. To check, look at the log messages
that appear when `invoke.py` is first starting up. One of the earlier lines
should say `Using device type cuda`. On AMD systems, it will also say "cuda",
and on Macintoshes, it should say "mps". If instead the message says it is
running on "cpu", then you may need to install the correct torch library.

You may be able to fix this by installing a different torch library. Here are
the magic incantations for Conda and PIP.

!!! todo "For CUDA systems"

    - conda

      ```bash
      conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
      ```

    - pip

      ```bash
      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
      ```

!!! todo "For AMD systems"

    - conda

      ```bash
      conda activate invokeai
      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2/
      ```

    - pip

      ```bash
      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2/
      ```

More information and troubleshooting tips can be found at https://pytorch.org.
