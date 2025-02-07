# Requirements

Invoke runs on Windows 10+, macOS 14+ and Linux (Ubuntu 20.04+ is well-tested).

## Hardware

Hardware requirements vary significantly depending on model and image output size.

The requirements below are rough guidelines for best performance. GPUs with less VRAM typically still work, if a bit slower. Follow the [Low-VRAM mode guide](./features/low-vram.md) to optimize performance.

- All Apple Silicon (M1, M2, etc) Macs work, but 16GB+ memory is recommended.
- AMD GPUs are supported on Linux only. The VRAM requirements are the same as Nvidia GPUs.

!!! info "Hardware Requirements (Windows/Linux)"

    === "SD1.5 - 512×512"

        - GPU: Nvidia 10xx series or later, 4GB+ VRAM.
        - Memory: At least 8GB RAM.
        - Disk: 10GB for base installation plus 30GB for models.

    === "SDXL - 1024×1024"

        - GPU: Nvidia 20xx series or later, 8GB+ VRAM.
        - Memory: At least 16GB RAM.
        - Disk: 10GB for base installation plus 100GB for models.

    === "FLUX - 1024×1024"

        - GPU: Nvidia 20xx series or later, 10GB+ VRAM.
        - Memory: At least 32GB RAM.
        - Disk: 10GB for base installation plus 200GB for models.

!!! info "`tmpfs` on Linux"

    If your temporary directory is mounted as a `tmpfs`, ensure it has sufficient space.

## Python

!!! tip "The launcher installs python for you"

    You don't need to do this if you are installing with the [Invoke Launcher](./quick_start.md).

Invoke requires python 3.10 or 3.11. If you don't already have one of these versions installed, we suggest installing 3.11, as it will be supported for longer.

Check that your system has an up-to-date Python installed by running `python3 --version` in the terminal (Linux, macOS) or cmd/powershell (Windows).

!!! info "Installing Python"

    === "Windows"

        - Install python 3.11 with [an official installer].
        - The installer includes an option to add python to your PATH. Be sure to enable this. If you missed it, re-run the installer, choose to modify an existing installation, and tick that checkbox.
        - You may need to install [Microsoft Visual C++ Redistributable].

    === "macOS"

        - Install python 3.11 with [an official installer].
        - If model installs fail with a certificate error, you may need to run this command (changing the python version to match what you have installed): `/Applications/Python\ 3.10/Install\ Certificates.command`
        - If you haven't already, you will need to install the XCode CLI Tools by running `xcode-select --install` in a terminal.

    === "Linux"

        - Installing python varies depending on your system. On Ubuntu, you can use the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).
        - You'll need to install `libglib2.0-0` and `libgl1-mesa-glx` for OpenCV to work. For example, on a Debian system: `sudo apt update && sudo apt install -y libglib2.0-0 libgl1-mesa-glx`

## Drivers

If you have an Nvidia or AMD GPU, you may need to manually install drivers or other support packages for things to work well or at all.

### Nvidia

Run `nvidia-smi` on your system's command line to verify that drivers and CUDA are installed. If this command fails, or doesn't report versions, you will need to install drivers.

Go to the [CUDA Toolkit Downloads] and carefully follow the instructions for your system to get everything installed.

Confirm that `nvidia-smi` displays driver and CUDA versions after installation.

#### Linux - via Nvidia Container Runtime

An alternative to installing CUDA locally is to use the [Nvidia Container Runtime] to run the application in a container.

#### Windows - Nvidia cuDNN DLLs

An out-of-date cuDNN library can greatly hamper performance on 30-series and 40-series cards. Check with the community on discord to compare your `it/s` if you think you may need this fix.

First, locate the destination for the DLL files and make a quick back up:

1. Find your InvokeAI installation folder, e.g. `C:\Users\Username\InvokeAI\`.
1. Open the `.venv` folder, e.g. `C:\Users\Username\InvokeAI\.venv` (you may need to show hidden files to see it).
1. Navigate deeper to the `torch` package, e.g. `C:\Users\Username\InvokeAI\.venv\Lib\site-packages\torch`.
1. Copy the `lib` folder inside `torch` and back it up somewhere.

Next, download and copy the updated cuDNN DLLs:

1. Go to <https://developer.nvidia.com/cudnn>.
1. Create an account if needed and log in.
1. Choose the newest version of cuDNN that works with your GPU architecture. Consult the [cuDNN support matrix] to determine the correct version for your GPU.
1. Download the latest version and extract it.
1. Find the `bin` folder, e.g. `cudnn-windows-x86_64-SOME_VERSION\bin`.
1. Copy and paste the `.dll` files into the `lib` folder you located earlier. Replace files when prompted.

If, after restarting the app, this doesn't improve your performance, either restore your back up or re-run the installer to reset `torch` back to its original state.

### AMD

!!! info "Linux Only"

    AMD GPUs are supported on Linux only, due to ROCm (the AMD equivalent of CUDA) support being Linux only.

!!! warning "Bumps Ahead"

    While the application does run on AMD GPUs, there are occasional bumps related to spotty torch support.

Run `rocm-smi` on your system's command line verify that drivers and ROCm are installed. If this command fails, or doesn't report versions, you will need to install them.

Go to the [ROCm Documentation] and carefully follow the instructions for your system to get everything installed.

Confirm that `rocm-smi` displays driver and CUDA versions after installation.

#### Linux - via Docker Container

An alternative to installing ROCm locally is to use a [ROCm docker container] to run the application in a container.

[ROCm docker container]: https://github.com/ROCm/ROCm-docker
[ROCm Documentation]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html
[cuDNN support matrix]: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
[Nvidia Container Runtime]: https://developer.nvidia.com/container-runtime
[CUDA Toolkit Downloads]: https://developer.nvidia.com/cuda-downloads
