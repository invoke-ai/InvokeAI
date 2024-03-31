# Requirements

## GPU

!!! warning "Problematic Nvidia GPUs"

    We do not recommend these GPUs. They cannot operate with half precision, but have insufficient VRAM to generate 512x512 images at full precision.

    - NVIDIA 10xx series cards such as the 1080 TI
    - GTX 1650 series cards
    - GTX 1660 series cards

Invoke runs best with a dedicated GPU, but will fall back to running on CPU, albeit much slower. You'll need a beefier GPU for SDXL.

!!! example "Stable Diffusion 1.5"

    === "Nvidia"

        ```
        Any GPU with at least 4GB VRAM.
        ```

    === "AMD"

        ```
        Any GPU with at least 4GB VRAM. Linux only.
        ```

    === "Mac"

        ```
        Any Apple Silicon Mac with at least 8GB memory.
        ```

!!! example "Stable Diffusion XL"

    === "Nvidia"

        ```
        Any GPU with at least 8GB VRAM. Linux only.
        ```

    === "AMD"

        ```
        Any GPU with at least 16GB VRAM.
        ```

    === "Mac"

        ```
        Any Apple Silicon Mac with at least 16GB memory.
        ```

## RAM

At least 12GB of RAM.

## Disk

SSDs will, of course, offer the best performance.

The base application disk usage depends on the torch backend.

!!! example "Disk"

    === "Nvidia (CUDA)"

        ```
        ~6.5GB
        ```

    === "AMD (ROCm)"

        ```
        ~12GB
        ```

    === "Mac (MPS)"

        ```
        ~3.5GB
        ```

You'll need to set aside some space for images, depending on how much you generate. A couple GB is enough to get started.

You'll need a good chunk of space for models. Even if you only install the most popular models and the usual support models (ControlNet, IP Adapter ,etc), you will quickly hit 50GB of models.

!!! info "`tmpfs` on Linux"

    If your temporary directory is mounted as a `tmpfs`, ensure it has sufficient space.

## Python

Invoke requires python 3.10 or 3.11. If you don't already have one of these versions installed, we suggest installing 3.11, as it will be supported for longer.

Check that your system has an up-to-date Python installed by running `python --version` in the terminal (Linux, macOS) or cmd/powershell (Windows).

<h3>Installing Python (Windows)</h3>

- Install python 3.11 with [an official installer].
- The installer includes an option to add python to your PATH. Be sure to enable this. If you missed it, re-run the installer, choose to modify an existing installation, and tick that checkbox.
- You may need to install [Microsoft Visual C++ Redistributable].

<h3>Installing Python (macOS)</h3>

- Install python 3.11 with [an official installer].
- If model installs fail with a certificate error, you may need to run this command (changing the python version to match what you have installed): `/Applications/Python\ 3.10/Install\ Certificates.command`
- If you haven't already, you will need to install the XCode CLI Tools by running `xcode-select --install` in a terminal.

<h3>Installing Python (Linux)</h3>

- Follow the [linux install instructions], being sure to install python 3.11.
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
[linux install instructions]: https://docs.python-guide.org/starting/install3/linux/
[Microsoft Visual C++ Redistributable]: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170
[an official installer]: https://www.python.org/downloads/release/python-3118/
[CUDA Toolkit Downloads]: https://developer.nvidia.com/cuda-downloads
