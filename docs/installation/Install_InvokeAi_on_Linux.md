---
Class: ai
Topic: InvokeAI Official Doc
Document Section: Installation
Created: 2024-07-08
Published to My Github: true
Pull Request: 
Author: Smile4yourself
---


# Install InvokeAI on Linux

## Requirements


### GPU
[[GPUs That Work with InvokeAI]]


|                  | Required                                          |     |
| ---------------- | ------------------------------------------------- | --- |
| RAM              | 12 GB minimum                                     |     |
| Hard Drive Type  | SSD gives you faster performance                  |     |
| Hard Drive Space | 100 + Gig for models + InvokeAI                   |     |
| tmpfs            | make your temp directory large enough if using it |     |
| VRAM             | [[GPUs That Work with InvokeAI]]                  |     |

%%
### Disk

SSDs will, of course, offer the best performance.

The base application disk usage depends on the torch backend.

You'll need to set aside some space for images, depending on how much you generate. A couple GB is enough to get started.

You'll need a good chunk of space for models. Even if you only install the most popular models and the usual support models (ControlNet, IP Adapter ,etc), you will quickly hit 50GB of models.

`tmpfs` on Linux

If your temporary directory is mounted as a `tmpfs`, ensure it has sufficient space.

%%

## Python

Invoke requires python 3.10 or 3.11. If you don't already have one of these versions installed, we suggest installing 3.11, as it will be supported for longer.

Check that your system has an up-to-date Python installed by running `python --version` in the terminal (Linux, macOS) or cmd/powershell (Windows).

### Installing Python (Linux)

-   Follow the [linux install instructions](https://docs.python-guide.org/starting/install3/linux/), being sure to install python 3.11.
-   You'll need to install `libglib2.0-0` and `libgl1-mesa-glx` for OpenCV to work. For example, on a Debian system: `sudo apt update && sudo apt install -y libglib2.0-0 libgl1-mesa-glx`

## Drivers

If you have an Nvidia or AMD GPU, you may need to manually install drivers or other support packages for things to work well or at all.

### Nvidia

Run `nvidia-smi` on your system's command line to verify that drivers and CUDA are installed. If this command fails, or doesn't report versions, you will need to install drivers.

Go to the [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) and carefully follow the instructions for your system to get everything installed.

Confirm that `nvidia-smi` displays driver and CUDA versions after installation.

#### Linux - via Nvidia Container Runtime[#](https://invoke-ai.github.io/InvokeAI/installation/INSTALL_REQUIREMENTS/#linux-via-nvidia-container-runtime "Permanent link")

An alternative to installing CUDA locally is to use the [Nvidia Container Runtime](https://developer.nvidia.com/container-runtime) to run the application in a container.

### AMD

Linux Only

AMD GPUs are supported on Linux only, due to ROCm (the AMD equivalent of CUDA) support being Linux only.

Bumps Ahead

While the application does run on AMD GPUs, there are occasional bumps related to spotty torch support.

Run `rocm-smi` on your system's command line verify that drivers and ROCm are installed. If this command fails, or doesn't report versions, you will need to install them.

Go to the [ROCm Documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) and carefully follow the instructions for your system to get everything installed.

Confirm that `rocm-smi` displays driver and CUDA versions after installation.

#### Linux - via Docker Container

An alternative to installing ROCm locally is to use a [ROCm docker container](https://github.com/ROCm/ROCm-docker) to run the application in a container.



## Getting the Latest Installer



[[Getting the Latest Installer for InvokeAI]] and running it for the first time.



