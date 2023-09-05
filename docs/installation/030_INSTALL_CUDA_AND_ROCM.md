---
title: NVIDIA Cuda / AMD ROCm
---

<figure markdown>

# :simple-nvidia: CUDA | :simple-amd: ROCm

</figure>

In order for InvokeAI to run at full speed, you will need a graphics
card with a supported GPU. InvokeAI supports NVidia cards via the CUDA
driver on Windows and Linux, and AMD cards via the ROCm driver on Linux.

## :simple-nvidia: CUDA

### Linux and Windows Install

If you have used your system for other graphics-intensive tasks, such
as gaming, you may very well already have the CUDA drivers
installed. To confirm, open up a command-line window and type:

```
nvidia-smi
```

If this command produces a status report on the GPU(s) installed on
your system, CUDA is installed and you have no more work to do. If
instead you get "command not found", or similar, then the driver will
need to be installed.

We strongly recommend that you install the CUDA Toolkit package
directly from NVIDIA. **Do not try to install Ubuntu's
nvidia-cuda-toolkit package. It is out of date and will cause
conflicts among the NVIDIA driver and binaries.**

Go to [CUDA Toolkit
Downloads](https://developer.nvidia.com/cuda-downloads), and use the
target selection wizard to choose your operating system, hardware
platform, and preferred installation method (e.g. "local" versus
"network").

This will provide you with a downloadable install file or, depending
on your choices, a recipe for downloading and running a install shell
script. Be sure to read and follow the full installation instructions.

After an install that seems successful, you can confirm by again
running `nvidia-smi` from the command line.

### Linux Install with a Runtime Container

On Linux systems, an alternative to installing CUDA Toolkit directly on
your system is to run an NVIDIA software container that has the CUDA
libraries already in place. This is recommended if you are already 
familiar with containerization technologies such as Docker.

For downloads and instructions, visit the [NVIDIA CUDA Container
Runtime Site](https://developer.nvidia.com/nvidia-container-runtime)

### (Optional) Cudnnn Installation for 40 series Optimization*

1) Find the InvokeAI folder
2) Click on .venv folder - e.g., YourInvokeFolderHere\.venv
3) Click on Lib folder - e.g., YourInvokeFolderHere\.venv\Lib
4) Click on site-packages folder - e.g., YourInvokeFolderHere\.venv\Lib\site-packages
5) Click on Torch directory - e.g., YourInvokeFolderHere\InvokeAI\.venv\Lib\site-packages\torch
6) Click on the lib folder - e.g., YourInvokeFolderHere\.venv\Lib\site-packages\torch\lib
7) __Copy everything inside the folder as a Backup in whatever folder you want, it's just in case.__
8) Go to https://developer.nvidia.com/cudnn
9) Log-in Or Create an account if you're not already connected
10) Download the latest version
11) Go to the folder and extract it.
12) Find the bin folder E\cudnn-windows-x86_64-__Whatever Version__\bin
13) Copy the 7 dll files
14) Go Back to YourInvokeFolderHere\.venv\Lib\site-packages\torch\lib
15) Paste the 7 dll took earlier. It should ask for replacement, accept it.
16) Enjoy !

__Very Important: You should Copy everything inside the folder of the torch lib. You do not moove It.__
*Note: 
If _no change is seen or a bug appears__ follow the same step instead just copy the Torch/lib back up folder you made earlier and replace it! If you 
didn't make a backup, you can also uninstall and reinstall torch through the command line to repair this folder.
This optimization is normally intented for the newer version of graphics card (4th series 3th series) but results have been seen with older graphics card.
So giving a try could be good.

### Torch Installation

When installing torch and torchvision manually with `pip`, remember to provide
the argument `--extra-index-url
https://download.pytorch.org/whl/cu118` as described in the [Manual
Installation Guide](020_INSTALL_MANUAL.md).

## :simple-amd: ROCm

### Linux Install

AMD GPUs are only supported on Linux platforms due to the lack of a
Windows ROCm driver at the current time. Also be aware that support
for newer AMD GPUs is spotty. Your mileage may vary.

It is possible that the ROCm driver is already installed on your
machine. To test, open up a terminal window and issue the following
command:

```
rocm-smi
```

If you get a table labeled "ROCm System Management Interface" the
driver is installed and you are done. If you get "command not found,"
then the driver needs to be installed.

Go to AMD's [ROCm Downloads
Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html#installation-methods)
and scroll to the _Installation Methods_ section. Find the subsection
for the install method for your preferred Linux distribution, and
issue the commands given in the recipe.

Annoyingly, the official AMD site does not have a recipe for the most
recent version of Ubuntu, 22.04. However, this [community-contributed
recipe](https://novaspirit.github.io/amdgpu-rocm-ubu22/) is reported
to work well.

After installation, please run `rocm-smi` a second time to confirm
that the driver is present and the GPU is recognized. You may need to
do a reboot in order to load the driver.

### Linux Install with a ROCm-docker Container

If you are comfortable with the Docker containerization system, then
you can build a ROCm docker file. The source code and installation
recipes are available
[Here](https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md)

### Torch Installation

When installing torch and torchvision manually with `pip`, remember to provide
the argument `--extra-index-url
https://download.pytorch.org/whl/rocm5.4.2` as described in the [Manual
Installation Guide](020_INSTALL_MANUAL.md).

This will be done automatically for you if you use the installer
script.

Be aware that the torch machine learning library does not seamlessly
interoperate with all AMD GPUs and you may experience garbled images,
black images, or long startup delays before rendering commences. Most
of these issues can be solved by Googling for workarounds. If you have
a problem and find a solution, please post an
[Issue](https://github.com/invoke-ai/InvokeAI/issues) so that other
users benefit and we can update this document.
