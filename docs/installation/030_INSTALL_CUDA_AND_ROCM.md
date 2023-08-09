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
