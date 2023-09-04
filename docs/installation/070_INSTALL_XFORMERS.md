---
title: Installing xFormers
---

# :material-image-size-select-large: Installing xformers

xFormers is toolbox that integrates with the pyTorch and CUDA
libraries to provide accelerated performance and reduced memory
consumption for applications using the transformers machine learning
architecture. After installing xFormers, InvokeAI users who have
CUDA GPUs will see a noticeable decrease in GPU memory consumption and
an increase in speed.

xFormers can be installed into a working InvokeAI installation without
any code changes or other updates. This document explains how to
install xFormers.

## Pip Install

For both Windows and Linux, you can install `xformers` in just a
couple of steps from the command line.

If you are used to launching `invoke.sh` or `invoke.bat` to start
InvokeAI, then run the launcher and select the "developer's console"
to get to the command line. If you run invoke.py directly from the
command line, then just be sure to activate it's virtual environment.

Then run the following three commands:

```sh
pip install xformers~=0.0.19
pip install triton    # WON'T WORK ON WINDOWS
python -m xformers.info output
```

The first command installs `xformers`, the second installs the
`triton` training accelerator, and the third prints out the `xformers`
installation status. On Windows, please omit the `triton` package,
which is not available on that platform.

If all goes well, you'll see a report like the
following:

```sh
xFormers 0.0.20
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.flshattF:               available
memory_efficient_attention.flshattB:               available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        available
memory_efficient_attention.tritonflashattB:        available
indexing.scaled_index_addF:                        available
indexing.scaled_index_addB:                        available
indexing.index_select:                             available
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               True
is_functorch_available:                            False
pytorch.version:                                   2.0.1+cu118
pytorch.cuda:                                      available
gpu.compute_capability:                            8.9
gpu.name:                                          NVIDIA GeForce RTX 4070
build.info:                                        available
build.cuda_version:                                1108
build.python_version:                              3.10.11
build.torch_version:                               2.0.1+cu118
build.env.TORCH_CUDA_ARCH_LIST:                    5.0+PTX 6.0 6.1 7.0 7.5 8.0 8.6
build.env.XFORMERS_BUILD_TYPE:                     Release
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              None
build.env.XFORMERS_PACKAGE_FROM:                   wheel-v0.0.20
build.nvcc_version:                                11.8.89
source.privacy:                                    open source
```

## Source Builds

`xformers` is currently under active development and at some point you
may wish to build it from sourcce to get the latest features and
bugfixes.

### Source Build on Linux

Note that xFormers only works with true NVIDIA GPUs and will not work
properly with the ROCm driver for AMD acceleration.

xFormers is not currently available as a pip binary wheel and must be
installed from source. These instructions were written for a system
running Ubuntu 22.04, but other Linux distributions should be able to
adapt this recipe.

#### 1. Install CUDA Toolkit 11.8

You will need the CUDA developer's toolkit in order to compile and
install xFormers. **Do not try to install Ubuntu's nvidia-cuda-toolkit
package.** It is out of date and will cause conflicts among the NVIDIA
driver and binaries. Instead install the CUDA Toolkit package provided
by NVIDIA itself. Go to [CUDA Toolkit 11.8
Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)
and use the target selection wizard to choose your platform and Linux
distribution. Select an installer type of "runfile (local)" at the
last step.

This will provide you with a recipe for downloading and running a
install shell script that will install the toolkit and drivers. For
example, the install script recipe for Ubuntu 22.04 running on a
x86_64 system is:

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Rather than cut-and-paste this example, We recommend that you walk
through the toolkit wizard in order to get the most up to date
installer for your system.

#### 2. Confirm/Install pyTorch 2.01 with CUDA 11.8 support

If you are using InvokeAI 3.0.2 or higher, these will already be
installed. If not, you can check whether you have the needed libraries
using a quick command. Activate the invokeai virtual environment,
either by entering the "developer's console", or manually with a
command similar to `source ~/invokeai/.venv/bin/activate` (depending
on where your `invokeai` directory is.

Then run the command:

```sh
python -c 'exec("import torch\nprint(torch.__version__)")'
```

If it prints __1.13.1+cu118__ you're good. If not, you can install the
most up to date libraries with this command:

```sh
pip install --upgrade --force-reinstall torch torchvision
```

#### 3. Install the triton module

This module isn't necessary for xFormers image inference optimization,
but avoids a startup warning.

```sh
pip install triton
```

#### 4. Install source code build prerequisites

To build xFormers from source, you will need the `build-essentials`
package. If you don't have it installed already, run:

```sh
sudo apt install build-essential
```

#### 5. Build xFormers

There is no pip wheel package for xFormers at this time (January
2023). Although there is a conda package, InvokeAI no longer
officially supports conda installations and you're on your own if you
wish to try this route.

Following the recipe provided at the [xFormers GitHub
page](https://github.com/facebookresearch/xformers), and with the
InvokeAI virtual environment active (see step 1) run the following
commands:

```sh
pip install ninja
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

The TORCH_CUDA_ARCH_LIST is a list of GPU architectures to compile
xFormer support for. You can speed up compilation by selecting
the architecture specific for your system. You'll find the list of
GPUs and their architectures at NVIDIA's [GPU Compute
Capability](https://developer.nvidia.com/cuda-gpus) table.

If the compile and install completes successfully, you can check that
xFormers is installed with this command:

```sh
python -m xformers.info
```

If suiccessful, the top of the listing should indicate "available" for
each of the `memory_efficient_attention` modules, as shown here:

```sh
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.flshattF:               available
memory_efficient_attention.flshattB:               available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        available
memory_efficient_attention.tritonflashattB:        available
[...]
```

You can now launch InvokeAI and enjoy the benefits of xFormers.

### Windows

To come


---
(c) Copyright 2023 Lincoln Stein and the InvokeAI Development Team
