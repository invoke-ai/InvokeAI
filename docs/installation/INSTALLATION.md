# Overview

We offer several ways to install InvokeAI, each one suited to your
experience and preferences. We suggest that everyone start by
reviewing the
[hardware](010_INSTALL_AUTOMATED.md#hardware_requirements) and
[software](010_INSTALL_AUTOMATED.md#software_requirements)
requirements, as they are the same across each install method. Then
pick the install method most suitable to your level of experience and
needs.

See the [troubleshooting
section](010_INSTALL_AUTOMATED.md#troubleshooting) of the automated
install guide for frequently-encountered installation issues.

This fork is supported across Linux, Windows and Macintosh. Linux users can use
either an Nvidia-based card (with CUDA support) or an AMD card (using the ROCm
driver).

### [Installation Getting Started Guide](installation)
#### **[Automated Installer](010_INSTALL_AUTOMATED.md)**
✅ This is the recommended installation method for first-time users. 
#### [Manual Installation](020_INSTALL_MANUAL.md)
This method is recommended for experienced users and developers
#### [Docker Installation](040_INSTALL_DOCKER.md)
This method is recommended for those familiar with running Docker containers
### Other Installation Guides
  - [PyPatchMatch](060_INSTALL_PATCHMATCH.md)
  - [XFormers](070_INSTALL_XFORMERS.md)
  - [CUDA and ROCm Drivers](030_INSTALL_CUDA_AND_ROCM.md)
  - [Installing New Models](050_INSTALLING_MODELS.md)

## :fontawesome-solid-computer: Hardware Requirements

### :octicons-cpu-24: System

You wil need one of the following:

- :simple-nvidia: An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- :simple-amd: An AMD-based graphics card with 4 GB or more VRAM memory (Linux
  only)
- :fontawesome-brands-apple: An Apple computer with an M1 chip.

** SDXL 1.0 Requirements*
To use SDXL, user must have one of the following: 
- :simple-nvidia: An NVIDIA-based graphics card with 8 GB or more VRAM memory.
- :simple-amd: An AMD-based graphics card with 16 GB or more VRAM memory (Linux
  only)
- :fontawesome-brands-apple: An Apple computer with an M1 chip.


### :fontawesome-solid-memory: Memory and Disk

- At least 12 GB Main Memory RAM.
- At least 18 GB of free disk space for the machine learning model, Python, and
  all its dependencies.

We do **not recommend** the following video cards due to issues with their
running in half-precision mode and having insufficient VRAM to render 512x512
images in full-precision mode:

- NVIDIA 10xx series cards such as the 1080ti
- GTX 1650 series cards
- GTX 1660 series cards

## Installation options

1. [Automated Installer](010_INSTALL_AUTOMATED.md)

    This is a script that will install all of InvokeAI's essential
    third party libraries and InvokeAI itself. It includes access to a
    "developer console" which will help us debug problems with you and
    give you to access experimental features.


    ✅ This is the recommended option for first time users. 

2. [Manual Installation](020_INSTALL_MANUAL.md)

    In this method you will manually run the commands needed to install
    InvokeAI and its dependencies. We offer two recipes: one suited to
    those who prefer the `conda` tool, and one suited to those who prefer
    `pip` and Python virtual environments. In our hands the pip install
    is faster and more reliable, but your mileage may vary.
    Note that the conda installation method is currently deprecated and
    will not be supported at some point in the future.

    This method is recommended for users who have previously used `conda`
    or `pip` in the past, developers, and anyone who wishes to remain on
    the cutting edge of future InvokeAI development and is willing to put
    up with occasional glitches and breakage.

3. [Docker Installation](040_INSTALL_DOCKER.md)

    We also offer a method for creating Docker containers containing
    InvokeAI and its dependencies. This method is recommended for
    individuals with experience with Docker containers and understand
    the pluses and minuses of a container-based install.

## Quick Guides

* [Installing CUDA and ROCm Drivers](./030_INSTALL_CUDA_AND_ROCM.md)
* [Installing XFormers](./070_INSTALL_XFORMERS.md)
* [Installing PyPatchMatch](./060_INSTALL_PATCHMATCH.md)
* [Installing New Models](./050_INSTALLING_MODELS.md)
