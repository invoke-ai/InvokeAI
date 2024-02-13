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


## **[Automated Installer (Recommended)](010_INSTALL_AUTOMATED.md)**
 ✅ This is the recommended installation method for first-time users. 

  This is a script that will install all of InvokeAI's essential
  third party libraries and InvokeAI itself.

🖥️ **Download the latest installer .zip file here** : https://github.com/invoke-ai/InvokeAI/releases/latest
  
- *Look for the file labelled "InvokeAI-installer-v3.X.X.zip" at the bottom of the page*
- If you experience issues, read through the full [installation instructions](010_INSTALL_AUTOMATED.md) to make sure you have met all of the installation requirements. If you need more help, join the [Discord](discord.gg/invoke-ai) or create an issue on [Github](https://github.com/invoke-ai/InvokeAI).



## **[Manual Installation](020_INSTALL_MANUAL.md)**
This method is recommended for experienced users and developers.

  In this method you will manually run the commands needed to install
  InvokeAI and its dependencies. We offer two recipes: one suited to
  those who prefer the `conda` tool, and one suited to those who prefer
  `pip` and Python virtual environments. In our hands the pip install
  is faster and more reliable, but your mileage may vary.
  Note that the conda installation method is currently deprecated and
  will not be supported at some point in the future.

## **[Docker Installation](040_INSTALL_DOCKER.md)**
This method is recommended for those familiar with running Docker containers.

We offer a method for creating Docker containers containing InvokeAI and its dependencies. This method is recommended for individuals with experience with Docker containers and understand the pluses and minuses of a container-based install.

## Other Installation Guides
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

