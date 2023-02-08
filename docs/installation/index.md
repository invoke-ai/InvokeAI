---
title: Overview
---

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

## Main Application

1. [Automated Installer](010_INSTALL_AUTOMATED.md)

    This is a script that will install all of InvokeAI's essential
    third party libraries and InvokeAI itself. It includes access to a
    "developer console" which will help us debug problems with you and
    give you to access experimental features.

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
