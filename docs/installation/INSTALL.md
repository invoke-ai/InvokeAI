---
title: Installation Overview
---

## Installation

We offer several ways to install InvokeAI, each one suited to your
experience and preferences.

1. [1-click installer](INSTALL_1CLICK.md)

    This is an automated shell script that will handle installation of
    all dependencies for you, and is recommended for those who have
    limited or no experience with the Python programming language, are
    not currently interested in contributing to the project, and just want
    the thing to install and run. In this version, you interact with the
    web server and command-line clients through a shell script named
    `invoke.sh` (Linux/Mac) or `invoke.bat` (Windows), and perform
    updates using `update.sh` and `update.bat`.

2. [Pre-compiled PIP installer](INSTALL_PCP.md)

    This is a series of installer files for which all the requirements
    for InvokeAI have been precompiled, thereby preventing the conflicts
    that sometimes occur when an external library is changed unexpectedly.
    It will leave you with an environment in which you interact directly
    with the scripts for running the web and command line clients, and
    you will update to new versions using standard developer commands.

    This method is recommended for users with a bit of experience using
    the `git` and `pip` tools.

3. [Manual Installation](INSTALL_MANUAL.md)

    In this method you will manually run the commands needed to install
    InvokeAI and its dependencies. We offer two recipes: one suited to
    those who prefer the `conda` tool, and one suited to those who prefer
    `pip` and Python virtual environments.

    This method is recommended for users who have previously used `conda`
    or `pip` in the past, developers, and anyone who wishes to remain on
    the cutting edge of future InvokeAI development and is willing to put
    up with occasional glitches and breakage.

4. [Docker Installation](INSTALL_DOCKER.md)

    We also offer a method for creating Docker containers containing
    InvokeAI and its dependencies. This method is recommended for
    individuals with experience with Docker containers and understand
    the pluses and minuses of a container-based install.

5. [Jupyter Notebooks Installation](INSTALL_JUPYTER.md)

    This method is suitable for running InvokeAI on a Google Colab
    account. It is recommended for individuals who have previously
    worked on the Colab and are comfortable with the Jupyter notebook
    environment.
