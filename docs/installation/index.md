---
title: Overview
---

We offer several ways to install InvokeAI, each one suited to your
experience and preferences.

1. [InvokeAI installer](INSTALL_INVOKE.md)

    This is a installer script that installs InvokeAI and all the
    third party libraries it depends on. When a new version of
    InvokeAI is released, you will download and reinstall the new
    version.

    This installer is designed for people who want the system to "just
    work", don't have an interest in tinkering with it, and do not
    care about upgrading to unreleased experimental features.

    **Important Caveats**
    - This script does not support AMD GPUs. For Linux AMD support,
    please use the manual or source code installer methods.
    - This script has difficulty on some Macintosh machines
    that have previously been used for Python development due to
    conflicting development tools versions. Mac developers may wish
    to try the source code installer or one of the manual methods instead.

2. [Source code installer](INSTALL_SOURCE.md)

    This is a script that will install InvokeAI and all its essential
    third party libraries. In contrast to the previous installer, it
    includes access to a "developer console" which will allow you to
    access experimental features on the development branch.

    This method is recommended for individuals who are wish to stay
    on the cutting edge of InvokeAI development and are not afraid
    of occasional breakage.

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
