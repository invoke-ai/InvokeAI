---
title: Overview
---

We offer several ways to install InvokeAI, each one suited to your
experience and preferences.

1.  [InvokeAI source code installer](INSTALL_SOURCE.md)
    This is a script that will install Python, the Anaconda ("conda")
    package manager, all of InvokeAI's its essential third party
    libraries and InvokeAI itself. It includes access to a "developer
    console" which will help us debug problems with you and give you
    to access experimental features.

    When a new InvokeAI feature is available, even between releases,
    you will be able to upgrade and try it out by running an `update`
    script. This method is recommended for individuals who wish to
    stay on the cutting edge of InvokeAI development and are not
    afraid of occasional breakage.
    
    To get started go to the bottom of the
    [Latest Release Page](https://github.com/invoke-ai/InvokeAI/releases/latest)
    and download the .zip file for your platform. Unzip the file.
    If you are on a Windows system, double-click on the `install.bat`
    script. On a Mac or Linux system, navigate to the file `install.sh`
    from within the terminal application, and run the script.
    
    Sit back and watch the script run.

    **Important Caveats**
    - This script is a bit cranky and occasionally hangs or times out,
    forcing you to cancel and restart the script (it will pick up where
    it left off).

2. [Manual Installation](INSTALL_MANUAL.md)

    In this method you will manually run the commands needed to install
    InvokeAI and its dependencies. We offer two recipes: one suited to
    those who prefer the `conda` tool, and one suited to those who prefer
    `pip` and Python virtual environments.

    This method is recommended for users who have previously used `conda`
    or `pip` in the past, developers, and anyone who wishes to remain on
    the cutting edge of future InvokeAI development and is willing to put
    up with occasional glitches and breakage.

3. [Docker Installation](INSTALL_DOCKER.md)

    We also offer a method for creating Docker containers containing
    InvokeAI and its dependencies. This method is recommended for
    individuals with experience with Docker containers and understand
    the pluses and minuses of a container-based install.

4. [Jupyter Notebooks Installation](INSTALL_JUPYTER.md)

    This method is suitable for running InvokeAI on a Google Colab
    account. It is recommended for individuals who have previously
    worked on the Colab and are comfortable with the Jupyter notebook
    environment.
