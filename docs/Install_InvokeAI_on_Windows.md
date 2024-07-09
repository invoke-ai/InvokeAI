---
Class: ai
Topic: InvokeAI Official Doc
Document Section: Installation
Created: 2024-07-08
Published to My Github: true
Pull Request: 
Author: Smile4yourself
---

# Install InvokeAI on Windows


## Requirements

[[GPUs That Work with InvokeAI]]


|                  | Required                         |     |
| ---------------- | -------------------------------- | --- |
| RAM              | 12 GB minimum                    |     |
| Hard Drive type  | SSD gives you faster performance |     |
| Hard Drive Space | 100 + Gig for models + InvokeAI  |     |
| VRAM             | [[GPUs That Work with InvokeAI]] |     |


%%
## Disk

SSDs will, of course, offer the best performance.

The base application disk usage depends on the torch backend.

You'll need to set aside some space for images, depending on how much you generate. A couple GB is enough to get started.

You'll need a good chunk of space for models. Even if you only install the most popular models and the usual support models (ControlNet, IP Adapter, etc), you will quickly hit 50GB of models.

`tmpfs` on Linux

If your temporary directory is mounted as a `tmpfs`, ensure it has sufficient space.

%%

## Python

Invoke requires python 3.10 or 3.11. If you don't already have one of these versions installed, we suggest installing 3.11, as it will be supported for longer.

Check that your system has an up-to-date Python installed by running `python --version` in the terminal (Linux, macOS) or cmd/powershell (Windows).

### Installing Python (Windows)

-   Install python 3.11 with [an official installer](https://www.python.org/downloads/release/python-3118/).
-   The installer includes an option to add python to your PATH. Be sure to enable this. If you missed it, re-run the installer, choose to modify an existing installation, and tick that checkbox.
-   You may need to install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170).

## Drivers

If you have an Nvidia or AMD GPU, you may need to manually install drivers or other support packages for things to work well or at all.

### Nvidia

Run `nvidia-smi` on your system's command line to verify that drivers and CUDA are installed. If this command fails, or doesn't report versions, you will need to install drivers.

Go to the [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) and carefully follow the instructions for your system to get everything installed.

Confirm that `nvidia-smi` displays driver and CUDA versions after installation.


#### Windows - Nvidia cuDNN DLLs

An out-of-date cuDNN library can greatly hamper performance on 30-series and 40-series cards. Check with the community on discord to compare your `it/s` if you think you may need this fix.

First, locate the destination for the DLL files and make a quick back up:

1.  Find your InvokeAI installation folder, e.g. `C:\Users\Username\InvokeAI\`.
2.  Open the `.venv` folder, e.g. `C:\Users\Username\InvokeAI\.venv` (you may need to show hidden files to see it).
3.  Navigate deeper to the `torch` package, e.g. `C:\Users\Username\InvokeAI\.venv\Lib\site-packages\torch`.
4.  Copy the `lib` folder inside `torch` and back it up somewhere.

Next, download and copy the updated cuDNN DLLs:

1.  Go to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn).
2.  Create an account if needed and log in.
3.  Choose the newest version of cuDNN that works with your GPU architecture. Consult the [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to determine the correct version for your GPU.
4.  Download the latest version and extract it.
5.  Find the `bin` folder, e.g. `cudnn-windows-x86_64-SOME_VERSION\bin`.
6.  Copy and paste the `.dll` files into the `lib` folder you located earlier. Replace files when prompted.

If, after restarting the app, this doesn't improve your performance, either restore your back up or re-run the installer to reset `torch` back to its original state.



[[Getting the Latest Installer for InvokeAI]] and running it for the first time.



