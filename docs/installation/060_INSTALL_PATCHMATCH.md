---
title: Installing PyPatchMatch
---

# :material-image-size-select-large: Installing PyPatchMatch

pypatchmatch is a Python module for inpainting images. It is not needed to run
InvokeAI, but it greatly improves the quality of inpainting and outpainting and
is recommended.

Unfortunately, it is a C++ optimized module and installation can be somewhat
challenging. This guide leads you through the steps.

## Windows

You're in luck! On Windows platforms PyPatchMatch will install automatically on
Windows systems with no extra intervention.

## Macintosh

You need to have opencv installed so that pypatchmatch can be built:

```bash
brew install opencv
```

The next time you start `invoke`, after successfully installing opencv, pypatchmatch will be built.

## Linux

Prior to installing PyPatchMatch, you need to take the following steps:

### Debian Based Distros

1. Install the `build-essential` tools:

    ```sh
    sudo apt update
    sudo apt install build-essential
    ```

2. Install `opencv`:

    ```sh
    sudo apt install python3-opencv libopencv-dev
    ```

3. Activate the environment you use for invokeai, either with `conda` or with a
   virtual environment.

4. Install pypatchmatch:

    ```sh
    pip install pypatchmatch
    ```

5. Confirm that pypatchmatch is installed. At the command-line prompt enter
   `python`, and then at the `>>>` line type
   `from patchmatch import patch_match`: It should look like the following:

    ```py
    Python 3.9.5 (default, Nov 23 2021, 15:27:38)
    [GCC 9.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from patchmatch import patch_match
    Compiling and loading c extensions from "/home/lstein/Projects/InvokeAI/.invokeai-env/src/pypatchmatch/patchmatch".
    rm -rf build/obj libpatchmatch.so
    mkdir: created directory 'build/obj'
    mkdir: created directory 'build/obj/csrc/'
    [dep] csrc/masked_image.cpp ...
    [dep] csrc/nnf.cpp ...
    [dep] csrc/inpaint.cpp ...
    [dep] csrc/pyinterface.cpp ...
    [CC] csrc/pyinterface.cpp ...
    [CC] csrc/inpaint.cpp ...
    [CC] csrc/nnf.cpp ...
    [CC] csrc/masked_image.cpp ...
    [link] libpatchmatch.so ...
    ```

### Arch Based Distros

1. Install the `base-devel` package:

    ```sh
    sudo pacman -Syu
    sudo pacman -S --needed base-devel
    ```

2. Install `opencv` and `blas`:

    ```sh
    sudo pacman -S opencv blas
    ```

    or for CUDA support

    ```sh
    sudo pacman -S opencv-cuda blas
    ```
    
3. Fix the naming of the `opencv` package configuration file:

    ```sh
    cd /usr/lib/pkgconfig/
    ln -sf opencv4.pc opencv.pc
    ```

[**Next, Follow Steps 4-6 from the Debian Section above**](#linux)

If you see no errors you're ready to go!
