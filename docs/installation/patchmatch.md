---
title: Installing PyPatchMatch
---

PatchMatch is an algorithm used to infill images. It can greatly improve outpainting results. PyPatchMatch is a python wrapper around a C++ implementation of the algorithm.

It uses the image data around the target area as a reference to generate new image data of a similar character and quality.

## Why Use PatchMatch

In the context of image generation, "outpainting" refers to filling in a transparent area using AI-generated image data. But the AI can't generate without some initial data. We need to first fill in the transparent area with _something_.

The first step in "outpainting" then, is to fill in the transparent area with something. Generally, you get better results when that initial infill resembles the rest of the image.

Because PatchMatch generates image data so similar to the rest of the image, it works very well as the first step in outpainting, typically producing better results than other infill methods supported by Invoke (e.g. LaMA, cv2 infill, random tiles).

### Performance Caveat

PatchMatch is CPU-bound, and the amount of time it takes increases proportionally as the infill area increases. While the numbers certainly vary depending on system specs, you can expect a noticeable slowdown once you start infilling areas around 512x512 pixels. 1024x1024 pixels can take several seconds to infill.

## Installation

Unfortunately, installation can be somewhat challenging, as it requires some things that `pip` cannot install for you.

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
   Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
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
