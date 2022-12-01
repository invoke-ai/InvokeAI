---
title: Installing PyPatchMatch
---

# :octicons-paintbrush-16: Installing PyPatchMatch

pypatchmatch is a Python module for inpainting images. It is not
needed to run InvokeAI, but it greatly improves the quality of
inpainting and outpainting and is recommended.

Unfortunately, it is a C++ optimized module and installation
can be somewhat challenging. This guide leads you through the steps.

## Windows

You're in luck! On Windows platforms PyPatchMatch will install
automatically on Windows systems with no extra intervention.

## Macintosh

PyPatchMatch is not currently supported, but the team is working on
it.

## Linux

Prior to installing PyPatchMatch, you need to take the following
steps:

1. Install the `build-essential` tools:

   ```
   sudo apt update
   sudo apt install build-essential
   ```

2. Install `opencv`:

   ```
   sudo apt install python3-opencv libopencv-dev
   ```

3. Fix the naming of the `opencv` package configuration file:

   ```
   cd /usr/lib/x86_64-linux-gnu/pkgconfig/
   ln -sf opencv4.pc opencv.pc

4. Activate the environment you use for invokeai, either with
`conda` or with a virtual environment.

5. Do a "develop" install of pypatchmatch:

   ```
   pip install -e git+https://github.com/invoke-ai/PyPatchMatch@0.1.3#egg=pypatchmatch
   ```

6. Confirm that pypatchmatch is installed.
   At the command-line prompt enter `python`, and
   then at the `>>>` line type `from patchmatch import patch_match`:
   It should look like the follwing:

   ```
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

If you see no errors, then you're ready to go!




