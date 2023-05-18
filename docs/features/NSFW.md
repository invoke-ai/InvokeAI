---
title: The NSFW Checker
---

# :material-image-off: NSFW Checker

## The NSFW ("Safety") Checker

The Stable Diffusion image generation models will produce sexual
imagery if deliberately prompted, and will occasionally produce such
images when this is not intended. Such images are colloquially known
as "Not Safe for Work" (NSFW). This behavior is due to the nature of
the training set that Stable Diffusion was trained on, which culled
millions of "aesthetic" images from the Internet.

You may not wish to be exposed to these images, and in some
jurisdictions it may be illegal to publicly distribute such imagery,
including mounting a publicly-available server that provides
unfiltered images to the public. Furthermore, the [Stable Diffusion
weights
License](https://github.com/invoke-ai/InvokeAI/blob/main/LICENSE-ModelWeights.txt)
forbids the model from being used to "exploit any of the
vulnerabilities of a specific group of persons."

For these reasons Stable Diffusion offers a "safety checker," a
machine learning model trained to recognize potentially disturbing
imagery. When a potentially NSFW image is detected, the checker will
blur the image and paste a warning icon on top. The checker can be
turned on and off on the command line using `--nsfw_checker` and
`--no-nsfw_checker`.

At installation time, InvokeAI will ask whether the checker should be
activated by default (neither argument given on the command line). The
response is stored in the InvokeAI initialization file (usually
`invokeai.init` in your home directory). You can change the default at any
time by opening this file in a text editor and commenting or
uncommenting the line `--nsfw_checker`.

## Caveats

There are a number of caveats that you need to be aware of.

### Accuracy

The checker is [not perfect](https://arxiv.org/abs/2210.04610).It will
occasionally flag innocuous images (false positives), and will
frequently miss violent and gory imagery (false negatives). It rarely
fails to flag sexual imagery, but this has been known to happen. For
these reasons, the InvokeAI team prefers to refer to the software as a
"NSFW Checker" rather than "safety checker."

### Memory Usage and Performance

The NSFW checker consumes an additional 1.2G of GPU VRAM on top of the
3.4G of VRAM used by Stable Diffusion v1.5 (this is with
half-precision arithmetic). This means that the checker will not run
successfully on GPU cards with less than 6GB VRAM, and will reduce the
size of the images that you can produce.

The checker also introduces a slight performance penalty. Images will
take ~1 second longer to generate when the checker is
activated. Generally this is not noticeable.

### Intermediate Images in the Web UI

The checker only operates on the final image produced by the Stable
Diffusion algorithm. If you are using the Web UI and have enabled the
display of intermediate images, you will briefly be exposed to a
low-resolution (mosaicized) version of the final image before it is
flagged by the checker and replaced by a fully blurred version. You
are encouraged to turn **off** intermediate image rendering when you
are using the checker. Future versions of InvokeAI will apply
additional blurring to intermediate images when the checker is active.

### Watermarking

InvokeAI does not apply any sort of watermark to images it
generates. However, it does write metadata into the PNG data area,
including the prompt used to generate the image and relevant parameter
settings. These fields can be examined using the `sd-metadata.py`
script that comes with the InvokeAI package.

Note that several other Stable Diffusion distributions offer
wavelet-based "invisible" watermarking. We have experimented with the
library used to generate these watermarks and have reached the
conclusion that while the watermarking library may be adding
watermarks to PNG images, the currently available version is unable to
retrieve them successfully. If and when a functioning version of the
library becomes available, we will offer this feature as well.
