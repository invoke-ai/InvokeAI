---
title: Watermarking, NSFW Image Checking
---

# :material-image-off: Invisible Watermark and the NSFW Checker

## Watermarking

InvokeAI does not apply watermarking to images by default. However,
many computer scientists working in the field of generative AI worry
that a flood of computer-generated imagery will contaminate the image
data sets needed to train future generations of generative models.

InvokeAI offers an optional watermarking mode that writes a small bit
of text, **InvokeAI**, into each image that it generates using an
"invisible" watermarking library that spreads the information
throughout the image in a way that is not perceptible to the human
eye. If you are planning to share your generated images on
internet-accessible services, we encourage you to activate the
invisible watermark mode in order to help preserve the digital image
environment.

The downside of watermarking is that it increases the size of the
image moderately, and has been reported by some individuals to degrade
image quality. Your mileage may vary.

To read the watermark in an image, activate the InvokeAI virtual
environment (called the "developer's console" in the launcher) and run
the command:

```
invisible-watermark -a decode -t bytes -m dwtDct -l 64 /path/to/image.png
```

## The NSFW ("Safety") Checker

Stable Diffusion 1.5-based image generation models will produce sexual
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
License](https://github.com/invoke-ai/InvokeAI/blob/main/LICENSE-SD1+SD2.txt),
and the [Stable Diffusion XL
License][https://github.com/invoke-ai/InvokeAI/blob/main/LICENSE-SDXL.txt]
both forbid the models from being used to "exploit any of the
vulnerabilities of a specific group of persons."

For these reasons Stable Diffusion offers a "safety checker," a
machine learning model trained to recognize potentially disturbing
imagery. When a potentially NSFW image is detected, the checker will
blur the image and paste a warning icon on top. The checker can be
turned on and off in the Web interface under Settings.

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

