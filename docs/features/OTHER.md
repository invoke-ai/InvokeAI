---
title: Others
---

# :fontawesome-regular-share-from-square: Others

## **Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg){ align="right" }](https://colab.research.google.com/github/lstein/stable-diffusion/blob/main/notebooks/Stable_Diffusion_AI_Notebook.ipynb)

Open and follow instructions to use an isolated environment running Dream.

Output Example:

![Colab Notebook](../assets/colab_notebook.png)

---

## **Seamless Tiling**

The seamless tiling mode causes generated images to seamlessly tile
with itself creating repetitive wallpaper-like patterns. To use it,
activate the Seamless Tiling option in the Web GUI and then select
whether to tile on the X (horizontal) and/or Y (vertical) axes. Tiling
will then be active for the next set of generations.

A nice prompt to test seamless tiling with is:

```
pond garden with lotus by claude monet"
```

---

## **Weighted Prompts**

You may weight different sections of the prompt to tell the sampler to attach different levels of
priority to them, by adding `:<percent>` to the end of the section you wish to up- or downweight. For
example consider this prompt:

```bash
tabby cat:0.25 white duck:0.75 hybrid
```

This will tell the sampler to invest 25% of its effort on the tabby cat aspect of the image and 75%
on the white duck aspect (surprisingly, this example actually works). The prompt weights can use any
combination of integers and floating point numbers, and they do not need to add up to 1.

## **Thresholding and Perlin Noise Initialization Options**

Under the Noise section of the Web UI, you will find two options named
Perlin Noise and Noise Threshold. [Perlin
noise](https://en.wikipedia.org/wiki/Perlin_noise) is a type of
structured noise used to simulate terrain and other natural
textures. The slider controls the percentage of perlin noise that will
be mixed into the image at the beginning of generation. Adding a little
perlin noise to a generation will alter the image substantially.

The noise threshold limits the range of the latent values during
sampling and helps combat the oversharpening seem with higher CFG
scale values.

For better intuition into what these options do in practice:

![here is a graphic demonstrating them both](../assets/truncation_comparison.jpg)

In generating this graphic, perlin noise at initialization was
programmatically varied going across on the diagram by values 0.0,
0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0; and the threshold was varied
going down from 0, 1, 2, 3, 4, 5, 10, 20, 100. The other options are
fixed using the prompt "a portrait of a beautiful young lady" a CFG of
20, 100 steps, and a seed of 1950357039.
