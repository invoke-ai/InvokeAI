---
title: Variations
---

# :material-tune-variant: Variations

## Intro

InvokeAI's support for variations enables you to do the following:

1. Generate a series of systematic variations of an image, given a prompt. The
   amount of variation from one image to the next can be controlled.

2. Given two or more variations that you like, you can combine them in a
   weighted fashion.

!!! Information ""

    This cheat sheet provides a quick guide for how this works in practice, using
    variations to create the desired image of Xena, Warrior Princess.

## Step 1 -- Find a base image that you like

The prompt we will use throughout is:

`#!bash "lucy lawless as xena, warrior princess, character portrait, high resolution."`

This will be indicated as `#!bash "prompt"` in the examples below.

First we let SD create a series of images in the usual way, in this case
requesting six iterations.

<figure markdown>
![var1](../assets/variation_walkthru/000001.3357757885.png)
<figcaption> Seed 3357757885 looks nice </figcaption>
</figure>

---

## Step 2 - Generating Variations

Let's try to generate some variations on this image. We select the "*"
symbol in the line of icons above the image in order to fix the prompt
and seed. Then we open up the "Variations" section of the generation
panel and use the slider to set the variation amount to 0.2. The
higher this value, the more each generated image will differ from the
previous one.

Now we run the prompt a second time, requesting six iterations. You
will see six images that are thematically related to each other. Try
increasing and decreasing the variation amount and see what happens.

### **Variation Sub Seeding**

Note that the output for each image has a `-V` option giving the "variant
subseed" for that image, consisting of a seed followed by the variation amount
used to generate it.

This gives us a series of closely-related variations, including the two shown
here.

<figure markdown>
![var2](../assets/variation_walkthru/000002.3647897225.png)
<figcaption>subseed 3647897225</figcaption>
</figure>

<figure markdown>
![var3](../assets/variation_walkthru/000002.1614299449.png)
<figcaption>subseed 1614299449</figcaption>
</figure>

I like the expression on Xena's face in the first one (subseed 3647897225), and
the armor on her shoulder in the second one (subseed 1614299449). Can we combine
them to get the best of both worlds?

We combine the two variations using `-V` (`--with_variations`). Again, we must
provide the seed for the originally-chosen image in order for this to work.

```bash
invoke> "prompt"  -S3357757885 -V3647897225,0.1,1614299449,0.1
Outputs:
./outputs/Xena/000003.1614299449.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1 -S3357757885
```

Here we are providing equal weights (0.1 and 0.1) for both the subseeds. The
resulting image is close, but not exactly what I wanted:

<figure markdown>
![var4](../assets/variation_walkthru/000003.1614299449.png)
<figcaption> subseed 1614299449 </figcaption>
</figure>

We could either try combining the images with different weights, or we can
generate more variations around the almost-but-not-quite image. We do the
latter, using both the `-V` (combining) and `-v` (variation strength) options.
Note that we use `-n6` to generate 6 variations:

```bash
invoke> "prompt" -S3357757885 -V3647897225,0.1,1614299449,0.1 -v0.05 -n6
Outputs:
./outputs/Xena/000004.3279757577.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,3279757577:0.05 -S3357757885
./outputs/Xena/000004.2853129515.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,2853129515:0.05 -S3357757885
./outputs/Xena/000004.3747154981.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,3747154981:0.05 -S3357757885
./outputs/Xena/000004.2664260391.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,2664260391:0.05 -S3357757885
./outputs/Xena/000004.1642517170.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,1642517170:0.05 -S3357757885
./outputs/Xena/000004.2183375608.png: "prompt" -s50 -W512 -H512 -C7.5 -Ak_lms -V 3647897225:0.1,1614299449:0.1,2183375608:0.05 -S3357757885
```

This produces six images, all slight variations on the combination of the chosen
two images. Here's the one I like best:

<figure markdown>
![var5](../assets/variation_walkthru/000004.3747154981.png)
<figcaption> subseed 3747154981 </figcaption>
</figure>

As you can see, this is a very powerful tool, which when combined with subprompt
weighting, gives you great control over the content and quality of your
generated images.

## Variations and Samplers

The sampler you choose has a strong effect on variation strength. Some
samplers, such as `k_euler_a` are very "creative" and produce significant
amounts of image-to-image variation even when the seed is fixed and the
`-v` argument is very low. Others are more deterministic. Feel free to
experiment until you find the combination that you like.

Also be aware of the [Perlin Noise](OTHER.md#thresholding-and-perlin-noise-initialization-options)
feature, which provides another way of introducing variability into your
image generation requests.
