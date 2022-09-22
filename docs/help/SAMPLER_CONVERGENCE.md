---
title: SAMPLER CONVERGENCE
---

## **Sampler Convergence**

As features keep increasing, making the right choices for your needs can become increasingly difficult. What sampler to use? And for how many steps? Do you change the CFG value? Do you use prompt weighting? Do you allow variations?

Even once you have a result, do you blend it with other images? Pass it through `img2img`? With what strength? Do you use inpainting to correct small details? Outpainting to extend cropped sections?

The purpose of this series of documents is to help you better understand these tools, so you can make the best out of them. Feel free to contribute with your own findings!

In this document, we will talk about sampler convergence.

---

### **Sampler results**

Let's start by choosing a prompt, `"an anime girl" -W512 -H512 -C7.5 -S3031912972` and using it with each of our 8 samplers.

![samplers_anime](https://user-images.githubusercontent.com/50542132/191617807-92ede41a-5be6-4a8f-a013-87e63e16bd51.png)

### **Sampler convergence**

Immediately, you can notice results tend to converge -that is, results with low `-s` (step) values are often indicative of results with high `-s` values.

You can also notice how DDIM and PLMS produce very similar results which eventually tend to converge to K-sampler results as steps are increased.
Among K-samplers, K_HEUN seems to be the quickest to converge. And finally, K_DPM_2_A and K_EULER_A seem to do a bit of their own thing and while they converge, they don't keep much similarity with the rest of samplers.

### **Batch generation speedup**

This realization is very useful because it means you don't need to create a batch of 100 images (`-n100`) at `-s100` to choose your favorite 2 or 3 images.
You can produce the same 100 images at `-s10` to `-s30` (more on that later) using a K-sampler (since they converge faster), get a rough idea of the final result, choose your 2 or 3 favorite ones, and then run `-s100` on those images to polish some details.
The latter technique is 3x to 10x faster.

### **Topic convergance**

Now, these results seem interesting, but do they hold for other topics? How about nature? Food? People? Animals? Let's try!
