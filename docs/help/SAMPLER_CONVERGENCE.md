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

<img width="1082" alt="image" src="https://user-images.githubusercontent.com/50542132/191636411-083c8282-6ed1-4f78-9273-ee87c0a0f1b6.png">

### **Sampler convergence**

Immediately, you can notice results tend to converge -that is, results with low `-s` (step) values are often indicative of results with high `-s` values.

You can also notice how DDIM and PLMS eventually tend to converge to K-sampler results as steps are increased.
Among K-samplers, K_HEUN and K_DPM_2 seem to be the quickest to converge. And finally, K_DPM_2_A and K_EULER_A seem to do a bit of their own thing and don't keep much similarity with the rest of the samplers.

### **Batch generation speedup**

This realization is very useful because it means you don't need to create a batch of 100 images (`-n100`) at `-s100` to choose your favorite 2 or 3 images.
You can produce the same 100 images at `-s10` to `-s30` (more on that later) using a K-sampler (since they converge faster), get a rough idea of the final result, choose your 2 or 3 favorite ones, and then run `-s100` on those images to polish some details.
The latter technique is 3x to 10x faster.

### **Topic convergance**

Now, these results seem interesting, but do they hold for other topics? How about nature? Food? People? Animals? Let's try!

Nature. `"valley landscape wallpaper, d&d art, fantasy, painted, 4k, high detail, sharp focus, washed colors, elaborate excellent painted illustration" -W512 -H512 -C7.5 -S1458228930`

![samplers-nature-2](https://user-images.githubusercontent.com/50542132/191632502-218cdd83-9808-47c5-8e71-c2bf113642fc.png)

Food. `"a hamburger with a bowl of french fries" -W512 -H512 -C7.5 -S4053222918`

<img width="1081" alt="image" src="https://user-images.githubusercontent.com/50542132/191639011-f81d9d38-0a15-45f0-9442-a5e8d5c25f1f.png">

Actual bowl of fries.




