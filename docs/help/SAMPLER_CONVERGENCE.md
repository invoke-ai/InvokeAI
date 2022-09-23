---
title: SAMPLER CONVERGENCE
---

## **Sampler Convergence**

As features keep increasing, making the right choices for your needs can become increasingly difficult. What sampler to use? And for how many steps? Do you change the CFG value? Do you use prompt weighting? Do you allow variations?

Even once you have a result, do you blend it with other images? Pass it through `img2img`? With what strength? Do you use inpainting to correct small details? Outpainting to extend cropped sections?

The purpose of this series of documents is to help you better understand these tools, so you can make the best out of them. Feel free to contribute with your own findings!

In this document, we will talk about sampler convergence.

Looking for a short version? Here's a TL;DR in 4 tables.

| Remember  |
|:---|
| Results tend to converge as steps (`-s`) are increased (except for `K_DPM_2_A` and `K_EULER_A`).  |
| Producing a batch of candidate images at low step counts can save you hours of computation.  |
| `K_HEUN` and `K_DPM_2`  converge in less steps (but are slower).  |
| `K_DPM_2_A` and `K_EULER_A` incorporate a lot of creativity/variability. |

| Sampler   | (3 sample avg) it/s (M1 Max 64GB, 512x512)  |
|---|---|
|  `DDIM` | 1.89  |
|  `PLMS` | 1.86  |
|  `K_EULER` | 1.86  |
|  `K_LMS` | 1.91  |
|  `K_HEUN` | 0.95 (slower)  |
|  `K_DPM_2` | 0.95 (slower)  |
|  `K_DPM_2_A` | 0.95 (slower)  |
|  `K_EULER_A` | 1.86  |

| Suggestions  |
|:---|
| For most use cases, `K_LMS` is a good choice, on par with `K_HEUN` and `K_DPM_2` (both run 0.5x as quick, converge 2x as quick as `K_LMS`).  |
| For variability, use `K_EULER_A` (runs 2x as quick as `K_DPM_2_A`).  |
| For people, use `K_HEUN` (converges quicker than `K_DPM_2` and runs 0.5x as quick but converges 4-10x as quick as others)  |

| Topic   | K_HEUN/K_DPM_2 steps to conv.  |
|:---|:---|
|  Nature |   |
|  Faces/bodies | Not guaranteed (but more steps increase coherence)  |
|  Food |   |
---

### **Sampler results**

Let's start by choosing a prompt and using it with each of our 8 samplers, running it for 10, 20, 30, 40, 50 and 100 steps.

Anime. `"an anime girl" -W512 -H512 -C7.5 -S3031912972`

![191636411-083c8282-6ed1-4f78-9273-ee87c0a0f1b6-min (1)](https://user-images.githubusercontent.com/50542132/191868725-7f7af991-e254-4c1f-83e7-bed8c9b2d34f.png)

### **Sampler convergence**

Immediately, you can notice results tend to converge -that is, results with low `-s` (step) values are often indicative of results with high `-s` values.

You can also notice how `DDIM` and `PLMS` eventually tend to converge to K-sampler results as steps are increased.
Among K-samplers, `K_HEUN` and `K_DPM_2` seem to require the fewest steps to converge. And finally, `K_DPM_2_A` and `K_EULER_A` seem to do a bit of their own thing and don't keep much similarity with the rest of the samplers.

### **Batch generation speedup**

This realization is very useful because it means you don't need to create a batch of 100 images (`-n100`) at `-s100` to choose your favorite 2 or 3 images.
You can produce the same 100 images at `-s10` to `-s30` (more on that later) using a K-sampler (since they converge faster), get a rough idea of the final result, choose your 2 or 3 favorite ones, and then run `-s100` on those images to polish some details.
The latter technique is 3-8x as quick.

Example:

At 60s per 100 steps.

(Option A) 60s * 100 images = 6000s (100 images at `-s100`, manually picking 3 favorites)

(Option B) 6s * 100 images + 60s * 3 images = 780s (100 images at `-s10`, manually picking 3 favorites, and running those 3 at `-s100` to polish details)

The result is 1 hour and 40 minutes (Option A) vs 13 minutes (Option B).

### **Topic convergance**

Now, these results seem interesting, but do they hold for other topics? How about nature? Food? People? Animals? Let's try!

Nature. `"valley landscape wallpaper, d&d art, fantasy, painted, 4k, high detail, sharp focus, washed colors, elaborate excellent painted illustration" -W512 -H512 -C7.5 -S1458228930`

![191736091-dda76929-00d1-4590-bef4-7314ea4ea419-min (1)](https://user-images.githubusercontent.com/50542132/191868763-b151c69e-0a72-4cf1-a151-5a64edd0c93e.png)

With nature, you can see how results tend to converge faster than with characters/people. `K_HEUN` and `K_DPM_2` are again the quickest to converge, almost right from the start.

Food. `"a hamburger with a bowl of french fries" -W512 -H512 -C7.5 -S4053222918`

![191639011-f81d9d38-0a15-45f0-9442-a5e8d5c25f1f-min (1)](https://user-images.githubusercontent.com/50542132/191868898-98801a62-885f-4ea1-aee8-563503522aa9.png)

Again, we see `K_HEUN` and `K_DPM_2` tend to converge in the fewest number of steps towards the final result. `K_DPM_2`_A and `K_EULER_A` seem to incorporate a lot of creativity/variability, capable of producing rotten hamburgers, but also of adding lettuce to the mix. And they're the only samplers that produced an actual 'bowl of fries'!

Animals. `"grown tiger, full body" -W512 -H512 -C7.5 -S3721629802`

![191771922-6029a4f5-f707-4684-9011-c6f96e25fe56-min (1)](https://user-images.githubusercontent.com/50542132/191868870-9e3b7d82-b909-429f-893a-13f6ec343454.png)

`K_HEUN` and `K_DPM_2` once again are the quickest step-wise -converging around `-s30`, while other samplers are still struggling with several tails or malformed back legs.

However, you can see how in general it takes longer to converge (for comparison, `K_HEUN` often converges between `-s10` and `-s20` for nature and food compositions, which means this took an extra 10 to 20 steps to converge). This is normal, as producing human/animal faces/bodies is one of the things the model struggles the most with. For these topics, running for more steps will often increase coherence within the composition.

People. `"Ultra realistic photo, (Miranda Bloom-Kerr), young, stunning model, blue eyes, blond hair, beautiful face, intricate, highly detailed, smooth, art by artgerm and greg rutkowski and alphonse mucha, stained glass" -W512 -H512 -C7.5 -S2131956332`

![Screenshot 2022-09-23 at 02 05 48-min (1)](https://user-images.githubusercontent.com/50542132/191871743-6802f199-0ffd-4986-98c5-df2d8db30d18.png)

### **Sampler generation times**

| Sampler   | (3 sample avg) it/s (M1 Max 64GB, 512x512)  |
|---|---|
|  `DDIM` | 1.89  |
|  `PLMS` | 1.86  |
|  `K_EULER` | 1.86  |
|  `K_LMS` | 1.91  |
|  `K_HEUN` | 0.95 (slower)  |
|  `K_DPM_2` | 0.95 (slower)  |
|  `K_DPM_2_A` | 0.95 (slower)  |
|  `K_EULER_A` | 1.86  |



Specific step
Anime -
Burger - 
Tiger -
