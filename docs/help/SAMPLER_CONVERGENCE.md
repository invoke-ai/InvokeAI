---
title: SAMPLER CONVERGENCE
---

## **Sampler Convergence**

As features keep increasing, making the right choices for your needs can become increasingly difficult. What sampler to use? And for how many steps? Do you change the CFG value? Do you use prompt weighting? Do you allow variations?

Even once you have a result, do you blend it with other images? Pass it through `img2img`? With what strength? Do you use inpainting to correct small details? Outpainting to extend cropped sections?

The purpose of this series of documents is to help you better understand these tools, so you can make the best out of them. Feel free to contribute with your own findings!

In this document, we will talk about sampler convergence.

| Tip   | Desc.  |
|---|---|
|  1 | Results tend to converge as steps (`-s`) are increased.  |
|  2 | Producing a batch of candidate images at low step counts can save a lot of time.  |
|  3 | K_HEUN and K_DPM_2 tend to converge in less steps (but are slower)  |
|  4 | K_DPM_2_A and K_EULER_A incorporate a lot of creativity/variability.  |

Adjusting convergence for it/s ->


| Topic   | K_HEUN/K_DPM_2 steps to conv.  |
|---|---|
|  Nature |   |
|  Faces/bodies | (more steps increase coherence)  |
|  Food |   |
---

### **Sampler results**

Let's start by choosing a prompt and using it with each of our 8 samplers, running it for 10, 20, 30, 40, 50 and 100 steps.

Anime. `"an anime girl" -W512 -H512 -C7.5 -S3031912972`
<img width="1082" alt="image" src="https://user-images.githubusercontent.com/50542132/191636411-083c8282-6ed1-4f78-9273-ee87c0a0f1b6.png">

### **Sampler convergence**

Immediately, you can notice results tend to converge -that is, results with low `-s` (step) values are often indicative of results with high `-s` values.

You can also notice how DDIM and PLMS eventually tend to converge to K-sampler results as steps are increased.
Among K-samplers, K_HEUN and K_DPM_2 seem to be the quickest to converge. And finally, K_DPM_2_A and K_EULER_A seem to do a bit of their own thing and don't keep much similarity with the rest of the samplers.

### **Batch generation speedup**

This realization is very useful because it means you don't need to create a batch of 100 images (`-n100`) at `-s100` to choose your favorite 2 or 3 images.
You can produce the same 100 images at `-s10` to `-s30` (more on that later) using a K-sampler (since they converge faster), get a rough idea of the final result, choose your 2 or 3 favorite ones, and then run `-s100` on those images to polish some details.
The latter technique is 3x to 8x faster.

Example:

At 60s per 100 steps:

(Option A) 60s * 100 images = 6000s

(Option B) 6s * 100 images + 60s * 3 images = 780s

### **Topic convergance**

Now, these results seem interesting, but do they hold for other topics? How about nature? Food? People? Animals? Let's try!

Nature. `"valley landscape wallpaper, d&d art, fantasy, painted, 4k, high detail, sharp focus, washed colors, elaborate excellent painted illustration" -W512 -H512 -C7.5 -S1458228930`

<img width="1082" alt="191617318-40e08e67-d147-4768-b27c-349844d10461 copy" src="https://user-images.githubusercontent.com/50542132/191736091-dda76929-00d1-4590-bef4-7314ea4ea419.png">

With nature, you can see how results tend to converge faster than with characters/people. K_HEUN and K_DPM_2 are again the fastest.

Food. `"a hamburger with a bowl of french fries" -W512 -H512 -C7.5 -S4053222918`

<img width="1081" alt="image" src="https://user-images.githubusercontent.com/50542132/191639011-f81d9d38-0a15-45f0-9442-a5e8d5c25f1f.png">

Again, we see K_HEUN and K_DPM_2 tend to converge faster towards the final result. K_DPM_2_A and K_EULER_A seem to incorporate a lot of creativity/variability, capable of producing rotten hamburgers, but also of adding lettuce to the mix. And they're the only samplers that produced an actual 'bowl of fries'!

Animals. `"grown tiger, full body" -W512 -H512 -C7.5 -S3721629802`

<img width="1081" alt="image" src="https://user-images.githubusercontent.com/50542132/191771922-6029a4f5-f707-4684-9011-c6f96e25fe56.png">

K_HEUN and K_DPM_2 once again are the quickest -converging around `-s30`, while other samplers are still struggling with several tails or malformed back legs.

However, you can see how in general it takes longer to converge (for comparison, K_HEUN often converges between `-s10` and `-s20` for nature and food compositions, which means this took an extra 10 to 20 steps to converge). This is normal, as producing faces/bodies (whether it is of animals or people) is one of the things the model struggles the most with. For these topics, running for more steps will often increase coherence within the composition.

### **Sampler generation times**

| Sampler   | it/s  |
|---|---|
|  DDIM |   |
|  PLMS |   |
|  K_EULER |   |
|  K_LMS |   |
|  K_HEUN |   |
|  K_DPM_2 |   |
|  K_DPM_2_A |   |
|  K_EULER_A |   |
