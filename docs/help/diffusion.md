Taking the time to understand the diffusion process will help you to understand how to more effectively use InvokeAI.

There are two main ways Stable Diffusion works - with images, and latents.

Image space represents images in pixel form that you look at. Latent space represents compressed inputs. It’s in latent space that Stable Diffusion processes images. A VAE (Variational Auto Encoder) is responsible for compressing and encoding inputs into latent space, as well as decoding outputs back into image space.

To fully understand the diffusion process, we need to understand a few more terms: UNet, CLIP, and conditioning.

A U-Net is a model trained on a large number of latent images with with known amounts of random noise added.  This means that the U-Net can be given a slightly noisy image and it will predict the pattern of noise needed to subtract from the image in order to recover the original. 

CLIP is a model that tokenizes and encodes text into conditioning. This conditioning guides the model during the denoising steps to produce a new image. 

The U-Net and CLIP work together during the image generation process at each denoising step, with the U-Net removing noise in such a way that the result is similar to images in the U-Net’s training set, while CLIP guides the U-Net towards creating images that are most similar to the prompt.


When you generate an image using text-to-image, multiple steps occur in latent space:
1. Random noise is generated at the chosen height and width. The noise’s characteristics are dictated by  seed. This noise tensor is passed into latent space. We’ll call this noise A.
2. Using a model’s U-Net, a noise predictor examines noise A, and the words tokenized by CLIP from your prompt (conditioning). It generates its own noise tensor to predict what the final image might look like in latent space. We’ll call this noise B.
3. Noise B is subtracted from noise A in an attempt to create a latent image consistent with the prompt. This step is repeated for the number of sampler steps chosen.
4. The VAE decodes the final latent image from latent space into image space.

Image-to-image is a similar process, with only step 1 being different:
1. The input image is encoded from image space into latent space by the VAE. Noise is then added to the input latent image. Denoising Strength dictates how may noise steps are added, and the amount of noise added at each step. A Denoising Strength of 0 means there are 0 steps and no noise added, resulting in an unchanged image, while a Denoising Strength of 1 results in the image being completely replaced with noise and a full set of denoising steps are performance. The process is then the same as steps 2-4 in the text-to-image process. 

Furthermore, a model provides the CLIP prompt tokenizer, the VAE, and a U-Net (where noise prediction occurs given a prompt and initial noise tensor).

A noise scheduler (eg. DPM++ 2M Karras) schedules the subtraction of noise from the latent image across the sampler steps chosen (step 3 above). Less noise is usually subtracted at higher sampler steps. 
