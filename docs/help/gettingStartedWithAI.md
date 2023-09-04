# Getting Started with AI Image Generation

New to image generation with AI? You’re in the right place! 

This is a high level walkthrough of some of the concepts and terms you’ll see as you start using InvokeAI. Please note, this is not an exhaustive guide and may be out of date due to the rapidly changing nature of the space. 

## Using InvokeAI

### **Prompt Crafting**

- Prompts are the basis of using InvokeAI, providing the models directions on what to generate. As a general rule of thumb, the more detailed your prompt is, the better your result will be.

 *To get started, here’s an easy template to use for structuring your prompts:*

- Subject, Style, Quality, Aesthetic
    - **Subject:** What your image will be about. E.g. “a futuristic city with trains”,  “penguins floating on icebergs”, “friends sharing beers”
    - **Style:** The style or medium in which your image will be in. E.g. “photograph”, “pencil sketch”, “oil paints”, or “pop art”, “cubism”, “abstract”
    - **Quality:** A particular aspect or trait that you would like to see emphasized in your image. E.g. "award-winning", "featured in {relevant set of high quality works}", "professionally acclaimed". Many people often use "masterpiece".
    - **Aesthetics:** The visual impact and design of the artwork. This can be colors, mood, lighting, setting, etc.
- There are two prompt boxes: *Positive Prompt* & *Negative Prompt*.
    - A **Positive** Prompt includes words you want the model to reference when creating an image.
    - Negative Prompt is for anything you want the model to eliminate when creating an image. It doesn’t always interpret things exactly the way you would, but helps control the generation process. Always try to include a few terms - you can typically use lower quality image terms like “blurry” or “distorted” with good success.
- Some examples prompts you can try on your own:
    - A detailed oil painting of a tranquil forest at sunset with vibrant+ colors and soft, golden light filtering through the trees
    - friends sharing beers in a busy city, realistic colored pencil sketch, twilight, masterpiece, bright, lively

### Generation Workflows

- Invoke offers a number of different workflows for interacting with models to produce images. Each is extremely powerful on its own, but together provide you an unparalleled way of producing high quality creative outputs that align with your vision.
    - **Text to Image:** The text to image tab focuses on the key workflow of using a prompt to generate a new image. It includes other features that help control the generation process as well.
    - **Image to Image:** With image to image, you provide an image as a reference (called the “initial image”), which provides more guidance around color and structure to the AI as it generates a new image. This is provided alongside the same features as Text to Image.
    - **Unified Canvas:** The Unified Canvas is an advanced AI-first image editing tool that is easy to use, but hard to master. Drag an image onto the canvas from your gallery in order to regenerate certain elements, edit content or colors (known as inpainting), or extend the image with an exceptional degree of consistency and clarity (called outpainting).

### Improving Image Quality

- Fine tuning your prompt - the more specific you are, the closer the image will turn out to what is in your head!  Adding more details in the Positive Prompt or Negative Prompt can help add / remove pieces of your image to improve it - You can also use advanced techniques like upweighting and downweighting to control the influence of certain words. [Learn more here](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#prompt-syntax-features).
    - **Tip: If you’re seeing poor results, try adding the things you don’t like about the image to your negative prompt may help. E.g. distorted, low quality, unrealistic, etc.**
- Explore different models - Other models can produce different results due to the data they’ve been trained on. Each model has specific language and settings it works best with; a model’s documentation is your friend here.  Play around with some and see what works best for you!
- Increasing Steps - The number of steps used controls how much time the model is given to produce an image, and depends on the “Scheduler” used. The schedule controls how each step is processed by the model. More steps tends to mean better results, but will take longer - We recommend at least 30 steps for most
- Tweak and Iterate - Remember, it’s best to change one thing at a time so you know what is working and what isn't. Sometimes you just need to try a new image, and other times using a new prompt might be the ticket. For testing, consider turning off the “random” Seed - Using the same seed with the same settings will produce the same image, which makes it the perfect way to learn exactly what your changes are doing.
- Explore Advanced Settings - InvokeAI has a full suite of tools available to allow you complete control over your image creation process - Check out our [docs if you want to learn more](https://invoke-ai.github.io/InvokeAI/features/).


## Terms & Concepts

If you're interested in learning more, check out [this presentation](https://docs.google.com/presentation/d/1IO78i8oEXFTZ5peuHHYkVF-Y3e2M6iM5tCnc-YBfcCM/edit?usp=sharing) from one of our maintainers (@lstein). 

### Stable Diffusion

Stable Diffusion is deep learning, text-to-image model that is the foundation of the capabilities found in InvokeAI. Since the release of Stable Diffusion, there have been many subsequent models created based on Stable Diffusion that are designed to generate specific types of images. 

### Prompts

Prompts provide the models directions on what to generate. As a general rule of thumb, the more detailed your prompt is, the better your result will be.

### Models

Models are the magic that power InvokeAI. These files represent the output of training a machine on understanding massive amounts of images - providing them with the capability to generate new images using just a text description of what you’d like to see. (Like Stable Diffusion!)

Invoke offers a simple way to download several different models upon installation, but many more can be discovered online, including at ****. Each model can produce a unique style of output, based on the images it was trained on - Try out different models to see which best fits your creative vision!

- *Models that contain “inpainting” in the name are designed for use with the inpainting feature of the Unified Canvas*

### Scheduler

Schedulers guide the process of removing noise (de-noising) from data. They determine:

1. The number of steps to take to remove the noise.
2. Whether the steps are random (stochastic) or predictable (deterministic).
3. The specific method (algorithm) used for de-noising.

Experimenting with different schedulers is recommended as each will produce different outputs!

### Steps

The number of de-noising steps each generation through. 

Schedulers can be intricate and there's often a balance to strike between how quickly they can de-noise data and how well they can do it. It's typically advised to experiment with different schedulers to see which one gives the best results. There has been a lot written on the internet about different schedulers, as well as exploring what the right level of "steps" are for each. You can save generation time by reducing the number of steps used, but you'll want to make sure that you are satisfied with the quality of images produced!

### Low-Rank Adaptations / LoRAs

Low-Rank Adaptations (LoRAs) are like a smaller, more focused version of models, intended to focus on training a better understanding of how a specific character, style, or concept looks.

### Textual Inversion Embeddings

Textual Inversion Embeddings, like LoRAs, assist with more easily prompting for certain characters, styles, or concepts. However, embeddings are trained to update the relationship between a specific word (known as the “trigger”) and the intended output. 

### ControlNet

ControlNets are neural network models that are able to extract key features from an existing image and use these features to guide the output of the image generation model. 

### VAE

Variational auto-encoder (VAE) is a encode/decode model that translates the "latents" image produced during the image generation procees to the large pixel images that we see. 

