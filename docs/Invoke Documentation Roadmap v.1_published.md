---
Class: ai
Topic: Invoke Doc
Subject: Document Roadmap
Type: published
---

# Documentation Roadmap for InvokeAI

Document Version: 2024-JULY-01
Author: smile4yourself

The purpose of this document is to make it easier to find information about InvokeAI, especially for new users trying to get it to work and understand how to use it.

It started out simply as a note to help me get my mind around InvokeAI, to install it, get it to work, and then learn how to use it. It includes every link to Invoke documents that I have found, but re-organized under topics that are relevant to a new user like me. 
- Some links might appear in two places when the web page covered more than the single topic.
- Many pages have other links buried within their content, and these are listed as bullets below the link
- Information is often similar on different web pages, and these are listed below by topic. (e.g "Getting Started")

After making this note for myself, I realized that it could also be helpful to others find the information they need, or to re-work the published documentation pages on the web. 

Feel free to copy this document and use it as you wish. 

smile4yourself  


![[Divider_2.svg]]


# Links from Invoke on Github 

[InvokeAI](https://github.com/invoke-ai)

- [Invoke-ai Repository on Github](https://github.com/invoke-ai/InvokeAI)
## About Invoke, the people, the project

[about the Invoke Project](https://www.invoke.com/about)
[the-open-model-initiative](https://www.invoke.com/the-open-model-initiative)
[comfyui-vs-invokeai](https://www.invoke.com/comparisons/comfyui-vs-invokeai)
[invoke-ai-enterprise-alternative-for-midjourney](https://www.invoke.com/comparisons/invoke-ai-enterprise-alternative-for-midjourney)

[support/home](https://support.invoke.ai/support/home) (For Subscription Users)
- [search window!](https://support.invoke.ai/support/home) At Top of Page; only searches within the Support Portal
- [Browse articles](https://support.invoke.ai/support/solutions) Knowledge Base
- [Submit a Ticket](https://support.invoke.ai/support/tickets/new) For Subscription Users
- [Using Invoke (9)](https://support.invoke.ai/support/solutions/151000159037)
- [Training (1)](https://support.invoke.ai/support/solutions/151000219061) Training Models for Subscription Users

# Invoke Blog 

[Invoke blog/main](https://www.invoke.com/blog/main)

- [why-game-studios-are-training-their-own-generative-ai-models](https://www.invoke.com/post/why-game-studios-are-training-their-own-generative-ai-models)
- [control-layers-a-powerful-ui-for-applying-regional-prompts](https://www.invoke.com/post/introducing-control-layers-a-powerful-ui-for-applying-regional-prompts)
- [invoke-workflows-taking-enterprise-ai-game-development-next-level](https://www.invoke.com/post/introducing-invoke-workflows-taking-enterprise-ai-game-development-next-level)
- [how-to-build-a-custom-ai-image-model-trained-on-your-companys-content](https://www.invoke.com/post/how-to-build-a-custom-ai-image-model-trained-on-your-companys-content)
- [invoke-enterprise-to-help-companies-with-sensitive-ip-deploy-generative-ai](https://www.invoke.com/post/invoke-introduces-invoke-enterprise-to-help-companies-with-sensitive-ip-deploy-generative-ai)
- [artists-generative-ai-evolving-landscape-copyright](https://www.invoke.com/post/artists-generative-ai-evolving-landscape-copyright)
- [invoke-ai-raises-3-75-million-seed-funding](https://www.invoke.com/post/invoke-ai-raises-3-75-million-seed-funding)
- [how-to-choose-ai-image-generator-for-your-business](https://www.invoke.com/post/how-to-choose-ai-image-generator-for-your-business)


# Installation of invokeAI

[Releases/Versions](https://github.com/invoke-ai/InvokeAI/releases)
[Release Notes ](https://support.invoke.ai/support/solutions/articles/151000178246-what-s-new-invoke-release-notes) (from Support Portal for Subscription Users)

[🟦 Installation](https://invoke-ai.github.io/InvokeAI/installation/INSTALLATION/) -this is an Overview of installing and updating Invoke ✔︎

[INSTALL_REQUIREMENTS/](https://invoke-ai.github.io/InvokeAI/installation/INSTALL_REQUIREMENTS/) ✔︎  Mixed with installation information for specific platforms
[Automatic Install & Updates](https://invoke-ai.github.io/InvokeAI/installation/010_INSTALL_AUTOMATED/) Install Script
[Manual Installation](https://invoke-ai.github.io/InvokeAI/installation/020_INSTALL_MANUAL/)
[Docker Installation](https://invoke-ai.github.io/InvokeAI/installation/040_INSTALL_DOCKER/) Linux and Windows, not MAC
[PATCHMATCH/](https://invoke-ai.github.io/InvokeAI/installation/060_INSTALL_PATCHMATCH/)✔︎ MAC information needs updating, required for inpainting on MAC

[🟦 FAQ/](https://invoke-ai.github.io/InvokeAI/help/FAQ/)✔︎ Re. Installation Issues

[Code and Downloads](https://github.com/invoke-ai/InvokeAI/)

# Configuration of InvokeAI

[Controlling Logging](https://invoke-ai.github.io/InvokeAI/features/LOGGING/)
[Invoke Runtime Configuration](https://invoke-ai.github.io/InvokeAI/features/CONFIGURATION/)
[Image Database](https://invoke-ai.github.io/InvokeAI/features/DATABASE/)

[Command Line Utilities](https://invoke-ai.github.io/InvokeAI/features/UTILITIES/)
These use the Developer's console to: 
- start invoke, set ram, 
- do textual inversion training, 
- install, list, or delete models, 
- configure a configuration that is broken

![[Divider_2.svg]]


# Using InvokeAI

[🟦 Getting Started With AI Image Generation](https://invoke-ai.github.io/InvokeAI/help/gettingStartedWithAI/) 
(AI Generation, Prompts, 3 Tool Sets: Text-to-Image, Image-to-Image, Unified Canvas)
- [Lean more here](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#prompt-syntax-features)
- [Docs if you want the learn more](https://invoke-ai.github.io/InvokeAI/features/)
- [how ai works presentation](https://docs.google.com/presentation/d/1IO78i8oEXFTZ5peuHHYkVF-Y3e2M6iM5tCnc-YBfcCM/edit#slide=id.g230a2ec44e1_2_81)
- [Simple way to download models](https://models.invoke.ai/)


[🟦 Overview 2 Documentation](https://invoke-ai.github.io/InvokeAI/features/)
(Getting Started, The Web UI, Unified Canvas, Prompt Engineering,LoRAs, ControlNet, Image-to-Image, Model Installation, Merging Models, Textual Inversion, NSFW, Logging, Command-line Utilities)

- [Getting Started Guide](https://invoke-ai.github.io/InvokeAI/help/gettingStartedWithAI/) 
- [The Basics -web UI](https://invoke-ai.github.io/InvokeAI/features/WEB/)
- [Unified Canvas](https://invoke-ai.github.io/InvokeAI/features/UNIFIED_CANVAS/)
- [Prompt Engineering](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/)
- [The LoRA,LyCORIS, LCM-LoRA Models](dead link) Dead Link
- [ControlNet](https://invoke-ai.github.io/InvokeAI/features/CONTROLNET/)
- [Image-to-Image Guide](https://invoke-ai.github.io/InvokeAI/features/CONTROLNET/)
- [Model Installation](https://invoke-ai.github.io/InvokeAI/installation/050_INSTALLING_MODELS/)
- [Merging Models](https://invoke-ai.github.io/InvokeAI/features/MODEL_MERGING/)
- [Textual Inversion files](https://invoke-ai.github.io/InvokeAI/features/TEXTUAL_INVERSIONS/)
- [NSFW Checker](https://invoke-ai.github.io/InvokeAI/features/WATERMARK%2BNSFW/)
- [Controlling Logging](https://invoke-ai.github.io/InvokeAI/features/LOGGING/)
- [Command-line Utilities](https://invoke-ai.github.io/InvokeAI/features/UTILITIES/)


[Overview 3 "Features"](https://invoke-ai.github.io/InvokeAI/#invokeai-features)
(Installation, Web Interface, Image Management, Model Management, Prompt Engineering, Invoke Config, Troubleshooting, Contributors, Support)

[Getting Started with Invoke](https://support.invoke.ai/support/solutions/articles/151000096602-getting-started-with-invoke) (from invoke Support Portal) [📺 Youtube Video Included](https://www.youtube.com/watch?v=GCt_tr-TAQw)


[User Interface Overview](https://support.invoke.ai/support/solutions/articles/151000170670-user-interface-overview) (WebUI from Support Portal)

[WebUI Interface Walkthrough](https://invoke-ai.github.io/InvokeAI/features/WEB/)
[Queue](https://support.invoke.ai/support/solutions/articles/151000158823-queue)


## Hot Keys

[WEBUI HOTKEYS/](https://invoke-ai.github.io/InvokeAI/features/WEBUIHOTKEYS/)
[WebUI Unified Canvas for Img2Img, inpainting and outpainting](https://invoke-ai.github.io/InvokeAI/features/UNIFIED_CANVAS/)

[Other Features/Invisible Watermark](https://invoke-ai.github.io/InvokeAI/features/OTHER/)


## How to write Prompts for InvokeAI

[Prompt Syntax](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/)
[Other Features/ Weighted Prompts](https://invoke-ai.github.io/InvokeAI/features/OTHER/)
[Prompting Features](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/) 
♥️[[Invoke Style Prompt Book]]

[Tips on Crafting Prompts](https://support.invoke.ai/support/solutions/articles/151000096606-tips-on-crafting-prompts) (from Invoke Support Portal)

![[Divider_2.svg]]

## How to Create Images (Image Management)

## Basics

[What is a Seed?](https://support.invoke.ai/support/solutions/articles/151000096684-what-is-a-seed-how-do-i-use-it-to-recreate-the-same-image-) (from Support Portal)
[What is Upscaling](https://support.invoke.ai/support/solutions/articles/151000096700-how-can-i-get-larger-images-what-does-upscaling-do-) (from Support Portal)
[What is Inpainting / Outpainting?](https://support.invoke.ai/support/solutions/articles/151000096702-what-are-inpainting-and-outpainting-) (from Support Portal)
[5-What are Control Adapters?](https://support.invoke.ai/support/solutions/articles/151000178148-what-are-control-adapters-) (from Support Portal) Three types, Settings, Weight, Begin/End
[Control Adapters -What are they?](https://invoke-ai.github.io/InvokeAI/features/CONTROLNET/)
[What are Low-Rank Adaptations (LoRAs)?](https://support.invoke.ai/support/solutions/articles/151000159072-concepts-low-rank-adaptations-loras-) (from Support Portal)

[Style/Subject Concepts and Embeddings](https://invoke-ai.github.io/InvokeAI/features/CONCEPTS.md)

[Image2Image](https://invoke-ai.github.io/InvokeAI/features/IMG2IMG/)

[Image to Image](https://support.invoke.ai/support/solutions/articles/151000094998-image-to-image) (from Invoke Support Portal)

[Adding custom styles and subjects](https://invoke-ai.github.io/InvokeAI/features/CONCEPTS.md) 
[WebUI walkthrough](https://invoke-ai.github.io/InvokeAI/features/WEB/)
[Improving Image Generations](https://support.invoke.ai/support/solutions/articles/151000157387-improving-image-generations) (from Support Portal)


**Control Layers (5) (from Support Portal)**
[1-Control Layers and Adapters Overview](https://support.invoke.ai/support/solutions/articles/151000189477-control-layers) (from Support Portal)
[2-Using ControlNet features to improve results](https://support.invoke.ai/support/solutions/articles/151000105880-using-controlnet)  (from Support Portal)
[3-Using the Image to Image Image Prompt Adapter (IP_Adapter)](https://support.invoke.ai/support/solutions/articles/151000159340-using-the-image-prompt-adapter-ip-adapter-)  (from Support Portal)
[4-Using Text to Image Adapter(T2I-Adapter)](https://support.invoke.ai/support/solutions/articles/151000165024-using-text-to-image-adapter-t2i-adapter-)  (from Support Portal)
[5-What are Control Adapters?](https://support.invoke.ai/support/solutions/articles/151000178148-what-are-control-adapters-) (from Support Portal) Three types, Settings, Weight, Begin/End


**Unified Canvas (4) (from Support Portal)**
[3-What is the Unified Canvas? When to use it?](https://support.invoke.ai/support/solutions/articles/151000096682-what-is-the-unified-canvas-and-when-should-i-use-it-)
[1-Unified Canvas -Composing Settings](https://support.invoke.ai/support/solutions/articles/151000158838-compositing-settings)
[2-Infill Settings](https://support.invoke.ai/support/solutions/articles/151000158841-infill-settings)
[4-Scale before Processing](https://support.invoke.ai/support/solutions/articles/151000179777-scale-before-processing) 

[Postprocessing -Fixing Faces](https://invoke-ai.github.io/InvokeAI/features/POSTPROCESS/)

**Workflows and Nodes**

[What are Nodes?](https://invoke-ai.github.io/InvokeAI/nodes/overview)
Workflow Editor(4)
[1-the Workflow Editor](https://support.invoke.ai/support/solutions/articles/151000159646-workflow-editor) (from Support Portal)
[2-Important Nodes Usage](https://support.invoke.ai/support/solutions/articles/151000159649-important-nodes-usage)(from Support Portal)
[3-Example Workflows](https://support.invoke.ai/support/solutions/articles/151000159663-example-workflows)(from Support Portal)
[4-Getting Started with Worlflows](https://support.invoke.ai/support/solutions/articles/151000189610-getting-started-with-workflows-denoise-latents)(from Support Portal)


![[Divider_2.svg]]

## Model Management

[What is an AI model? Which ones does InvokeAI Support?](https://support.invoke.ai/support/solutions/articles/151000096601-what-is-a-model-which-should-i-use-) (from Support Portal)

[Installing Models](https://invoke-ai.github.io/InvokeAI/installation/050_INSTALLING_MODELS/)
[Simple way to download models](https://models.invoke.ai/)
[Invoke offers an easy way to download models](https://models.invoke.ai/)

[Model Merging](https://invoke-ai.github.io/InvokeAI/features/MODEL_MERGING/)

Models (3) from Support Portal
[1-Uploading AI models and LoRAs](https://support.invoke.ai/support/solutions/articles/151000170960-uploading-models-and-loras) (from Support Portal)
[2-Supported Models](https://support.invoke.ai/support/solutions/articles/151000170961-supported-models)
[3-Understanding Models & Concepts](https://support.invoke.ai/support/solutions/articles/151000177848-understanding-models-concepts)

[Control Adapters and ControlNet Models](https://invoke-ai.github.io/InvokeAI/features/CONTROLNET/)
[LoRAs and LCM-LoRSa/](https://invoke-ai.github.io/InvokeAI/features/LORAS/)

![[Divider_2.svg]]

# Contribution and Support

[🟦 Contributing](https://invoke-ai.github.io/InvokeAI/contributing/CONTRIBUTING/)✔︎

[Contributing](https://invoke-ai.github.io/InvokeAI/contributing/CONTRIBUTING/) Link from Top Web Menu and Home Page Quicklinks
[Contributors](https://invoke-ai.github.io/InvokeAI/other/CONTRIBUTORS/)✔︎

[🟦 Bug Reports ](https://github.com/invoke-ai/InvokeAI/issues) Link to Github✔︎

[🟦 Join the Discord Server](https://discord.com/invite/ZmtBAhwWhy)✔︎


Advanced Features (6) from Support Portal

[1-keyboard Shortcuts](https://support.invoke.ai/support/solutions/articles/151000170408-keyboard-shortcuts)
[2-Dynamic Prompting](https://support.invoke.ai/support/solutions/articles/151000158854-dynamic-prompting)
[3-Unified Canvas Keyboard Shortcuts](https://support.invoke.ai/support/solutions/articles/151000170476-unified-canvas-keyboard-shortcuts)
[4-Advanced Prompting Syntax](https://support.invoke.ai/support/solutions/articles/151000096723-advanced-prompting-syntax)
[5-Advanced Settings](https://support.invoke.ai/support/solutions/articles/151000178161-advanced-settings) VAE, VAE Precision, Clip Skip, CFG Rescale Multipler, Seamless Tiling X Axis, Y Axis
[6-Using the Refiner](https://support.invoke.ai/support/solutions/articles/151000178333-using-the-refiner)


![[Divider_2.svg]]

# Tutorials 

[How Text to Image Works by Lincoln Stein](https://docs.google.com/presentation/d/1IO78i8oEXFTZ5peuHHYkVF-Y3e2M6iM5tCnc-YBfcCM/edit#slide=id.g230a2ec44e1_2_81)

[📺 Invoke YouTube Channel](https://www.youtube.com/@invokeai)
New Videos are posted here.

[📺 Invoke 3.0 Demo](https://www.youtube.com/watch?v=1Iz4F7o6hgQ&t=8s)
Model Manager, 
[📺 Unified Canvas Basics](https://www.youtube.com/watch?v=s4EqQRxRR7k&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM)
[📺 Unified Canvas Drive through -1](https://www.youtube.com/watch?v=QSmQ_19rszU&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=2)
[Unified Canvas Drive through -2](https://www.youtube.com/watch?v=GAlaOlihZ20&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=3)

[📺 Advanced Canvas Inpainting Techniques](https://www.youtube.com/watch?v=-xzvIpfGTXg&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=9)

[📺 Unified Canvas Tips and Tricks -1](https://www.youtube.com/watch?v=2pcBtNkTZ40&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=4)
1. Create a white background and draw a character.
2. Use the [Character turner Lora](https://civitai.com/models/3036/charturner-character-turnaround-helper-for-15-and-21)Using a pre-defined image templates of a character looking 
Try out different settings but keep the seed and prompt to see what the different in three different directions. 
settings will do. 
3. Canvas can be used to explore images and 
thinking. Paint squares of different colours and generate on top.

InvokeAI Training Series
[📺 Getting Started -AI Image Basics](https://www.youtube.com/watch?v=GCt_tr-TAQw)
[📺 SDXL - Getting Started](https://www.youtube.com/watch?v=c7zAhlC7xKE&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=5)
[📺 Fundamentals -Creating with AI](https://www.youtube.com/watch?v=m8hnpX2XzL0&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=6) Creating with AI Fundamentals
[📺 Workflow Fundamentals](https://www.youtube.com/watch?v=L3isi26qy0Y&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=7)
[📺 Invoke Canvas Fundamentals](https://www.youtube.com/watch?v=2pcBtNkTZ40&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=4)
[📺 AI Image Prompting](https://www.youtube.com/watch?v=WpPVf_XepIg&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=10)
[📺 Creating Embeddings and Textual Inversions](https://www.youtube.com/watch?v=OZIz2vvtlM4&list=PLvWK1Kc8iXGpxQgr-Z89JEY_kM8iEgguM&index=11)

OTHER TUTORIALS

[📺 Good Overview Video of Stable Diffusion Image Making -Good for Beginners!](https://www.youtube.com/watch?v=OeqpwYLMaN0) by Monzon Media
[📺  Introduction at InvokeAI](https://www.youtube.com/watch?v=GKjifMhcvOk&list=PLwz5mNb2fSE8on03OGraM1t-No-HBT30e)By Oliio Sarikas
[📺 InvokeAI 3.0 Walkthrough -by Monzon Media](https://www.youtube.com/watch?v=1Iz4F7o6hgQ)

# Subscription Available As Well

[Subscriptions](https://support.invoke.ai/support/solutions/folders/151000398898)
[Account Management for Subscriptions](https://support.invoke.ai/support/solutions/folders/151000416969)

[How does my Generation Usage Work?](https://support.invoke.ai/support/solutions/articles/151000096665-how-does-my-generation-usage-work-) (From Support Portal)
[Sharing and Collaboration Features](https://support.invoke.ai/support/solutions/articles/151000170652-sharing-and-collaboration-features)
[Creating and Managing Boards](https://support.invoke.ai/support/solutions/articles/151000170653-creating-and-managing-boards)
[Creating a new Project](https://support.invoke.ai/support/solutions/articles/151000170816-creating-a-new-project)
[Get Started with Invoke Training](https://support.invoke.ai/support/solutions/articles/151000189290-getting-started-with-invoke-training)

[[knowledge Base]]

