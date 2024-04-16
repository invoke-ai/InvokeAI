---
title: InvokeAI YouTube Videos
---

# :material-web: InvokeAI YouTube Videos

## Living list of all InvokeAI YouTube Videos

The InvokeAI team has produced many detailed and informative videos on many of the ways InvokeAI works on the [InvokeAI YouTube channel](https://www.youtube.com/@invokeai). This page will be maintained with links and annotated descriptions of what those videos cover. This is a curated list and is organized, as much as possible, by theme not by release date, this can mean that some of the content may have changed in more-recent versions of Invoke. The Invoke team has also created curated [playlists](https://www.youtube.com/@invokeai/playlists) if you'd prefer to watch a series of videos one after the other.

## Topic Videos

Short-form videos that introduce parts of InvokeAI.


<table width="100%">
<tr><th colspan="2"><h2>AI Image Generation</h2></th></tr>
<tr>
  <td width="40%">
  <strong>The Basics of AI Image Generation (Invoke - Getting Started Series #1) (13:12)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=GCt_tr-TAQw" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/GCt_tr-TAQw/0.jpg"></a>
  </td>
  <td>
  overview of Invoke Studio interface, positive and negative prompts, Embeddings, image settings, generation settings (Model and Concepts), Controlnet, IP Adapter
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Fundamentals - Creating with AI (13:39)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=m8hnpX2XzL0" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/m8hnpX2XzL0/0.jpg"></a>
  </td>
  <td>
  how Diffusion image modelling works, how images are generated, how prompts are structured (Subject, Category, Quality Modifiers, Aesthetics & Composition), how Image2Image works (Denoising Strength), brief overview of Control Adapters (Denoising Timeline, Begin / End Step Percentage)
  </td>
</tr>


<tr>
  <td width="40%">
  <strong>InvokeAI - SDXL Getting Started (6:39)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=c7zAhlC7xKE" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/c7zAhlC7xKE/0.jpg"></a>
  </td>
  <td>
  SDXL prompting (Concatenate Prompt with Style button), VAE Precision, ideal image ratios, Refiner (refinger steps to denoising steps ratio), <a href="https://gist.github.com/keturn/beb33f4f71cf88aaa34ae9e59c5e719f">keturn SDXL Prompt Styles</a>
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>Using AI Image to Image Transformation (Invoke - Getting Started Series #3) (8:26)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=z4uT-tppfEc" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/z4uT-tppfEc/0.jpg"></a>
  </td>
  <td>
  review of denoising process in Image2Image, Denoising Strength, using high contrast images to control composition and placement
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - AI Image Prompting (25:34)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=WpPVf_XepIg" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/WpPVf_XepIg/0.jpg"></a>
  </td>
  <td>
  Controlnet Adapters (SDXLcanny), IP Adapters (ip_adapter_sdxl_vit_h, Weight influence, using result images as input), use of Controlnet Adapters and IP Adapters with Image2Image, Unified Canvas and Controlnet Adapters (SDXLsoftedge), Controlnet Adapters to imitate style
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>Using Schedulers and CFG Scale - Advanced Generation Settings (Invoke - Getting Started Series #4) (9:35)</strong>
  <br>
  <a href="Vhttps://www.youtube.com/watch?v=1OeHEJrsTpIIDEO" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/1OeHEJrsTpI/0.jpg"></a>
  </td>
  <td>
  DESusing advanced generation settings, brief explanation of sampler/scheduler, explanation of Steps and relationship to quality/efficiency, brief explanation of CFG Scale settingC
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>Controlling AI Image Generation with ControlNet & IP Adapter (Invoke - Getting Started Series #2) (13:57)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=7Q5PcxkbEjE" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/7Q5PcxkbEjE/0.jpg"></a>
  </td>
  <td>
  explanation of SDXLcanny ControlNet (Weight, Begin/End Step Percentage, Control Mode, Resize Mode, Processor, Low & High Threshold), explanation of using multiple ControlNets at the same time (SDXLDepth & SDXLcanny), explanation of IP Adapters (Weight, ip_adapter_sdxl vs ip-adapter-plus_sdxl_vit-h)
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Workflow Fundamentals - Creating with Generative AI (23:29)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=L3isi26qy0Y" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/L3isi26qy0Y/0.jpg"></a>
  </td>
  <td>
  deeper discussion of the denoising process (CLIP: Text Encoder, UNet: Model Weights, VAE: Image Decoder), overview of the Denoise Latents node (Positive Conditioning, Negative Conditioning, UNet, Noise, Denoising Start, Denoising End, Latents Object), overview of the Latents to Image node, overview of the Basic Workflow (Positive Prompt, Negative Prompot, Noise, Denoising, Decoding), demonstration of using the Workflow Editor (adding nodes, connecting nodes, renaming nodes and fields, adding nodes to the Linear View, Noise Node, Random Integer Node, minimizing nodes, selecting nodes to see their output, Image Primitive Node, Image to Latents Node, Progress Image Node, Resize Latents Node, errors and error indicators, adding workflow details for distribution, Notes Node)
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>AI Workflows that Accelerate Production & Respect Artists' Process (GDC 2024) (35:56)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=cdpnazNI4Ig" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/cdpnazNI4Ig/0.jpg"></a>
  </td>
  <td>
  presentation at the Game Developers Conference (2024) on AI image generation at time of recording, Invoke's philosophy and approach to AI image generation, how workflows work and generate images, demonstration of the Unified Canvas, Q&A from the audience
  </td>
</tr>
</table>

<br>

<table width="100%">
<tr><th colspan="2"><h2>Unified Canvas</h2></th></tr>
<tr>
<tr>
  <td width="40%">
  <strong>Creating and Composing on the Unified Canvas (Invoke - Getting Started Series #6) (21:23)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=6pxr9B3fl_Q" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/6pxr9B3fl_Q/0.jpg"></a>
  </td>
  <td>
  detailed exploration of using the Unified Canvas, hotkeys, settings, Inpainting and prompting to change images, Staging Area, Inpainting to correct small details, Scale Before Generating, starting from scratch on the Unified Canvas (Outpainting, Denoising Strength, Coherence Pass settings), Manual Infills (painting to the Base layer to extend images)
  </td>
</tr>

  <td width="40%">
  <strong>InvokeAI - Unified Canvas Basics (28:37)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=s4EqQRxRR7k" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/s4EqQRxRR7k/0.jpg"></a>
  </td>
  <td>
  discussions of Context, Inpainting (Base and Mask layers), Image Colouring, Outpainting (importantance of Strength, overview of Seam Correction settings, approaches & strategies for improving results), survey of the toolbar items and settings
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Fundamentals (38:06)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=kzRL88ffv1o" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/kzRL88ffv1o/0.jpg"></a>
  </td>
  <td>
  explanation of the Bounding Box (connection to Context, use of low and high Denoising Strength, use with ControlNet, generation methods), diagram showing Infill techniques, Scale Before Processing, Compositing (Mask Adjustments, blur size and types, coherence pass), slow detailed step-by-step demonstration of generating a scene from scratch on the Unified Canvas using drawing on the Base layer, Inpainting, and multiple iterations, followed by an "overdrive" demonstration of creating a "yellow mage of the dunes" from scratch
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #1 (7:22)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=RwVGDGc6-3o" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/RwVGDGc6-3o/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #2 (4:43)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=WmOUl8Gab5U" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/WmOUl8Gab5U/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #3 (7:01)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=e_rRQeee6-0" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/e_rRQeee6-0/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #4 (7:08)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=OFiJ1Bv0FIM" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/OFiJ1Bv0FIM/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #5 (7:09)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=afSbZEJj2r8" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/afSbZEJj2r8/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Overdrive #6 - Titans in the Gorge Featuring Canvas Controls, Gradient Denoising (6:42)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=gY7II-fjgiw" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/gY7II-fjgiw/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Drivethrough #1 (50:39)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=QSmQ_19rszU" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/QSmQ_19rszU/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Drivethrough #2 (23:45)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=GAlaOlihZ20" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/GAlaOlihZ20/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>InvokeAI - Canvas Tips & Tricks #1 (17:58)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=2pcBtNkTZ40" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/2pcBtNkTZ40/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>Advanced Canvas Inpainting Techniques with Invoke for Concept Art Pro Diffusion Deep Dive (21:05)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=-xzvIpfGTXg" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/-xzvIpfGTXg/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>
</table>

<br>

<table width="100%">
<tr><th colspan="2"><h2>Models</h2></th></tr>
<tr>
  <td width="40%">
  <strong>InvokeAI - Adding Models (v2.2.5) (4:59)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=4aCsmiwwyEI" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/4aCsmiwwyEI/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>

<tr>
  <td width="40%">
  <strong>Exploring AI Models and Concept Adapters/LoRAs (Invoke - Getting Started Series #5) (8:52)</strong>
  <br>
  <a href="https://www.youtube.com/watch?v=iwBmBQMZ0UA" style="color: inherit; text-decoration: inherit;"><img src="https://img.youtube.com/vi/iwBmBQMZ0UA/0.jpg"></a>
  </td>
  <td>
  DESCRIPTION
  </td>
</tr>
</table>

<br>


## Studio Sessions

Long-form videos that go into depth about how to use InvokeAI at a deep level.

***Creating Consistent Character Concepts with IP Adapter and Img2Img (1:05:35)***
<br>[![img](https://img.youtube.com/vi/8p92pqHU9Ag/0.jpg)](https://www.youtube.com/watch?v=8p92pqHU9Ag)


***Creating 2D to 3D Environment Art Renders with AI feat. Blender and Depth Anything (56:22)***
<br>[![img](https://img.youtube.com/vi/Bh-dBXlmh4Q/0.jpg)](https://www.youtube.com/watch?v=Bh-dBXlmh4Q)


***Generate Character and Environment Textures for 3D Renders using Stable Diffusion (58:20)***
<br>[![img](https://img.youtube.com/vi/4F9w4akbnI8/0.jpg)](https://www.youtube.com/watch?v=4F9w4akbnI8)


***Inpainting, Outpainting, and Image Refinement in the Unified Canvas (55:24)***
<br>[![img](https://img.youtube.com/vi/32hyxQQfkW0/0.jpg)](https://www.youtube.com/watch?v=32hyxQQfkW0)


***Workflow to Improve Your Vehicle Concept Art using Shapes and Generative AI (59:07)***
<br>[![img](https://img.youtube.com/vi/NNTUGQhzJE4/0.jpg)](https://www.youtube.com/watch?v=NNTUGQhzJE4)


***Mastering Inpainting: Turn Sketches into Detailed Characters with AI (56:24)***
<br>[![img](https://img.youtube.com/vi/_w5RYKUsN74/0.jpg)](https://www.youtube.com/watch?v=_w5RYKUsN74)


***Mastering Text Prompts and Embeddings in Your Image Creation Workflow (59:04)***
<br>[![img](https://img.youtube.com/vi/ZDYM8ftVGlM/0.jpg)](https://www.youtube.com/watch?v=ZDYM8ftVGlM)


***Create UI/UX Components for Games in Minutes with AI (57:13)***
<br>[![img](https://img.youtube.com/vi/Vp2dEr7KV9o/0.jpg)](https://www.youtube.com/watch?v=Vp2dEr7KV9o)


***How to Create Digital Wallpaper Art for Your Game with Advanced Inpainting and SDXL Models (55:19)***
<br>[![img](https://img.youtube.com/vi/VP44x66gYcg/0.jpg)](https://www.youtube.com/watch?v=VP44x66gYcg)


***How to Train, Test, and Use a LoRA Model for Character Art Consistency (1:01:59)***
<br>[![img](https://img.youtube.com/vi/ej7ruT7aF04/0.jpg)](https://www.youtube.com/watch?v=ej7ruT7aF04)

<br>

## Version Releases 

Short-form videos that show new versions of InvokeAI as they're released (from newest to oldest).

***Invoke - 4.0 Open Source Release - New Model Manager Interface, Gradient Denoising & More... (6:42)***
<br>[![img](https://img.youtube.com/vi/rpWgSMIPwN4/0.jpg)](https://www.youtube.com/watch?v=rpWgSMIPwN4)


***Invoke - 3.7 Release - New Workflow Interface, Seamless Textures, and DW Pose (4:35)***
<br>[![img](https://img.youtube.com/vi/kmJ9VFQNNhU/0.jpg)](https://www.youtube.com/watch?v=kmJ9VFQNNhU)


***Invoke - 3.6 Release - New User Interface Updates, and more (18:48)***
<br>[![img](https://img.youtube.com/vi/XeS4PAJyczw/0.jpg)](https://www.youtube.com/watch?v=XeS4PAJyczw)


***InvokeAI 3.4 Release - LCM LoRAs, Multi-Image IP Adapter, SD1.5 High Res, and more (15:33)***
<br>[![img](https://img.youtube.com/vi/QUXiRfHYRFg/0.jpg)](https://www.youtube.com/watch?v=QUXiRfHYRFg)


***InvokeAI 3.3 Release - T2I Adapters, Multiple IP Adapters and more... (9:42)***
<br>[![img](https://img.youtube.com/vi/EzpCkXB8DL8/0.jpg)](https://www.youtube.com/watch?v=EzpCkXB8DL8)


***InvokeAI 3.2 Release - Queue Manager, Image Prompts, and more... (14:09)***
<br>[![img](https://img.youtube.com/vi/hFBJwYTCvHg/0.jpg)](https://www.youtube.com/watch?v=hFBJwYTCvHg)


***InvokeAI 3.1 Release - Workflows, SDXL Unified Canvas, and more... (13:52)***
<br>[![img](https://img.youtube.com/vi/ECbZs5hcD-s/0.jpg)](https://www.youtube.com/watch?v=ECbZs5hcD-s)


***InvokeAI 3.0 Release (12:36)***
<br>[![img](https://img.youtube.com/vi/A7uipq4lhrk/0.jpg)](https://www.youtube.com/watch?v=A7uipq4lhrk)


***InvokeAI 2.2 Release (16:26)***
<br>[![img](https://img.youtube.com/vi/hIYBfDtKaus/0.jpg)](https://www.youtube.com/watch?v=hIYBfDtKaus)


***InvokeAI 2.1 Release (14:59)***
<br>[![img](https://img.youtube.com/vi/iRTGti44dp4/0.jpg)](https://www.youtube.com/watch?v=iRTGti44dp4)