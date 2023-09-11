# List of Default Nodes

The table below contains a list of the default nodes shipped with InvokeAI and their descriptions. 

| Node <img width=160 align="right"> | Function                                                                              |
|: ---------------------------------- | :--------------------------------------------------------------------------------------|
|Add Integers 			| Adds two numbers|
|Boolean Primitive Collection 			| A collection of boolean primitive values|
|Boolean Primitive 			| A boolean primitive value|
|Canny Processor 			| Canny edge detection for ControlNet|
|CLIP Skip 			| Skip layers in clip text_encoder model.|
|Collect 			| Collects values into a collection|
|Color Correct 			| Shifts the colors of a target image to match the reference image, optionally using a mask to only color-correct certain regions of the target image.|
|Color Primitive 			| A color primitive value|
|Compel Prompt 			| Parse prompt using compel package to conditioning.|
|Conditioning Primitive Collection 			| A collection of conditioning tensor primitive values|
|Conditioning Primitive 			| A conditioning tensor primitive value|
|Content Shuffle Processor 			| Applies content shuffle processing to image|
|ControlNet 			| Collects ControlNet info to pass to other nodes|
|OpenCV Inpaint 			| Simple inpaint using opencv.|
|Denoise Latents 			| Denoises noisy latents to decodable images|
|Divide Integers 			| Divides two numbers|
|Dynamic Prompt 			| Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator|
|Upscale (RealESRGAN) 			| Upscales an image using RealESRGAN.|
|Float Primitive Collection 			| A collection of float primitive values|
|Float Primitive 			| A float primitive value|
|Float Range 			| Creates a range|
|HED (softedge) Processor 			| Applies HED edge detection to image|
|Blur Image 			| Blurs an image|
|Extract Image Channel 			| Gets a channel from an image.|
|Image Primitive Collection 			| A collection of image primitive values|
|Convert Image Mode 			| Converts an image to a different mode.|
|Crop Image 			| Crops an image to a specified box. The box can be outside of the image.|
|Image Hue Adjustment 			| Adjusts the Hue of an image.|
|Inverse Lerp Image 			| Inverse linear interpolation of all pixels of an image|
|Image Primitive 			| An image primitive value|
|Lerp Image 			| Linear interpolation of all pixels of an image|
|Offset Image Channel 			| Add to or subtract from an image color channel by a uniform value.|
|Multiply Image Channel 			| Multiply or Invert an image color channel by a scalar value.|
|Multiply Images 			| Multiplies two images together using `PIL.ImageChops.multiply()`.|
|Blur NSFW Image 			| Add blur to NSFW-flagged images|
|Paste Image 			| Pastes an image into another image.|
|ImageProcessor 			| Base class for invocations that preprocess images for ControlNet|
|Resize Image 			| Resizes an image to specific dimensions|
|Scale Image 			| Scales an image by a factor|
|Image to Latents 			| Encodes an image into latents.|
|Add Invisible Watermark 			| Add an invisible watermark to an image|
|Solid Color Infill 			| Infills transparent areas of an image with a solid color|
|PatchMatch Infill 			| Infills transparent areas of an image using the PatchMatch algorithm|
|Tile Infill 			| Infills transparent areas of an image with tiles of the image|
|Integer Primitive Collection 			| A collection of integer primitive values|
|Integer Primitive 			| An integer primitive value|
|Iterate 			| Iterates over a list of items|
|Latents Primitive Collection 			| A collection of latents tensor primitive values|
|Latents Primitive 			| A latents tensor primitive value|
|Latents to Image 			| Generates an image from latents.|
|Leres (Depth) Processor 			| Applies leres processing to image|
|Lineart Anime Processor 			| Applies line art anime processing to image|
|Lineart Processor 			| Applies line art processing to image|
|LoRA Loader 			| Apply selected lora to unet and text_encoder.|
|Main Model Loader 			| Loads a main model, outputting its submodels.|
|Combine Mask 			| Combine two masks together by multiplying them using `PIL.ImageChops.multiply()`.|
|Mask Edge 			| Applies an edge mask to an image|
|Mask from Alpha 			| Extracts the alpha channel of an image as a mask.|
|Mediapipe Face Processor 			| Applies mediapipe face processing to image|
|Midas (Depth) Processor 			| Applies Midas depth processing to image|
|MLSD Processor 			| Applies MLSD processing to image|
|Multiply Integers 			| Multiplies two numbers|
|Noise 			| Generates latent noise.|
|Normal BAE Processor 			| Applies NormalBae processing to image|
|ONNX Latents to Image 			| Generates an image from latents.|
|ONNX Prompt (Raw) 			| A node to process inputs and produce outputs. May use dependency injection in __init__ to receive providers.|
|ONNX Text to Latents 			| Generates latents from conditionings.|
|ONNX Model Loader 			| Loads a main model, outputting its submodels.|
|Openpose Processor 			| Applies Openpose processing to image|
|PIDI Processor 			| Applies PIDI processing to image|
|Prompts from File 			| Loads prompts from a text file|
|Random Integer 			| Outputs a single random integer.|
|Random Range 			| Creates a collection of random numbers|
|Integer Range 			| Creates a range of numbers from start to stop with step|
|Integer Range of Size 			| Creates a range from start to start + size with step|
|Resize Latents 			| Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8.|
|SDXL Compel Prompt 			| Parse prompt using compel package to conditioning.|
|SDXL LoRA Loader 			| Apply selected lora to unet and text_encoder.|
|SDXL Main Model Loader 			| Loads an sdxl base model, outputting its submodels.|
|SDXL Refiner Compel Prompt 			| Parse prompt using compel package to conditioning.|
|SDXL Refiner Model Loader 			| Loads an sdxl refiner model, outputting its submodels.|
|Scale Latents 			| Scales latents by a given factor.|
|Segment Anything Processor 			| Applies segment anything processing to image|
|Show Image 			| Displays a provided image, and passes it forward in the pipeline.|
|Step Param Easing 			| Experimental per-step parameter easing for denoising steps|
|String Primitive Collection 			| A collection of string primitive values|
|String Primitive 			| A string primitive value|
|Subtract Integers 			| Subtracts two numbers|
|Tile Resample Processor 			| Tile resampler processor|
|VAE Loader 			| Loads a VAE model, outputting a VaeLoaderOutput|
|Zoe (Depth) Processor 			| Applies Zoe depth processing to image|