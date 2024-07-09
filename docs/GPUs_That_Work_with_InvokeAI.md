---
Class: ai
Topic: InvokeAI Official Doc
Document Section: Installation
Created: 2024-07-08
Published to My Github: true
Pull Request: 
Author: Smile4yourself
---
### GPUs

Problematic Nvidia GPUs

We do not recommend these GPUs. They cannot operate with half precision, but have insufficient VRAM to generate 512x512 images at full precision.

| Cards Not Recommended     | Manufacturer |
| ------------------------- | ------------ |
| 10xx cards such as 1080TI | NVIDIA       |
| GTX 1650 series cards     | NVIDIA       |
| GTX 1660 series cards     | NVIDIA       |


Invoke runs best with a dedicated GPU, but will fall back to running on CPU, albeit much slower. You'll need a beefier GPU for SDXL.

| VRAM | GPU | Model Used |
| ---- | --- | ---------- |
| 4 GB | any | SD 1.5     |
| 8 GB | any | SDXL       |


## Recommended GPUs

let me know what GPUs works well for you and I'll post it here. (@Smile4yourself on Discord)

