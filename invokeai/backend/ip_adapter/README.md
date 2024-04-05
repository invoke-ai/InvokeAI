# IP-Adapter Model Formats

The official IP-Adapter models are released here: [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)

This official model repo does not integrate well with InvokeAI's current approach to model management, so we have defined a new file structure for IP-Adapter models. The InvokeAI format is described below.

## CLIP Vision Models

CLIP Vision models are organized in `diffusers`` format. The expected directory structure is:

```bash
ip_adapter_sd_image_encoder/
├── config.json
└── model.safetensors
```

## IP-Adapter Models

IP-Adapter models are stored in a directory containing two files
- `image_encoder.txt`: A text file containing the model identifier for the CLIP Vision encoder that is intended to be used with this IP-Adapter model.
- `ip_adapter.bin`: The IP-Adapter weights.

Sample directory structure:
```bash
ip_adapter_sd15/
├── image_encoder.txt
└── ip_adapter.bin
```

### Why save the weights in a .safetensors file?

The weights in `ip_adapter.bin` are stored in a nested dict, which is not supported by `safetensors`. This could be solved by splitting `ip_adapter.bin` into multiple files, but for now we have decided to maintain consistency with the checkpoint structure used in the official [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) repo.

## InvokeAI Hosted IP-Adapters

CLIP Vision Image Encoders:
- [InvokeAI/ip_adapter_sd_image_encoder](https://huggingface.co/InvokeAI/ip_adapter_sd_image_encoder)
- [InvokeAI/ip_adapter_sdxl_image_encoder](https://huggingface.co/InvokeAI/ip_adapter_sdxl_image_encoder)

IP-Adapters:
- [InvokeAI/ip_adapter_sd15](https://huggingface.co/InvokeAI/ip_adapter_sd15)
- [InvokeAI/ip_adapter_plus_sd15](https://huggingface.co/InvokeAI/ip_adapter_plus_sd15)
- [InvokeAI/ip_adapter_plus_face_sd15](https://huggingface.co/InvokeAI/ip_adapter_plus_face_sd15)
- [InvokeAI/ip_adapter_sdxl](https://huggingface.co/InvokeAI/ip_adapter_sdxl)
- [InvokeAI/ip_adapter_sdxl_vit_h](https://huggingface.co/InvokeAI/ip_adapter_sdxl_vit_h)