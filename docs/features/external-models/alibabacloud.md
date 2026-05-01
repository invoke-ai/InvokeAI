---
title: Alibaba Cloud DashScope
---

# :material-cloud-outline: Alibaba Cloud DashScope

Invoke supports Alibaba Cloud's **DashScope** image generation service, giving access to the **Qwen Image** family and **Wan 2.6** text-to-image. Qwen Image is particularly strong at bilingual (Chinese / English) text rendering.

## Getting an API Key

1. Sign in to [Alibaba Cloud Model Studio](https://www.alibabacloud.com/en/product/modelstudio) (the international DashScope portal).
2. Enable **DashScope** and activate the image generation models you plan to use.
3. Create an API key from the **API Keys** section of the console.

## Configuration

Add your key to `api_keys.yaml` in your Invoke root directory:

```yaml
external_alibabacloud_api_key: "your-dashscope-api-key"

# Optional — default is the international endpoint. Use the China endpoint if your account lives there:
#   https://dashscope.aliyuncs.com
external_alibabacloud_base_url: "https://dashscope-intl.aliyuncs.com"
```

Restart Invoke for the change to take effect.

!!! info "International vs. China endpoints"
    DashScope has separate international (`dashscope-intl.aliyuncs.com`) and China (`dashscope.aliyuncs.com`) deployments. Your API key only works on the deployment it was issued on — if you get authentication errors, check that `external_alibabacloud_base_url` matches.

## Available Models

| Model | Modes | Aspect Ratios | Batch | Notes |
| --- | --- | --- | --- | --- |
| **Qwen Image 2.0 Pro** | txt2img | 1:1, 4:3, 3:4, 16:9, 9:16 | up to 4 | Best quality, 2K output, excellent bilingual text. |
| **Qwen Image 2.0** | txt2img | 1:1, 4:3, 3:4, 16:9, 9:16 | up to 4 | Faster / cheaper 2K sibling of 2.0 Pro. |
| **Qwen Image Max** | txt2img | 1:1, 4:3, 3:4, 16:9, 9:16 | up to 4 | High quality at ~1.3K native size. |
| **Qwen Image Edit Max** | txt2img + reference images | 1:1, 4:3, 3:4, 16:9, 9:16 | up to 4 | Image editing with industrial / geometric reasoning. Accepts up to 3 reference images. |
| **Wan 2.6 Text-to-Image** | txt2img | 1:1, 4:3, 3:4, 16:9, 9:16 | up to 4 | Photorealistic T2I at 1K. |

All models support **seed**. Negative prompts are not currently plumbed through to DashScope, so the negative prompt input is ignored for these providers.

## Tips

- **Bilingual prompts.** Qwen Image is unusually good at rendering Chinese text and mixed-language prompts — it's a strong choice when your prompt or desired output contains non-Latin script.
- **Editing** is only supported by Qwen Image Edit Max. Provide up to 3 reference images via the reference-images panel; masks and denoising strength are not supported for this provider.
- **Batching** is capped at 4 images per request. Larger batches are split across multiple API calls.
- **Costs** vary per model — Qwen Image 2.0 Pro is the most expensive, Qwen Image 2.0 the cheapest of the 2.0 family. Check Alibaba Cloud's pricing page before running large batches.
