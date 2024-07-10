---
title: Others
---

# :fontawesome-regular-share-from-square: Others

## **Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg){ align="right" }](https://colab.research.google.com/github/lstein/stable-diffusion/blob/main/notebooks/Stable_Diffusion_AI_Notebook.ipynb)

Open and follow instructions to use an isolated environment running Dream.

Output Example:

![Colab Notebook](../assets/colab_notebook.png)

---

## **Invisible Watermark**

In keeping with the principles for responsible AI generation, and to
help AI researchers avoid synthetic images contaminating their
training sets, InvokeAI adds an invisible watermark to each of the
final images it generates. The watermark consists of the text
"InvokeAI" and can be viewed using the
[invisible-watermarks](https://github.com/ShieldMnt/invisible-watermark)
tool.

Watermarking is controlled using the `invisible-watermark` setting in
`invokeai.yaml`. To turn it off, add the following line under the `Features`
category.

```
invisible_watermark: false
```


## **Weighted Prompts**

You may weight different sections of the prompt to tell the sampler to attach different levels of
priority to them, by adding `:<percent>` to the end of the section you wish to up- or downweight. For
example consider this prompt:

```bash
(tabby cat):0.25 (white duck):0.75 hybrid
```

This will tell the sampler to invest 25% of its effort on the tabby cat aspect of the image and 75%
on the white duck aspect (surprisingly, this example actually works). The prompt weights can use any
combination of integers and floating point numbers, and they do not need to add up to 1.

