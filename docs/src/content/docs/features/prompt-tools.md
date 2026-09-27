---
title: LLM Prompt Tools
sidebar:
  order: 3
lastUpdated: 2026-09-26
---

InvokeAI includes two built-in tools that use local language models to help you write better prompts. Both tools appear as small buttons in the top-right corner of the positive prompt area and are only visible when you have a compatible model installed.

## Expand Prompt

Takes your short prompt and expands it into a detailed, vivid description suitable for image generation.

**How to use:**

1. Type a brief prompt (e.g. "a cat in a garden")
2. Click the sparkle button in the prompt area
3. Select a Text LLM model from the dropdown
4. Click **Expand**
5. Your prompt is replaced with the expanded version

**Compatible models:** Any HuggingFace model with a `ForCausalLM` architecture, plus Gemma-4 checkpoints (`Gemma4ForConditionalGeneration`), which can also read an image. Recommended options:

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen2.5 1.5B Instruct | ~3 GB | `Qwen/Qwen2.5-1.5B-Instruct` |
| Phi-3 Mini Instruct | ~7.5 GB | `microsoft/Phi-3-mini-4k-instruct` |
| TinyLlama Chat | ~2 GB | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

Install by pasting the HuggingFace ID into the Model Manager. The model is automatically detected as a **Text LLM** type.

### Prompt enhancement for LTX-2.5

LTX-2.5 was trained on long, single-paragraph audio-visual captions, and its own text encoder is not trained to write them. Lightricks pairs the model with a small dedicated enhancer, and InvokeAI offers the same pairing through Expand Prompt:

* Install **LTX-2.5 Prompt Enhancer (Gemma-4 E2B)** from the starter models (`google/gemma-4-E2B-it`, ~10 GB). It is optional and not part of the LTX-2.5 bundle.
* Two system prompts ship with InvokeAI: **LTX-2.5 Text-to-Video** and **LTX-2.5 Image-to-Video**, the release's own texts.

In the Video panel with an LTX-2 model selected, Expand Prompt preselects the enhancer — when it was installed from the starter models or as `google/gemma-4-E2B-it`; otherwise the popover points you to the starter models — and the matching system prompt. With a **first frame** set it shows the frame with **Describe from the first frame** ticked and uses the Image-to-Video prompt, so the caption begins from what the image shows. Untick it, or pick a model that cannot read images (the popover says so), and it uses the Text-to-Video prompt instead. Any model or system prompt you choose in the popover overrides the suggestion.

The rewritten prompt replaces yours in the prompt box, where you can edit it before pressing Invoke; the prompt that is generated and recorded is exactly what the box holds.

### Reasoning ("thinking") models

Reasoning models such as Qwen3 or the DeepSeek-R1 distills normally write out a chain of thought before their answer. Prompt expansion returns the generated text as-is, so that reasoning would end up in your prompt. InvokeAI therefore renders the chat template with thinking disabled, which makes these models answer directly. Models whose chat template does not support the switch are unaffected, and models that always reason (with no way to turn it off) are not suitable for prompt expansion.

## Image to Prompt

Upload an image and generate a descriptive prompt from it using a vision-language model.

**How to use:**

1. Click the image button in the prompt area
2. Select a LLaVA OneVision model from the dropdown
3. Click **Upload Image** and select an image
4. Click **Generate Prompt**
5. The generated description is set as your prompt

**Compatible models:** LLaVA OneVision models (already supported by InvokeAI).

## Undo

Both tools overwrite your current prompt. You can undo this change:

- Press **Ctrl+Z** (or **Cmd+Z** on macOS) in the prompt textarea within 30 seconds
- The undo state is cleared when you start typing manually

## Workflow Node

The workflow editor provides **Text LLM** and **Text LLM (with System Prompt Preset)** nodes for automated pipelines. Both accept a prompt, model, maximum token count, and seed, then output generated text as a string. The preset variant reads its system prompt from the System Prompts library.

Using the same seed reproduces sampling when the model, prompt, settings, hardware, and software versions remain the same. Results may differ across devices or software versions. The prompt area's **Expand Prompt** tool chooses a fresh seed for each request so repeated expansions can vary.

API callers can pass an optional seed to **Expand Prompt**; the response returns the effective seed for replay. They can also pass an `image_name` to condition the rewrite on a stored image, which requires a Text LLM whose `supports_images` is true.

When a saved `1.0.0` Text LLM workflow is migrated, its new seed defaults to `0`, so repeated runs become deterministic. Connect a random integer node to **seed** if the workflow should vary between runs.
