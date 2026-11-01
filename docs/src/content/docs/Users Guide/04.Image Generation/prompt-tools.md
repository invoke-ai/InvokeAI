---
title: Prompt Tools
sidebar:
  order: 4
lastUpdated: 2026-11-01
---

The prompt toolbar holds tools that help you write prompts. **Expand Prompt** and **Image to Prompt** use local language models to write or rewrite a prompt for you; if no compatible model is installed, the tool's popover says so and offers an **Open Model Manager** button. **Prompt triggers** insert the trigger phrases, embeddings and wildcards your models need.

## Expand Prompt

Takes your short prompt and expands it into a detailed, vivid description suitable for image generation.

**How to use:**

1. Type a brief prompt (e.g. "a cat in a garden")
2. Click **Expand prompt** (the pencil-and-sparkles button) in the prompt toolbar
3. Select a Text LLM model from the dropdown. The panel remembers your choice for the project.
4. Choose a **System prompt**, the instructions that tell the model how to rewrite your prompt
5. Click **Expand**
6. Your prompt is replaced with the expanded version, which you can edit before generating

### System prompts

InvokeAI ships with a **Default** system prompt and several tuned for particular model families, such as FLUX.2, Z-Image, Qwen Image, Krea-2 and MOSS-Image. To add your own, click the gear button next to the System prompt selector (**Manage system prompts**) and choose **New**. A system prompt has a **Name**, the **Instructions** given to the language model, and an optional **Max output length** in tokens (300 by default).

You can duplicate, edit and delete your own system prompts. Shared system prompts, including the built-in ones, are marked **Shared**. On a multi-user installation only administrators can edit them, and changes apply to every user.

**Compatible models:** Any HuggingFace model with a `ForCausalLM` architecture, plus Gemma-4 (`Gemma4ForConditionalGeneration`) and Qwen3.5 vision-language models (`Qwen3_5ForConditionalGeneration`), which can also read an image. The FLUX.3 prompt upsampler is one of these, as a folder or as ComfyUI's single file; see [Model Format Support](/users-guide/models/model-format-support/#text-llms). Recommended options:

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen2.5 1.5B Instruct | ~3 GB | `Qwen/Qwen2.5-1.5B-Instruct` |
| Phi-3 Mini Instruct | ~7.5 GB | `microsoft/Phi-3-mini-4k-instruct` |
| TinyLlama Chat | ~2 GB | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

Install by pasting the HuggingFace ID into the Model Manager. The model is automatically detected as a **Text LLM** type.

### The FLUX.3 prompt upsampler

Black Forest Labs' [prompt upsampler for FLUX.3](/users-guide/models/local-models/flux-3/#prompt-upsampler-model) is a Text LLM. Install **FLUX.3 Prompt Upsampler** (~17.5 GiB) or **FLUX.3 Prompt Upsampler (int8)** (~9.1 GiB) from the starter models and pick it in Expand Prompt. It rewrites your text prompt: in the Generate panel, Expand Prompt sends it no reference images. Where Expand Prompt does send an image, such as the Video panel's first frame, it reads that image like any Text LLM that can. The [FLUX.3 page](/users-guide/models/local-models/flux-3/#prompt-upsampler-model) lists how fast each build ran and how much VRAM it took on an RTX 4090.

Expand Prompt sends the model the system prompt you pick. The upsampler was trained with Black Forest Labs' four system prompts, published in its [`system_prompts.json`](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/raw/3f3edb52b2c55de0d49feab28598b68f7f8e4830/system_prompts.json); `t2i` is the one for rewriting a text-to-image prompt. To use it:

1. Open **Manage system prompts** next to the System prompt selector and choose **New**.
2. Paste the text of the `t2i` entry as the **Instructions**. The file is JSON, so undo its escapes as you paste: each `\n` becomes a line break and each `\"` a plain `"`.
3. Set **Max output length** to about 1024 tokens, so a long caption is not cut off.
4. Choose this system prompt whenever you expand a prompt with the upsampler.

Expand Prompt samples its rewrite, so repeated expansions differ; Black Forest Labs' own pipeline decodes greedily.

### Prompt enhancement for LTX-2.5

LTX-2.5 was trained on long, single-paragraph audio-visual captions, and its own text encoder is not trained to write them. Lightricks pairs the model with a small dedicated enhancer, and InvokeAI offers the same pairing through Expand Prompt:

* Install **LTX-2.5 Prompt Enhancer (Gemma-4 E2B)** from the starter models (`google/gemma-4-E2B-it`, ~10 GB). It is optional and not part of the LTX-2.5 bundle.
* Two system prompts ship with InvokeAI: **LTX-2.5 Text-to-Video** and **LTX-2.5 Image-to-Video**, the release's own texts.

In the Video panel with an LTX-2 model selected, Expand Prompt preselects the matching system prompt. For the model, a pick in the open popover comes first, then the model saved for the project, then the enhancer when it was installed from the starter models or as `google/gemma-4-E2B-it`, then the first available model. When the enhancer is installed but another model is selected, the popover shows **Recommended for this model** with a **Use it** button, which saves it like any other pick; when it is not installed, the popover points you to the starter models. With a **first frame** set it shows the frame with **Describe from the first frame** ticked and uses the Image-to-Video prompt, so the caption begins from what the image shows. Untick it, or pick a model that cannot read images (the popover says so), and it uses the Text-to-Video prompt instead. A model or system prompt you pick in the open popover overrides the suggestion.

The rewritten prompt replaces yours in the prompt box, where you can edit it before pressing Invoke; the prompt that is generated and recorded is exactly what the box holds.

### Prompt enhancement for MOSS-Image

The **MOSS-Image Prompt Enhancer** system prompt, derived from OpenMOSS's prompt-enhancer skill, has the language model write either a detailed caption or, for posters and other images with placed text, a layout draft that MOSS-Image compiles into its JSON layout at generation time. With a MOSS-Image model selected, Expand Prompt in the Generate panel (whose prompt the Canvas also uses) preselects this system prompt and the **Gemma 4 E4B (Prompt Enhancer)** starter model (`google/gemma-4-E4B-it`, ~16 GB), following the same rule as for LTX-2 above. Small models such as Qwen2.5 1.5B are not recommended for this system prompt. See [MOSS-Image](/users-guide/models/local-models/moss-image/#prompt-enhancer) for caption versus layout, image size and canvas regions.

### Reasoning ("thinking") models

Reasoning models such as Qwen3 or the DeepSeek-R1 distills normally write out a chain of thought before their answer. Prompt expansion returns the generated text as-is, so that reasoning would end up in your prompt. InvokeAI therefore renders the chat template with thinking disabled, which makes these models answer directly. Models whose chat template does not support the switch are unaffected, and models that always reason (with no way to turn it off) are not suitable for prompt expansion.

## Image to Prompt

Upload an image and generate a descriptive prompt from it using a vision-language model.

**How to use:**

1. Select an image in the Gallery or Preview, or drag an image from the Gallery onto the prompt box (**Describe this image**), which opens the tool for you
2. Click **Image to prompt** in the prompt toolbar if it is not already open
3. Select a LLaVA OneVision model from the dropdown
4. Click **Generate Prompt**
5. The generated description replaces your prompt

**Compatible models:** LLaVA OneVision models (already supported by InvokeAI).

## Replacing your prompt

Both tools replace the text in the prompt box. To get an earlier prompt back, open **Prompt history** in the prompt toolbar: every prompt you have submitted with Invoke is kept there.

## Prompt triggers

Some models only produce their effect when the prompt contains a particular word or phrase. A LoRA trained on a character may need the character's name; a style LoRA may need a keyword such as `pixel art style`. These **trigger phrases** are easy to forget, so InvokeAI collects them, together with embeddings and wildcards, in a menu you can insert from while writing a prompt.

### Inserting a trigger

Click the **+** button in the prompt toolbar (**Add prompt trigger**). It is available on the positive and negative prompts in the Generate, Video and Upscale panels. The menu lists, in this order:

| Group | What it contains | Inserted as |
|---|---|---|
| **Wildcards** | Your [wildcards](/users-guide/image-generation/prompting-guide/#wildcards) (positive prompt only). | `__name__` |
| *The selected main model's name* | The main model's trigger phrases. | the phrase |
| *Each enabled concept's name* | The trigger phrases of each LoRA that is switched on in the Guidance section. Disabled concepts are skipped. | the phrase |
| **Compatible embeddings** | Every installed embedding (textual inversion) that works with the selected main model. | `<embedding-name>` |

Type in **Search prompt triggers** to filter the list, then click an item. It is inserted at the cursor exactly as shown, without extra spaces or commas, so add any separators you need.

If the menu is empty, none of your installed models have trigger phrases and no embeddings or wildcards are available. The **Open Model Manager** button takes you to where trigger phrases are set. The **+** button is disabled while you are previewing the merged prompt of a [template](/users-guide/image-generation/prompting-guide/#prompt-templates).

Two kinds of triggers can also be completed as you type:

* Type `<` in either prompt to list compatible **embeddings**.
* Type `__` (two underscores) in the positive prompt to list **wildcards**.

Use the arrow keys to choose, **Enter** or **Tab** to insert, and **Escape** to close the list. Trigger phrases of models and LoRAs are only offered in the **+** menu.

### Defining trigger phrases

Trigger phrases belong to a model and are set in the **Model Manager**. Select a main model or a LoRA and find the **Trigger Phrases** field in its detail view. Type a phrase and press **Enter** to add it; each phrase can be up to 200 characters, and duplicates are refused. Changes are saved as you make them.

Trigger phrases come from the model's creator, so check the model's download page (Civitai or HuggingFace) for the words it expects. Two things fill them in for you:

* When you install a LoRA that has a companion `.json` file next to it with an `activation text` entry, as some download tools create, those phrases become the LoRA's trigger phrases.
* Trigger phrases are included when you [export and import a model's settings](/users-guide/models/introduction/#exporting-and-importing-model-settings).

The Guidance section of the Generate panel shows each LoRA's trigger phrases under its name as a reminder. Adding a LoRA does not insert them into your prompt; use the **+** menu for that.

## Text LLM workflow nodes

The workflow editor provides **Text LLM** and **Text LLM (with System Prompt Preset)** nodes for automated pipelines. Both accept a prompt, model, maximum token count, and seed, then output generated text as a string. The preset variant reads its system prompt from the System Prompts library.

Using the same seed reproduces sampling when the model, prompt, settings, hardware, and software versions remain the same. Results may differ across devices or software versions. The prompt area's **Expand Prompt** tool chooses a fresh seed for each request so repeated expansions can vary.

API callers can pass an optional seed to **Expand Prompt**; the response returns the effective seed for replay. They can also pass an `image_name` to condition the rewrite on a stored image, which requires a Text LLM whose `supports_images` is true.

When a saved `1.0.0` Text LLM workflow is migrated, its new seed defaults to `0`, so repeated runs become deterministic. Connect a random integer node to **seed** if the workflow should vary between runs.
