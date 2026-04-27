# LLM Prompt Tools

InvokeAI includes two built-in tools that use local language models to help you write better prompts. Both tools appear as small buttons in the top-right corner of the positive prompt area and are only visible when you have a compatible model installed.

## Expand Prompt

Takes your short prompt and expands it into a detailed, vivid description suitable for image generation.

**How to use:**

1. Type a brief prompt (e.g. "a cat in a garden")
2. Click the sparkle button in the prompt area
3. Select a Text LLM model from the dropdown
4. Click **Expand**
5. Your prompt is replaced with the expanded version

**Compatible models:** Any HuggingFace model with a `ForCausalLM` architecture. Recommended options:

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen2.5 1.5B Instruct | ~3 GB | `Qwen/Qwen2.5-1.5B-Instruct` |
| Phi-3 Mini Instruct | ~7.5 GB | `microsoft/Phi-3-mini-4k-instruct` |
| TinyLlama Chat | ~2 GB | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

Install by pasting the HuggingFace ID into the Model Manager. The model is automatically detected as a **Text LLM** type.

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

A **Text LLM** node is also available in the workflow editor for use in automated pipelines. It accepts a prompt string and model selection as inputs and outputs the expanded text as a string.
