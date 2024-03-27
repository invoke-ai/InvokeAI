# Installing Models

## Checkpoint and Diffusers Models

The model checkpoint files (`*.ckpt`) are the Stable Diffusion "secret sauce". They are the product of training the AI on millions of captioned images gathered from multiple sources.

Originally there was only a single Stable Diffusion weights file, which many people named `model.ckpt`.

Today, there are thousands of models, fine tuned to excel at specific styles, genres, or themes.

!!! tip "Model Formats"

    We also have two more popular model formats, both created [HuggingFace]:

    - `safetensors`: Single file, like `.ckpt` files. Prevents malware from lurking in a model.
    - `diffusers`: Splits the model components into separate files, allowing very fast loading.

    InvokeAI supports all three formats. Our backend will convert models to `diffusers` format before running them. This is a transparent process.

## Starter Models

When you first start InvokeAI, you'll see a popup prompting you to install some starter models from the Model Manager.

You'll find a collection of popular and high-quality models available for easy download.

Some models carry license terms that limit their use in commercial applications or on public servers. It's your responsibility to adhere to the license terms.

## Other Models

You can install other models using the Model Manager. Supported install sources include:

- Local path: The file path to the model on your computer.
- URL: A link directly to the model, typically to a model marketplace. Some sites require you to use an API token to download models, which you can [set up in the config file].
- `HuggingFace` repo ID: This points to a HF model. Repo IDs look like this: `XpucT/Deliberate`.
- Folder: Scan a local folder for models. You can install all of the detected models in one click.

### Diffusers models in HF repo subfolders

HuggingFace repos can be structured in any way. Some model authors include multiple models within the same folder.

In this situation, you may need to provide some additional information to identify the model you want, by adding `:subfolder_name` to the repo ID.

!!! example

    Say you have a repo ID `monster-labs/control_v1p_sd15_qrcode_monster`, and the model you want is inside the `v2` subfolder.

    Add `:v2` to the repo ID and use that when installing the model: `monster-labs/control_v1p_sd15_qrcode_monster:v2`
