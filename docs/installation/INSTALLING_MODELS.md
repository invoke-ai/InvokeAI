---
title: Installing Models
---

# :octicons-paintbrush-16: Installing Models

## Model Weight Files

The model weight files ('\*.ckpt') are the Stable Diffusion "secret sauce". They
are the product of training the AI on millions of captioned images gathered from
multiple sources.

Originally there was only a single Stable Diffusion weights file, which many
people named `model.ckpt`. Now there are dozens or more that have been "fine
tuned" to provide particulary styles, genres, or other features. InvokeAI allows
you to install and run multiple model weight files and switch between them
quickly in the command-line and web interfaces.

This manual will guide you through installing and configuring model weight
files.

## Base Models

InvokeAI comes with support for a good initial set of models listed in the model
configuration file `configs/models.yaml`. They are:

| Model                | Weight File                       | Description                                                | DOWNLOAD FROM                                                  |
| -------------------- | --------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| stable-diffusion-1.5 | v1-5-pruned-emaonly.ckpt          | Most recent version of base Stable Diffusion model         | https://huggingface.co/runwayml/stable-diffusion-v1-5          |
| stable-diffusion-1.4 | sd-v1-4.ckpt                      | Previous version of base Stable Diffusion model            | https://huggingface.co/CompVis/stable-diffusion-v-1-4-original |
| inpainting-1.5       | sd-v1-5-inpainting.ckpt           | Stable Diffusion 1.5 model specialized for inpainting      | https://huggingface.co/runwayml/stable-diffusion-inpainting    |
| waifu-diffusion-1.3  | model-epoch09-float32.ckpt        | Stable Diffusion 1.4 trained to produce anime images       | https://huggingface.co/hakurei/waifu-diffusion-v1-3            |
| `<all models>`       | vae-ft-mse-840000-ema-pruned.ckpt | A fine-tune file add-on file that improves face generation | https://huggingface.co/stabilityai/sd-vae-ft-mse-original/     |

Note that these files are covered by an "Ethical AI" license which forbids
certain uses. You will need to create an account on the Hugging Face website and
accept the license terms before you can access the files.

The predefined configuration file for InvokeAI (located at
`configs/models.yaml`) provides entries for each of these weights files.
`stable-diffusion-1.5` is the default model used, and we strongly recommend that
you install this weights file if nothing else.

## Community-Contributed Models

There are too many to list here and more are being contributed every day.
Hugging Face maintains a
[fast-growing repository](https://huggingface.co/sd-concepts-library) of
fine-tune (".bin") models that can be imported into InvokeAI by passing the
`--embedding_path` option to the `invoke.py` command.

[This page](https://rentry.org/sdmodels) hosts a large list of official and
unofficial Stable Diffusion models and where they can be obtained.

## Installation

There are three ways to install weights files:

1. During InvokeAI installation, the `preload_models.py` script can download
   them for you.

2. You can use the command-line interface (CLI) to import, configure and modify
   new models files.

3. You can download the files manually and add the appropriate entries to
   `models.yaml`.

### Installation via `preload_models.py`

This is the most automatic way. Run `scripts/preload_models.py` from the
console. It will ask you to select which models to download and lead you through
the steps of setting up a Hugging Face account if you haven't done so already.

To start, run `python scripts/preload_models.py` from within the InvokeAI:
directory

!!! example ""

    ```text
    Loading Python libraries...

    ** INTRODUCTION **
    Welcome to InvokeAI. This script will help download the Stable Diffusion weight files
    and other large models that are needed for text to image generation. At any point you may interrupt
    this program and resume later.

    ** WEIGHT SELECTION **
    Would you like to download the Stable Diffusion model weights now? [y]

    Choose the weight file(s) you wish to download. Before downloading you
    will be given the option to view and change your selections.

    [1] stable-diffusion-1.5:
        The newest Stable Diffusion version 1.5 weight file (4.27 GB) (recommended)
        Download? [y]
    [2] inpainting-1.5:
        RunwayML SD 1.5 model optimized for inpainting (4.27 GB) (recommended)
        Download? [y]
    [3] stable-diffusion-1.4:
        The original Stable Diffusion version 1.4 weight file (4.27 GB)
        Download? [n] n
    [4] waifu-diffusion-1.3:
        Stable Diffusion 1.4 fine tuned on anime-styled images (4.27)
        Download? [n] y
    [5] ft-mse-improved-autoencoder-840000:
        StabilityAI improved autoencoder fine-tuned for human faces (recommended; 335 MB) (recommended)
        Download? [y] y
    The following weight files will be downloaded:
      [1] stable-diffusion-1.5*
      [2] inpainting-1.5
      [4] waifu-diffusion-1.3
      [5] ft-mse-improved-autoencoder-840000
    *default
    Ok to download? [y]
    ** LICENSE AGREEMENT FOR WEIGHT FILES **

    1. To download the Stable Diffusion weight files you need to read and accept the
      CreativeML Responsible AI license. If you have not already done so, please
      create an account using the "Sign Up" button:

      https://huggingface.co

      You will need to verify your email address as part of the HuggingFace
      registration process.

    2. After creating the account, login under your account and accept
      the license terms located here:

      https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

    Press <enter> when you are ready to continue:
    ...
    ```

When the script is complete, you will find the downloaded weights files in
`models/ldm/stable-diffusion-v1` and a matching configuration file in
`configs/models.yaml`.

You can run the script again to add any models you didn't select the first time.
Note that as a safety measure the script will _never_ remove a
previously-installed weights file. You will have to do this manually.

### Installation via the CLI

You can install a new model, including any of the community-supported ones, via
the command-line client's `!import_model` command.

1.  First download the desired model weights file and place it under
    `models/ldm/stable-diffusion-v1/`. You may rename the weights file to
    something more memorable if you wish. Record the path of the weights file
    (e.g. `models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt`)

2.  Launch the `invoke.py` CLI with `python scripts/invoke.py`.

3.  At the `invoke>` command-line, enter the command
    `!import_model <path to model>`. For example:

    `invoke> !import_model models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt`

    !!! tip "the CLI supports file path autocompletion"

         Type a bit of the path name and hit ++tab++ in order to get a choice of
         possible completions.

4.  Follow the wizard's instructions to complete installation as shown in the
    example here:

    !!! example ""

        ```text
        invoke> !import_model models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt
        >> Model import in process. Please enter the values needed to configure this model:

        Name for this model: arabian-nights
        Description of this model: Arabian Nights Fine Tune v1.0
        Configuration file for this model: configs/stable-diffusion/v1-inference.yaml
        Default image width: 512
        Default image height: 512
        >> New configuration:
        arabian-nights:
          config: configs/stable-diffusion/v1-inference.yaml
          description: Arabian Nights Fine Tune v1.0
          height: 512
          weights: models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt
          width: 512
        OK to import [n]? y
        >> Caching model stable-diffusion-1.4 in system RAM
        >> Loading waifu-diffusion from models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt
          | LatentDiffusion: Running in eps-prediction mode
          | DiffusionWrapper has 859.52 M params.
          | Making attention of type 'vanilla' with 512 in_channels
          | Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
          | Making attention of type 'vanilla' with 512 in_channels
          | Using faster float16 precision
        ```

If you've previously installed the fine-tune VAE file
`vae-ft-mse-840000-ema-pruned.ckpt`, the wizard will also ask you if you want to
add this VAE to the model.

The appropriate entry for this model will be added to `configs/models.yaml` and
it will be available to use in the CLI immediately.

The CLI has additional commands for switching among, viewing, editing, deleting
the available models. These are described in
[Command Line Client](../features/CLI.md#model-selection-and-importation), but
the two most frequently-used are `!models` and `!switch <name of model>`. The
first prints a table of models that InvokeAI knows about and their load status.
The second will load the requested model and lets you switch back and forth
quickly among loaded models.

### Manually editing of `configs/models.yaml`

If you are comfortable with a text editor then you may simply edit `models.yaml`
directly.

First you need to download the desired .ckpt file and place it in
`models/ldm/stable-diffusion-v1` as descirbed in step #1 in the previous
section. Record the path to the weights file, e.g.
`models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt`

Then using a **text** editor (e.g. the Windows Notepad application), open the
file `configs/models.yaml`, and add a new stanza that follows this model:

```yaml
arabian-nights-1.0:
  description: A great fine-tune in Arabian Nights style
  weights: ./models/ldm/stable-diffusion-v1/arabian-nights-1.0.ckpt
  config: ./configs/stable-diffusion/v1-inference.yaml
  width: 512
  height: 512
  vae: ./models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
  default: false
```

| name               | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| arabian-nights-1.0 | This is the name of the model that you will refer to from within the CLI and the WebGUI when you need to load and use the model.                                                                                                                                                                                                                                                                                                                                                                                                  |
| description        | Any description that you want to add to the model to remind you what it is.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| weights            | Relative path to the .ckpt weights file for this model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| config             | This is the confusingly-named configuration file for the model itself. Use `./configs/stable-diffusion/v1-inference.yaml` unless the model happens to need a custom configuration, in which case the place you downloaded it from will tell you what to use instead. For example, the runwayML custom inpainting model requires the file `configs/stable-diffusion/v1-inpainting-inference.yaml`. This is already inclued in the InvokeAI distribution and is configured automatically for you by the `preload_models.py` script. |
| vae                | If you want to add a VAE file to the model, then enter its path here.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| width, height      | This is the width and height of the images used to train the model. Currently they are always 512 and 512.                                                                                                                                                                                                                                                                                                                                                                                                                        |

Save the `models.yaml` and relaunch InvokeAI. The new model should now be
available for your use.
