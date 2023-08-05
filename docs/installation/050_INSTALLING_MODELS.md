---
title: Installing Models
---

# :octicons-paintbrush-16: Installing Models

## Checkpoint and Diffusers Models

The model checkpoint files ('\*.ckpt') are the Stable Diffusion
"secret sauce". They are the product of training the AI on millions of
captioned images gathered from multiple sources.

Originally there was only a single Stable Diffusion weights file,
which many people named `model.ckpt`. Now there are dozens or more
that have been fine tuned to provide particulary styles, genres, or
other features. In addition, there are several new formats that
improve on the original checkpoint format: a `.safetensors` format
which prevents malware from masquerading as a model, and `diffusers`
models, the most recent innovation.

InvokeAI supports all three formats but strongly prefers the
`diffusers` format. These are distributed as directories containing
multiple subfolders, each of which contains a different aspect of the
model. The advantage of this is that the models load from disk really
fast. Another advantage is that `diffusers` models are supported by a
large and active set of open source developers working at and with
HuggingFace organization, and improvements in both rendering quality
and performance are being made at a rapid pace. Among other features
is the ability to download and install a `diffusers` model just by
providing its HuggingFace repository ID.

While InvokeAI will continue to support `.ckpt` and `.safetensors`
models for the near future, these are deprecated and support will
likely be withdrawn at some point in the not-too-distant future.

This manual will guide you through installing and configuring model
weight files and converting legacy `.ckpt` and `.safetensors` files
into performant `diffusers` models.

## Base Models

InvokeAI comes with support for a good set of starter models. You'll
find them listed in the master models file
`configs/INITIAL_MODELS.yaml` in the InvokeAI root directory. The
subset that are currently installed are found in
`configs/models.yaml`.

Note that these files are covered by an "Ethical AI" license which
forbids certain uses. When you initially download them, you are asked
to accept the license terms. In addition, some of these models carry
additional license terms that limit their use in commercial
applications or on public servers. Be sure to familiarize yourself
with the model terms by visiting the URLs in the table above.

## Community-Contributed Models

[HuggingFace](https://huggingface.co/models?library=diffusers)
is a great resource for diffusers models, and is also the home of a
[fast-growing repository](https://huggingface.co/sd-concepts-library)
of embedding (".bin") models that add subjects and/or styles to your
images. The latter are automatically installed on the fly when you
include the text `<concept-name>` in your prompt. See [Concepts
Library](../features/CONCEPTS.md) for more information.

Another popular site for community-contributed models is
[CIVITAI](https://civitai.com). This extensive site currently supports
only `.safetensors` and `.ckpt` models, but they can be easily loaded
into InvokeAI and/or converted into optimized `diffusers` models. Be
aware that CIVITAI hosts many models that generate NSFW content.

## Installation

There are two ways to install and manage models:

1. The `invokeai-model-install` script which will download and install
them for you.  In addition to supporting main models, you can install
ControlNet, LoRA and Textual Inversion models.

2. The web interface (WebUI) has a GUI for importing and managing
   models.

3. By placing models (or symbolic links to models) inside one of the
InvokeAI root directory's `autoimport` folder.

### Installation via `invokeai-model-install`

From the `invoke` launcher, choose option [5] "Download and install
models." This will launch the same script that prompted you to select
models at install time. You can use this to add models that you
skipped the first time around. It is all right to specify a model that
was previously downloaded; the script will just confirm that the files
are complete.

The installer has different panels for installing main models from
HuggingFace, models from Civitai and other arbitrary web sites,
ControlNet models, LoRA/LyCORIS models, and Textual Inversion
embeddings. Each section has a text box in which you can enter a new
model to install. You can refer to a model using its:

1. Local path to the .ckpt, .safetensors or diffusers folder on your local machine
2. A directory on your machine that contains multiple models
3. A URL that points to a downloadable model
4. A HuggingFace repo id

Previously-installed models are shown with checkboxes. Uncheck a box
to unregister the model from InvokeAI. Models that are physically
installed inside the InvokeAI root directory will be deleted and
purged (after a confirmation warning). Models that are located outside
the InvokeAI root directory will be unregistered but not deleted.

Note: The installer script uses a console-based text interface that requires
significant amounts of horizontal and vertical space. If the display
looks messed up, just enlarge the terminal window and/or relaunch the
script.

If you wish you can script model addition and deletion, as well as
listing installed models. Start the "developer's console" and give the
command `invokeai-model-install --help`. This will give you a series
of command-line parameters that will let you control model
installation. Examples:

```
# (list all controlnet models)
invokeai-model-install --list controlnet

# (install the model at the indicated URL)
invokeai-model-install --add https://civitai.com/api/download/models/128713

# (delete the named model)
invokeai-model-install --delete sd-1/main/analog-diffusion
```

### Installation via the Web GUI

To install a new model using the Web GUI, do the following:

1. Open the InvokeAI Model Manager (cube at the bottom of the
left-hand panel) and navigate to *Import Models*

2. In the field labeled *Location* type in the path to the model you
wish to install. You may use a URL, HuggingFace repo id, or a path on
your local disk.

3. Alternatively, the *Scan for Models* button allows you to paste in
the path to a folder somewhere on your machine. It will be scanned for
importable models and prompt you to add the ones of your choice.

4. Press *Add Model* and wait for confirmation that the model
was added.

To delete a model, Select *Model Manager* to list all the currently
installed models. Press the trash can icons to delete any models you
wish to get rid of. Models whose weights are located inside the
InvokeAI `models` directory will be purged from disk, while those
located outside will be unregistered from InvokeAI, but not deleted.

You can see where model weights are located by clicking on the model name.
This will bring up an editable info panel showing the model's characteristics,
including the `Model Location` of its files.

### Installation via the `autoimport` function

In the InvokeAI root directory you will find a series of folders under
`autoimport`, one each for main models, controlnets, embeddings and
Loras.  Any models that you add to these directories will be scanned
at startup time and registered automatically.

You may create symbolic links from these folders to models located
elsewhere on disk and they will be autoimported. You can also create
subfolders and organize them as you wish.

The location of the autoimport directories are controlled by settings
in `invokeai.yaml`. See [Configuration](../features/CONFIGURATION.md).
