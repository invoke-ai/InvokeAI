---
title: Low-Rank Adaptation (LoRA) Models
---

# :material-library-shelves: Using Low-Rank Adaptation (LoRA) Models

## Introduction

LoRA is a technique for fine-tuning Stable Diffusion models using much
less time and memory than traditional training techniques. The
resulting model files are much smaller than full model files, and can
be used to generate specialized styles and subjects.

LoRAs are built on top of Stable Diffusion v1.x or 2.x checkpoint or
diffusers models. To load a LoRA, you include its name in the text
prompt using a simple syntax described below. While you will generally
get the best results when you use the same model the LoRA was trained
on, they will work to a greater or lesser extent with other models.
The major caveat is that a LoRA built on top of a SD v1.x model cannot
be used with a v2.x model, and vice-versa. If you try, you will get an
error! You may refer to multiple LoRAs in your prompt.

When you apply a LoRA in a prompt you can specify a weight. The higher
the weight, the more influence it will have on the image. Useful
ranges for weights are usually in the 0.0 to 1.0 range (with ranges
between 0.5 and 1.0 being most typical). However you can specify a
higher weight if you wish. Like models, each LoRA has a slightly
different useful weight range and will interact with other generation
parameters such as the CFG, step count and sampler. The author of the
LoRA will often provide guidance on the best settings, but feel free
to experiment. Be aware that it often helps to reduce the CFG value
when using LoRAs.

## Installing LoRAs

This is very easy! Download a LoRA model file from your favorite site
(e.g. [CIVITAI](https://civitai.com) and place it in the `loras`
folder in the InvokeAI root directory (usually `~invokeai/loras` on
Linux/Macintosh machines, and `C:\Users\your-name\invokeai/loras` on
Windows systems). If the `loras` folder does not already exist, just
create it. The vast majority of LoRA models use the Kohya file format,
which is a type of `.safetensors` file.

!!! warning "LoRA Naming Restrictions"

    InvokeAI will only recognize LoRA files that contain the
    characters a-z, A-Z, 0-9 and the underscore character
    _. Other characters, including the hyphen, will cause the
    LoRA file not to load. These naming restrictions may be
    relaxed in the future, but for now you will need to rename
    files that contain hyphens, commas, brackets, and other
    non-word characters.

You may change where InvokeAI looks for the `loras` folder by passing the
`--lora_directory` option to the `invoke.sh`/`invoke.bat` launcher, or
by placing the option in `invokeai.init`. For example:

```
invoke.sh --lora_directory=C:\Users\your-name\SDModels\lora
```

## Using a LoRA in your prompt

To activate a LoRA use the syntax `withLora(my-lora-name,weight)`
somewhere in the text of the prompt. The position doesn't matter; use
whatever is most comfortable for you.

For example, if you have a LoRA named `parchment_people.safetensors`
in your `loras` directory, you can load it with a weight of 0.9 with a
prompt like this one:

```
family sitting at dinner table withLora(parchment_people,0.9)
```

Add additional `withLora()` phrases to load more LoRAs.

You may omit the weight entirely to default to a weight of 1.0:

```
family sitting at dinner table withLora(parchment_people)
```

If you watch the console as your prompt executes, you will see
messages relating to the loading and execution of the LoRA. If things
don't work as expected, note down the console messages and report them
on the InvokeAI Issues pages or Discord channel.

That's pretty much all you need to know!

## Training Kohya Models

InvokeAI cannot currently train LoRA models, but it can load and use
existing LoRA ones to generate images. While there are several LoRA
model file formats, the predominant one is ["Kohya"
format](https://github.com/kohya-ss/sd-scripts), written by [Kohya
S.](https://github.com/kohya-ss). InvokeAI provides support for this
format. For creating your own Kohya models, we recommend the Windows
GUI written by former InvokeAI-team member
[bmaltais](https://github.com/bmaltais), which can be found at
[kohya_ss](https://github.com/bmaltais/kohya_ss).

We can also recommend the [HuggingFace DreamBooth Training
UI](https://huggingface.co/spaces/lora-library/LoRA-DreamBooth-Training-UI),
a paid service that supports both Textual Inversion and LoRA training.

You may also be interested in [Textual
Inversion](TEXTUAL_INVERSION.md) training, which is supported by
InvokeAI as a text console and command-line tool.

