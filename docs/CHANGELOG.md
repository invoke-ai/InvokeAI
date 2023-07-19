---
title: Changelog
---

# :octicons-log-16: **Changelog**

## v2.3.5 <small>(22 May 2023)</small>

This release (along with the post1 and post2 follow-on releases) expands support for additional LoRA and LyCORIS models, upgrades diffusers versions, and fixes a few bugs.

### LoRA and LyCORIS Support Improvement

    A number of LoRA/LyCORIS fine-tune files (those which alter the text encoder as well as the unet model) were not having the desired effect in InvokeAI. This bug has now been fixed. Full documentation of LoRA support is available at InvokeAI LoRA Support.
    Previously, InvokeAI did not distinguish between LoRA/LyCORIS models based on Stable Diffusion v1.5 vs those based on v2.0 and 2.1, leading to a crash when an incompatible model was loaded. This has now been fixed. In addition, the web pulldown menus for LoRA and Textual Inversion selection have been enhanced to show only those files that are compatible with the currently-selected Stable Diffusion model.
    Support for the newer LoKR LyCORIS files has been added.

### Library Updates and Speed/Reproducibility Advancements
The major enhancement in this version is that NVIDIA users no longer need to decide between speed and reproducibility. Previously, if you activated the Xformers library, you would see improvements in speed and memory usage, but multiple images generated with the same seed and other parameters would be slightly different from each other. This is no longer the case. Relative to 2.3.5 you will see improved performance when running without Xformers, and even better performance when Xformers is activated. In both cases, images generated with the same settings will be identical.

Here are the new library versions:
Library 	Version
Torch 	2.0.0
Diffusers 	0.16.1
Xformers 	0.0.19
Compel 	1.1.5
Other Improvements

### Performance Improvements

    When a model is loaded for the first time, InvokeAI calculates its checksum for incorporation into the PNG metadata. This process could take up to a minute on network-mounted disks and WSL mounts. This release noticeably speeds up the process.

### Bug Fixes

    The "import models from directory" and "import from URL" functionality in the console-based model installer has now been fixed.
    When running the WebUI, we have reduced the number of times that InvokeAI reaches out to HuggingFace to fetch the list of embeddable Textual Inversion models. We have also caught and fixed a problem with the updater not correctly detecting when another instance of the updater is running 


## v2.3.4 <small>(7 April 2023)</small>

What's New in 2.3.4

This features release adds support for LoRA (Low-Rank Adaptation) and LyCORIS (Lora beYond Conventional) models, as well as some minor bug fixes.
### LoRA and LyCORIS Support

LoRA files contain fine-tuning weights that enable particular styles, subjects or concepts to be applied to generated images. LyCORIS files are an extended variant of LoRA. InvokeAI supports the most common LoRA/LyCORIS format, which ends in the suffix .safetensors. You will find numerous LoRA and LyCORIS models for download at Civitai, and a small but growing number at Hugging Face. Full documentation of LoRA support is available at InvokeAI LoRA Support.( Pre-release note: this page will only be available after release)

To use LoRA/LyCORIS models in InvokeAI:

    Download the .safetensors files of your choice and place in /path/to/invokeai/loras. This directory was not present in earlier version of InvokeAI but will be created for you the first time you run the command-line or web client. You can also create the directory manually.

    Add withLora(lora-file,weight) to your prompts. The weight is optional and will default to 1.0. A few examples, assuming that a LoRA file named loras/sushi.safetensors is present:

family sitting at dinner table eating sushi withLora(sushi,0.9)
family sitting at dinner table eating sushi withLora(sushi, 0.75)
family sitting at dinner table eating sushi withLora(sushi)

Multiple withLora() prompt fragments are allowed. The weight can be arbitrarily large, but the useful range is roughly 0.5 to 1.0. Higher weights make the LoRA's influence stronger. Negative weights are also allowed, which can lead to some interesting effects.

    Generate as you usually would! If you find that the image is too "crisp" try reducing the overall CFG value or reducing individual LoRA weights. As is the case with all fine-tunes, you'll get the best results when running the LoRA on top of the model similar to, or identical with, the one that was used during the LoRA's training. Don't try to load a SD 1.x-trained LoRA into a SD 2.x model, and vice versa. This will trigger a non-fatal error message and generation will not proceed.

    You can change the location of the loras directory by passing the --lora_directory option to `invokeai.

### New WebUI LoRA and Textual Inversion Buttons

This version adds two new web interface buttons for inserting LoRA and Textual Inversion triggers into the prompt as shown in the screenshot below.

Clicking on one or the other of the buttons will bring up a menu of available LoRA/LyCORIS or Textual Inversion trigger terms. Select a menu item to insert the properly-formatted withLora() or <textual-inversion> prompt fragment into the positive prompt. The number in parentheses indicates the number of trigger terms currently in the prompt. You may click the button again and deselect the LoRA or trigger to remove it from the prompt, or simply edit the prompt directly.

Currently terms are inserted into the positive prompt textbox only. However, some textual inversion embeddings are designed to be used with negative prompts. To move a textual inversion trigger into the negative prompt, simply cut and paste it.

By default the Textual Inversion menu only shows locally installed models found at startup time in /path/to/invokeai/embeddings. However, InvokeAI has the ability to dynamically download and install additional Textual Inversion embeddings from the HuggingFace Concepts Library. You may choose to display the most popular of these (with five or more likes) in the Textual Inversion menu by going to Settings and turning on "Show Textual Inversions from HF Concepts Library." When this option is activated, the locally-installed TI embeddings will be shown first, followed by uninstalled terms from Hugging Face. See The Hugging Face Concepts Library and Importing Textual Inversion files for more information.
### Minor features and fixes

This release changes model switching behavior so that the command-line and Web UIs save the last model used and restore it the next time they are launched. It also improves the behavior of the installer so that the pip utility is kept up to date.
  
### Known Bugs in 2.3.4

These are known bugs in the release.

    The Ancestral DPMSolverMultistepScheduler (k_dpmpp_2a) sampler is not yet implemented for diffusers models and will disappear from the WebUI Sampler menu when a diffusers model is selected.
    Windows Defender will sometimes raise Trojan or backdoor alerts for the codeformer.pth face restoration model, as well as the CIDAS/clipseg and runwayml/stable-diffusion-v1.5 models. These are false positives and can be safely ignored. InvokeAI performs a malware scan on all models as they are loaded. For additional security, you should use safetensors models whenever they are available.


## v2.3.3 <small>(28 March 2023)</small>

This is a bugfix and minor feature release.
### Bugfixes

Since version 2.3.2 the following bugs have been fixed:
Bugs

    When using legacy checkpoints with an external VAE, the VAE file is now scanned for malware prior to loading. Previously only the main model weights file was scanned.
    Textual inversion will select an appropriate batchsize based on whether xformers is active, and will default to xformers enabled if the library is detected.
    The batch script log file names have been fixed to be compatible with Windows.
    Occasional corruption of the .next_prefix file (which stores the next output file name in sequence) on Windows systems is now detected and corrected.
    Support loading of legacy config files that have no personalization (textual inversion) section.
    An infinite loop when opening the developer's console from within the invoke.sh script has been corrected.
    Documentation fixes, including a recipe for detecting and fixing problems with the AMD GPU ROCm driver.

Enhancements

    It is now possible to load and run several community-contributed SD-2.0 based models, including the often-requested "Illuminati" model.
    The "NegativePrompts" embedding file, and others like it, can now be loaded by placing it in the InvokeAI embeddings directory.
    If no --model is specified at launch time, InvokeAI will remember the last model used and restore it the next time it is launched.
    On Linux systems, the invoke.sh launcher now uses a prettier console-based interface. To take advantage of it, install the dialog package using your package manager (e.g. sudo apt install dialog).
    When loading legacy models (safetensors/ckpt) you can specify a custom config file and/or a VAE by placing like-named files in the same directory as the model following this example:

my-favorite-model.ckpt
my-favorite-model.yaml
my-favorite-model.vae.pt      # or my-favorite-model.vae.safetensors

### Known Bugs in 2.3.3

These are known bugs in the release.

    The Ancestral DPMSolverMultistepScheduler (k_dpmpp_2a) sampler is not yet implemented for diffusers models and will disappear from the WebUI Sampler menu when a diffusers model is selected.
    Windows Defender will sometimes raise Trojan or backdoor alerts for the codeformer.pth face restoration model, as well as the CIDAS/clipseg and runwayml/stable-diffusion-v1.5 models. These are false positives and can be safely ignored. InvokeAI performs a malware scan on all models as they are loaded. For additional security, you should use safetensors models whenever they are available.


## v2.3.2 <small>(11 March 2023)</small>
This is a bugfix and minor feature release.

### Bugfixes

Since version 2.3.1 the following bugs have been fixed:

    Black images appearing for potential NSFW images when generating with legacy checkpoint models and both --no-nsfw_checker and --ckpt_convert turned on.
    Black images appearing when generating from models fine-tuned on Stable-Diffusion-2-1-base. When importing V2-derived models, you may be asked to select whether the model was derived from a "base" model (512 pixels) or the 768-pixel SD-2.1 model.
    The "Use All" button was not restoring the Hi-Res Fix setting on the WebUI
    When using the model installer console app, models failed to import correctly when importing from directories with spaces in their names. A similar issue with the output directory was also fixed.
    Crashes that occurred during model merging.
    Restore previous naming of Stable Diffusion base and 768 models.
    Upgraded to latest versions of diffusers, transformers, safetensors and accelerate libraries upstream. We hope that this will fix the assertion NDArray > 2**32 issue that MacOS users have had when generating images larger than 768x768 pixels. Please report back.

As part of the upgrade to diffusers, the location of the diffusers-based models has changed from models/diffusers to models/hub. When you launch InvokeAI for the first time, it will prompt you to OK a one-time move. This should be quick and harmless, but if you have modified your models/diffusers directory in some way, for example using symlinks, you may wish to cancel the migration and make appropriate adjustments.
New "Invokeai-batch" script

### Invoke AI Batch
2.3.2 introduces a new command-line only script called invokeai-batch that can be used to generate hundreds of images from prompts and settings that vary systematically. This can be used to try the same prompt across multiple combinations of models, steps, CFG settings and so forth. It also allows you to template prompts and generate a combinatorial list like:

a shack in the mountains, photograph
a shack in the mountains, watercolor
a shack in the mountains, oil painting
a chalet in the mountains, photograph
a chalet in the mountains, watercolor
a chalet in the mountains, oil painting
a shack in the desert, photograph
...

If you have a system with multiple GPUs, or a single GPU with lots of VRAM, you can parallelize generation across the combinatorial set, reducing wait times and using your system's resources efficiently (make sure you have good GPU cooling).

To try invokeai-batch out. Launch the "developer's console" using the invoke launcher script, or activate the invokeai virtual environment manually. From the console, give the command invokeai-batch --help in order to learn how the script works and create your first template file for dynamic prompt generation.


### Known Bugs in 2.3.2

These are known bugs in the release.

    The Ancestral DPMSolverMultistepScheduler (k_dpmpp_2a) sampler is not yet implemented for diffusers models and will disappear from the WebUI Sampler menu when a diffusers model is selected.
    Windows Defender will sometimes raise a Trojan alert for the codeformer.pth face restoration model. As far as we have been able to determine, this is a false positive and can be safely whitelisted.


## v2.3.1 <small>(22 February 2023)</small>
This is primarily a bugfix release, but it does provide several new features that will improve the user experience.

### Enhanced support for model management

InvokeAI now makes it convenient to add, remove and modify models. You can individually import models that are stored on your local system, scan an entire folder and its subfolders for models and import them automatically, and even directly import models from the internet by providing their download URLs. You also have the option of designating a local folder to scan for new models each time InvokeAI is restarted.

There are three ways of accessing the model management features:

    From the WebUI, click on the cube to the right of the model selection menu. This will bring up a form that allows you to import models individually from your local disk or scan a directory for models to import.

    Using the Model Installer App

Choose option (5) download and install models from the invoke launcher script to start a new console-based application for model management. You can use this to select from a curated set of starter models, or import checkpoint, safetensors, and diffusers models from a local disk or the internet. The example below shows importing two checkpoint URLs from popular SD sites and a HuggingFace diffusers model using its Repository ID. It also shows how to designate a folder to be scanned at startup time for new models to import.

Command-line users can start this app using the command invokeai-model-install.

    Using the Command Line Client (CLI)

The !install_model and !convert_model commands have been enhanced to allow entering of URLs and local directories to scan and import. The first command installs .ckpt and .safetensors files as-is. The second one converts them into the faster diffusers format before installation.

Internally InvokeAI is able to probe the contents of a .ckpt or .safetensors file to distinguish among v1.x, v2.x and inpainting models. This means that you do not need to include "inpaint" in your model names to use an inpainting model. Note that Stable Diffusion v2.x models will be autoconverted into a diffusers model the first time you use it.

Please see INSTALLING MODELS for more information on model management.

### An Improved Installer Experience

The installer now launches a console-based UI for setting and changing commonly-used startup options:

After selecting the desired options, the installer installs several support models needed by InvokeAI's face reconstruction and upscaling features and then launches the interface for selecting and installing models shown earlier. At any time, you can edit the startup options by launching invoke.sh/invoke.bat and entering option (6) change InvokeAI startup options

Command-line users can launch the new configure app using invokeai-configure.

This release also comes with a renewed updater. To do an update without going through a whole reinstallation, launch invoke.sh or invoke.bat and choose option (9) update InvokeAI . This will bring you to a screen that prompts you to update to the latest released version, to the most current development version, or any released or unreleased version you choose by selecting the tag or branch of the desired version.

Command-line users can run this interface by typing invokeai-configure

### Image Symmetry Options

There are now features to generate horizontal and vertical symmetry during generation. The way these work is to wait until a selected step in the generation process and then to turn on a mirror image effect. In addition to generating some cool images, you can also use this to make side-by-side comparisons of how an image will look with more or fewer steps. Access this option from the WebUI by selecting Symmetry from the image generation settings, or within the CLI by using the options --h_symmetry_time_pct and --v_symmetry_time_pct (these can be abbreviated to --h_sym and --v_sym like all other options).

### A New Unified Canvas Look

This release introduces a beta version of the WebUI Unified Canvas. To try it out, open up the settings dialogue in the WebUI (gear icon) and select Use Canvas Beta Layout:

Refresh the screen and go to to Unified Canvas (left side of screen, third icon from the top). The new layout is designed to provide more space to work in and to keep the image controls close to the image itself:

Model conversion and merging within the WebUI

The WebUI now has an intuitive interface for model merging, as well as for permanent conversion of models from legacy .ckpt/.safetensors formats into diffusers format. These options are also available directly from the invoke.sh/invoke.bat scripts.
An easier way to contribute translations to the WebUI

We have migrated our translation efforts to Weblate, a FOSS translation product. Maintaining the growing project's translations is now far simpler for the maintainers and community. Please review our brief translation guide for more information on how to contribute.
Numerous internal bugfixes and performance issues

### Bug Fixes
This releases quashes multiple bugs that were reported in 2.3.0. Major internal changes include upgrading to diffusers 0.13.0, and using the compel library for prompt parsing. See Detailed Change Log for a detailed list of bugs caught and squished.
Summary of InvokeAI command line scripts (all accessible via the launcher menu)
Command 	Description
invokeai 	Command line interface
invokeai --web 	Web interface
invokeai-model-install 	Model installer with console forms-based front end
invokeai-ti --gui 	Textual inversion, with a console forms-based front end
invokeai-merge --gui 	Model merging, with a console forms-based front end
invokeai-configure 	Startup configuration; can also be used to reinstall support models
invokeai-update 	InvokeAI software updater

### Known Bugs in 2.3.1

These are known bugs in the release.
    MacOS users generating 768x768 pixel images or greater using diffusers models may experience a hard crash with assertion NDArray > 2**32 This appears to be an issu...



## v2.3.0 <small>(15 January 2023)</small>

**Transition to diffusers

Version 2.3 provides support for both the traditional `.ckpt` weight
checkpoint files as well as the HuggingFace `diffusers` format. This
introduces several changes you should know about.

1. The models.yaml format has been updated. There are now two
   different type of configuration stanza. The traditional ckpt
   one will look like this, with a `format` of `ckpt` and a
   `weights` field that points to the absolute or ROOTDIR-relative
   location of the ckpt file.

   ```
   inpainting-1.5:
      description: RunwayML SD 1.5 model optimized for inpainting (4.27 GB)
      repo_id: runwayml/stable-diffusion-inpainting
      format: ckpt
      width: 512
      height: 512
      weights: models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt
      config: configs/stable-diffusion/v1-inpainting-inference.yaml
      vae: models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
   ```

  A configuration stanza for a diffusers model hosted at HuggingFace will look like this,
  with a `format` of `diffusers` and a `repo_id` that points to the
  repository ID of the model on HuggingFace:

  ```
  stable-diffusion-2.1:
  description: Stable Diffusion version 2.1 diffusers model (5.21 GB)
  repo_id: stabilityai/stable-diffusion-2-1
  format: diffusers
  ```

  A configuration stanza for a diffuers model stored locally should
  look like this, with a `format` of `diffusers`, but a `path` field
  that points at the directory that contains `model_index.json`:

  ```
  waifu-diffusion:
  description: Latest waifu diffusion 1.4
  format: diffusers
  path: models/diffusers/hakurei-haifu-diffusion-1.4
  ```

2. In order of precedence, InvokeAI will now use HF_HOME, then
   XDG_CACHE_HOME, then finally default to `ROOTDIR/models` to
   store HuggingFace diffusers models.

   Consequently, the format of the models directory has changed to
   mimic the HuggingFace cache directory. When HF_HOME and XDG_HOME
   are not set, diffusers models are now automatically downloaded
   and retrieved from the directory `ROOTDIR/models/diffusers`,
   while other models are stored in the directory
   `ROOTDIR/models/hub`. This organization is the same as that used
   by HuggingFace for its cache management.

   This allows you to share diffusers and ckpt model files easily with
   other machine learning applications that use the HuggingFace
   libraries. To do this, set the environment variable HF_HOME
   before starting up InvokeAI to tell it what directory to
   cache models in. To tell InvokeAI to use the standard HuggingFace
   cache directory, you would set HF_HOME like this (Linux/Mac):

   `export HF_HOME=~/.cache/huggingface`

   Both HuggingFace and InvokeAI will fall back to the XDG_CACHE_HOME
   environment variable if HF_HOME is not set; this path
   takes precedence over `ROOTDIR/models` to allow for the same sharing
   with other machine learning applications that use HuggingFace
   libraries.

3. If you upgrade to InvokeAI 2.3.* from an earlier version, there
   will be a one-time migration from the old models directory format
   to the new one. You will see a message about this the first time
   you start `invoke.py`.

4. Both the front end back ends of the model manager have been
   rewritten to accommodate diffusers. You can import models using
   their local file path, using their URLs, or their HuggingFace
   repo_ids. On the command line, all these syntaxes work:

   ```
   !import_model stabilityai/stable-diffusion-2-1-base
   !import_model /opt/sd-models/sd-1.4.ckpt
   !import_model https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model/blob/main/PaperCut_v1.ckpt
   ```

**KNOWN BUGS (15 January 2023)

1. On CUDA systems, the 768 pixel stable-diffusion-2.0 and
   stable-diffusion-2.1 models can only be run as `diffusers` models
   when the `xformer` library is installed and configured. Without
   `xformers`, InvokeAI returns black images.

2. Inpainting and outpainting have regressed in quality.

Both these issues are being actively worked on.

## v2.2.4 <small>(11 December 2022)</small>

**the `invokeai` directory**

Previously there were two directories to worry about, the directory that
contained the InvokeAI source code and the launcher scripts, and the `invokeai`
directory that contained the models files, embeddings, configuration and
outputs. With the 2.2.4 release, this dual system is done away with, and
everything, including the `invoke.bat` and `invoke.sh` launcher scripts, now
live in a directory named `invokeai`. By default this directory is located in
your home directory (e.g. `\Users\yourname` on Windows), but you can select
where it goes at install time.

After installation, you can delete the install directory (the one that the zip
file creates when it unpacks). Do **not** delete or move the `invokeai`
directory!

**Initialization file `invokeai/invokeai.init`**

You can place frequently-used startup options in this file, such as the default
number of steps or your preferred sampler. To keep everything in one place, this
file has now been moved into the `invokeai` directory and is named
`invokeai.init`.

**To update from Version 2.2.3**

The easiest route is to download and unpack one of the 2.2.4 installer files.
When it asks you for the location of the `invokeai` runtime directory, respond
with the path to the directory that contains your 2.2.3 `invokeai`. That is, if
`invokeai` lives at `C:\Users\fred\invokeai`, then answer with `C:\Users\fred`
and answer "Y" when asked if you want to reuse the directory.

The `update.sh` (`update.bat`) script that came with the 2.2.3 source installer
does not know about the new directory layout and won't be fully functional.

**To update to 2.2.5 (and beyond) there's now an update path**

As they become available, you can update to more recent versions of InvokeAI
using an `update.sh` (`update.bat`) script located in the `invokeai` directory.
Running it without any arguments will install the most recent version of
InvokeAI. Alternatively, you can get set releases by running the `update.sh`
script with an argument in the command shell. This syntax accepts the path to
the desired release's zip file, which you can find by clicking on the green
"Code" button on this repository's home page.

**Other 2.2.4 Improvements**

- Fix InvokeAI GUI initialization by @addianto in #1687
- fix link in documentation by @lstein in #1728
- Fix broken link by @ShawnZhong in #1736
- Remove reference to binary installer by @lstein in #1731
- documentation fixes for 2.2.3 by @lstein in #1740
- Modify installer links to point closer to the source installer by @ebr in
  #1745
- add documentation warning about 1650/60 cards by @lstein in #1753
- Fix Linux source URL in installation docs by @andybearman in #1756
- Make install instructions discoverable in readme by @damian0815 in #1752
- typo fix by @ofirkris in #1755
- Non-interactive model download (support HUGGINGFACE_TOKEN) by @ebr in #1578
- fix(srcinstall): shell installer - cp scripts instead of linking by @tildebyte
  in #1765
- stability and usage improvements to binary & source installers by @lstein in
  #1760
- fix off-by-one bug in cross-attention-control by @damian0815 in #1774
- Eventually update APP_VERSION to 2.2.3 by @spezialspezial in #1768
- invoke script cds to its location before running by @lstein in #1805
- Make PaperCut and VoxelArt models load again by @lstein in #1730
- Fix --embedding_directory / --embedding_path not working by @blessedcoolant in
  #1817
- Clean up readme by @hipsterusername in #1820
- Optimized Docker build with support for external working directory by @ebr in
  #1544
- disable pushing the cloud container by @mauwii in #1831
- Fix docker push github action and expand with additional metadata by @ebr in
  #1837
- Fix Broken Link To Notebook by @VedantMadane in #1821
- Account for flat models by @spezialspezial in #1766
- Update invoke.bat.in isolate environment variables by @lynnewu in #1833
- Arch Linux Specific PatchMatch Instructions & fixing conda install on linux by
  @SammCheese in #1848
- Make force free GPU memory work in img2img by @addianto in #1844
- New installer by @lstein

## v2.2.3 <small>(2 December 2022)</small>

!!! Note

    This point release removes references to the binary installer from the
    installation guide. The binary installer is not stable at the current
    time. First time users are encouraged to use the "source" installer as
    described in [Installing InvokeAI with the Source Installer](installation/deprecated_documentation/INSTALL_SOURCE.md)

With InvokeAI 2.2, this project now provides enthusiasts and professionals a
robust workflow solution for creating AI-generated and human facilitated
compositions. Additional enhancements have been made as well, improving safety,
ease of use, and installation.

Optimized for efficiency, InvokeAI needs only ~3.5GB of VRAM to generate a
512x768 image (and less for smaller images), and is compatible with
Windows/Linux/Mac (M1 & M2).

You can see the [release video](https://youtu.be/hIYBfDtKaus) here, which
introduces the main WebUI enhancement for version 2.2 -
[The Unified Canvas](features/UNIFIED_CANVAS.md). This new workflow is the
biggest enhancement added to the WebUI to date, and unlocks a stunning amount of
potential for users to create and iterate on their creations. The following
sections describe what's new for InvokeAI.

## v2.2.2 <small>(30 November 2022)</small>

!!! note

    The binary installer is not ready for prime time. First time users are recommended to install via the "source" installer accessible through the links at the bottom of this page.****

With InvokeAI 2.2, this project now provides enthusiasts and professionals a
robust workflow solution for creating AI-generated and human facilitated
compositions. Additional enhancements have been made as well, improving safety,
ease of use, and installation.

Optimized for efficiency, InvokeAI needs only ~3.5GB of VRAM to generate a
512x768 image (and less for smaller images), and is compatible with
Windows/Linux/Mac (M1 & M2).

You can see the [release video](https://youtu.be/hIYBfDtKaus) here, which
introduces the main WebUI enhancement for version 2.2 -
[The Unified Canvas](https://invoke-ai.github.io/InvokeAI/features/UNIFIED_CANVAS/).
This new workflow is the biggest enhancement added to the WebUI to date, and
unlocks a stunning amount of potential for users to create and iterate on their
creations. The following sections describe what's new for InvokeAI.

## v2.2.0 <small>(2 December 2022)</small>

With InvokeAI 2.2, this project now provides enthusiasts and professionals a
robust workflow solution for creating AI-generated and human facilitated
compositions. Additional enhancements have been made as well, improving safety,
ease of use, and installation.

Optimized for efficiency, InvokeAI needs only ~3.5GB of VRAM to generate a
512x768 image (and less for smaller images), and is compatible with
Windows/Linux/Mac (M1 & M2).

You can see the [release video](https://youtu.be/hIYBfDtKaus) here, which
introduces the main WebUI enhancement for version 2.2 -
[The Unified Canvas](features/UNIFIED_CANVAS.md). This new workflow is the
biggest enhancement added to the WebUI to date, and unlocks a stunning amount of
potential for users to create and iterate on their creations. The following
sections describe what's new for InvokeAI.

## v2.1.3 <small>(13 November 2022)</small>

- A choice of installer scripts that automate installation and configuration.
  See
  [Installation](installation/index.md).
- A streamlined manual installation process that works for both Conda and
  PIP-only installs. See
  [Manual Installation](installation/020_INSTALL_MANUAL.md).
- The ability to save frequently-used startup options (model to load, steps,
  sampler, etc) in a `.invokeai` file. See
  [Client](deprecated/CLI.md)
- Support for AMD GPU cards (non-CUDA) on Linux machines.
- Multiple bugs and edge cases squashed.

## v2.1.0 <small>(2 November 2022)</small>

- update mac instructions to use invokeai for env name by @willwillems in #1030
- Update .gitignore by @blessedcoolant in #1040
- reintroduce fix for m1 from #579 missing after merge by @skurovec in #1056
- Update Stable_Diffusion_AI_Notebook.ipynb (Take 2) by @ChloeL19 in #1060
- Print out the device type which is used by @manzke in #1073
- Hires Addition by @hipsterusername in #1063
- fix for "1 leaked semaphore objects to clean up at shutdown" on M1 by
  @skurovec in #1081
- Forward dream.py to invoke.py using the same interpreter, add deprecation
  warning by @db3000 in #1077
- fix noisy images at high step counts by @lstein in #1086
- Generalize facetool strength argument by @db3000 in #1078
- Enable fast switching among models at the invoke> command line by @lstein in
  #1066
- Fix Typo, committed changing ldm environment to invokeai by @jdries3 in #1095
- Update generate.py by @unreleased in #1109
- Update 'ldm' env to 'invokeai' in troubleshooting steps by @19wolf in #1125
- Fixed documentation typos and resolved merge conflicts by @rupeshs in #1123
- Fix broken doc links, fix malaprop in the project subtitle by @majick in #1131
- Only output facetool parameters if enhancing faces by @db3000 in #1119
- Update gitignore to ignore codeformer weights at new location by
  @spezialspezial in #1136
- fix links to point to invoke-ai.github.io #1117 by @mauwii in #1143
- Rework-mkdocs by @mauwii in #1144
- add option to CLI and pngwriter that allows user to set PNG compression level
  by @lstein in #1127
- Fix img2img DDIM index out of bound by @wfng92 in #1137
- Fix gh actions by @mauwii in #1128
- update mac instructions to use invokeai for env name by @willwillems in #1030
- Update .gitignore by @blessedcoolant in #1040
- reintroduce fix for m1 from #579 missing after merge by @skurovec in #1056
- Update Stable_Diffusion_AI_Notebook.ipynb (Take 2) by @ChloeL19 in #1060
- Print out the device type which is used by @manzke in #1073
- Hires Addition by @hipsterusername in #1063
- fix for "1 leaked semaphore objects to clean up at shutdown" on M1 by
  @skurovec in #1081
- Forward dream.py to invoke.py using the same interpreter, add deprecation
  warning by @db3000 in #1077
- fix noisy images at high step counts by @lstein in #1086
- Generalize facetool strength argument by @db3000 in #1078
- Enable fast switching among models at the invoke> command line by @lstein in
  #1066
- Fix Typo, committed changing ldm environment to invokeai by @jdries3 in #1095
- Fixed documentation typos and resolved merge conflicts by @rupeshs in #1123
- Only output facetool parameters if enhancing faces by @db3000 in #1119
- add option to CLI and pngwriter that allows user to set PNG compression level
  by @lstein in #1127
- Fix img2img DDIM index out of bound by @wfng92 in #1137
- Add text prompt to inpaint mask support by @lstein in #1133
- Respect http[s] protocol when making socket.io middleware by @damian0815 in
  #976
- WebUI: Adds Codeformer support by @psychedelicious in #1151
- Skips normalizing prompts for web UI metadata by @psychedelicious in #1165
- Add Asymmetric Tiling by @carson-katri in #1132
- Web UI: Increases max CFG Scale to 200 by @psychedelicious in #1172
- Corrects color channels in face restoration; Fixes #1167 by @psychedelicious
  in #1175
- Flips channels using array slicing instead of using OpenCV by @psychedelicious
  in #1178
- Fix typo in docs: s/Formally/Formerly by @noodlebox in #1176
- fix clipseg loading problems by @lstein in #1177
- Correct color channels in upscale using array slicing by @wfng92 in #1181
- Web UI: Filters existing images when adding new images; Fixes #1085 by
  @psychedelicious in #1171
- fix a number of bugs in textual inversion by @lstein in #1190
- Improve !fetch, add !replay command by @ArDiouscuros in #882
- Fix generation of image with s>1000 by @holstvoogd in #951
- Web UI: Gallery improvements by @psychedelicious in #1198
- Update CLI.md by @krummrey in #1211
- outcropping improvements by @lstein in #1207
- add support for loading VAE autoencoders by @lstein in #1216
- remove duplicate fix_func for MPS by @wfng92 in #1210
- Metadata storage and retrieval fixes by @lstein in #1204
- nix: add shell.nix file by @Cloudef in #1170
- Web UI: Changes vite dist asset paths to relative by @psychedelicious in #1185
- Web UI: Removes isDisabled from PromptInput by @psychedelicious in #1187
- Allow user to generate images with initial noise as on M1 / mps system by
  @ArDiouscuros in #981
- feat: adding filename format template by @plucked in #968
- Web UI: Fixes broken bundle by @psychedelicious in #1242
- Support runwayML custom inpainting model by @lstein in #1243
- Update IMG2IMG.md by @talitore in #1262
- New dockerfile - including a build- and a run- script as well as a GH-Action
  by @mauwii in #1233
- cut over from karras to model noise schedule for higher steps by @lstein in
  #1222
- Prompt tweaks by @lstein in #1268
- Outpainting implementation by @Kyle0654 in #1251
- fixing aspect ratio on hires by @tjennings in #1249
- Fix-build-container-action by @mauwii in #1274
- handle all unicode characters by @damian0815 in #1276
- adds models.user.yml to .gitignore by @JakeHL in #1281
- remove debug branch, set fail-fast to false by @mauwii in #1284
- Protect-secrets-on-pr by @mauwii in #1285
- Web UI: Adds initial inpainting implementation by @psychedelicious in #1225
- fix environment-mac.yml - tested on x64 and arm64 by @mauwii in #1289
- Use proper authentication to download model by @mauwii in #1287
- Prevent indexing error for mode RGB by @spezialspezial in #1294
- Integrate sd-v1-5 model into test matrix (easily expandable), remove
  unecesarry caches by @mauwii in #1293
- add --no-interactive to configure_invokeai step by @mauwii in #1302
- 1-click installer and updater. Uses micromamba to install git and conda into a
  contained environment (if necessary) before running the normal installation
  script by @cmdr2 in #1253
- configure_invokeai.py script downloads the weight files by @lstein in #1290

## v2.0.1 <small>(13 October 2022)</small>

- fix noisy images at high step count when using k\* samplers
- dream.py script now calls invoke.py module directly rather than via a new
  python process (which could break the environment)

## v2.0.0 <small>(9 October 2022)</small>

- `dream.py` script renamed `invoke.py`. A `dream.py` script wrapper remains for
  backward compatibility.
- Completely new WebGUI - launch with `python3 scripts/invoke.py --web`
- img2img runs on all k\* samplers
- Support for
  [negative prompts](features/PROMPTS.md#negative-and-unconditioned-prompts)
- Support for CodeFormer face reconstruction
- Support for Textual Inversion on Macintoshes
- Support in both WebGUI and CLI for
  [post-processing of previously-generated images](features/POSTPROCESS.md)
  using facial reconstruction, ESRGAN upscaling, outcropping (similar to DALL-E
  infinite canvas), and "embiggen" upscaling. See the `!fix` command.
- New `--hires` option on `invoke>` line allows
  [larger images to be created without duplicating elements](deprecated/CLI.md#this-is-an-example-of-txt2img),
  at the cost of some performance.
- New `--perlin` and `--threshold` options allow you to add and control
  variation during image generation (see
  [Thresholding and Perlin Noise Initialization](features/OTHER.md#thresholding-and-perlin-noise-initialization-options))
- Extensive metadata now written into PNG files, allowing reliable regeneration
  of images and tweaking of previous settings.
- Command-line completion in `invoke.py` now works on Windows, Linux and Mac
  platforms.
- Improved [command-line completion behavior](deprecated/CLI.md) New commands
  added:
  - List command-line history with `!history`
  - Search command-line history with `!search`
  - Clear history with `!clear`
- Deprecated `--full_precision` / `-F`. Simply omit it and `invoke.py` will auto
  configure. To switch away from auto use the new flag like
  `--precision=float32`.

## v1.14 <small>(11 September 2022)</small>

- Memory optimizations for small-RAM cards. 512x512 now possible on 4 GB GPUs.
- Full support for Apple hardware with M1 or M2 chips.
- Add "seamless mode" for circular tiling of image. Generates beautiful effects.
  ([prixt](https://github.com/prixt)).
- Inpainting support.
- Improved web server GUI.
- Lots of code and documentation cleanups.

## v1.13 <small>(3 September 2022)</small>

- Support image variations (see [VARIATIONS](features/VARIATIONS.md)
  ([Kevin Gibbons](https://github.com/bakkot) and many contributors and
  reviewers)
- Supports a Google Colab notebook for a standalone server running on Google
  hardware [Arturo Mendivil](https://github.com/artmen1516)
- WebUI supports GFPGAN/ESRGAN facial reconstruction and upscaling
  [Kevin Gibbons](https://github.com/bakkot)
- WebUI supports incremental display of in-progress images during generation
  [Kevin Gibbons](https://github.com/bakkot)
- A new configuration file scheme that allows new models (including upcoming
  stable-diffusion-v1.5) to be added without altering the code.
  ([David Wager](https://github.com/maddavid12))
- Can specify --grid on invoke.py command line as the default.
- Miscellaneous internal bug and stability fixes.
- Works on M1 Apple hardware.
- Multiple bug fixes.

---

## v1.12 <small>(28 August 2022)</small>

- Improved file handling, including ability to read prompts from standard input.
  (kudos to [Yunsaki](https://github.com/yunsaki)
- The web server is now integrated with the invoke.py script. Invoke by adding
  --web to the invoke.py command arguments.
- Face restoration and upscaling via GFPGAN and Real-ESGAN are now automatically
  enabled if the GFPGAN directory is located as a sibling to Stable Diffusion.
  VRAM requirements are modestly reduced. Thanks to both
  [Blessedcoolant](https://github.com/blessedcoolant) and
  [Oceanswave](https://github.com/oceanswave) for their work on this.
- You can now swap samplers on the invoke> command line.
  [Blessedcoolant](https://github.com/blessedcoolant)

---

## v1.11 <small>(26 August 2022)</small>

- NEW FEATURE: Support upscaling and face enhancement using the GFPGAN module.
  (kudos to [Oceanswave](https://github.com/Oceanswave)
- You now can specify a seed of -1 to use the previous image's seed, -2 to use
  the seed for the image generated before that, etc. Seed memory only extends
  back to the previous command, but will work on all images generated with the
  -n# switch.
- Variant generation support temporarily disabled pending more general solution.
- Created a feature branch named **yunsaki-morphing-invoke** which adds
  experimental support for iteratively modifying the prompt and its parameters.
  Please
  see[Pull Request #86](https://github.com/lstein/stable-diffusion/pull/86) for
  a synopsis of how this works. Note that when this feature is eventually added
  to the main branch, it will may be modified significantly.

---

## v1.10 <small>(25 August 2022)</small>

- A barebones but fully functional interactive web server for online generation
  of txt2img and img2img.

---

## v1.09 <small>(24 August 2022)</small>

- A new -v option allows you to generate multiple variants of an initial image
  in img2img mode. (kudos to [Oceanswave](https://github.com/Oceanswave).
  [ See this discussion in the PR for examples and details on use](https://github.com/lstein/stable-diffusion/pull/71#issuecomment-1226700810))
- Added ability to personalize text to image generation (kudos to
  [Oceanswave](https://github.com/Oceanswave) and
  [nicolai256](https://github.com/nicolai256))
- Enabled all of the samplers from k_diffusion

---

## v1.08 <small>(24 August 2022)</small>

- Escape single quotes on the invoke> command before trying to parse. This
  avoids parse errors.
- Removed instruction to get Python3.8 as first step in Windows install.
  Anaconda3 does it for you.
- Added bounds checks for numeric arguments that could cause crashes.
- Cleaned up the copyright and license agreement files.

---

## v1.07 <small>(23 August 2022)</small>

- Image filenames will now never fill gaps in the sequence, but will be assigned
  the next higher name in the chosen directory. This ensures that the alphabetic
  and chronological sort orders are the same.

---

## v1.06 <small>(23 August 2022)</small>

- Added weighted prompt support contributed by
  [xraxra](https://github.com/xraxra)
- Example of using weighted prompts to tweak a demonic figure contributed by
  [bmaltais](https://github.com/bmaltais)

---

## v1.05 <small>(22 August 2022 - after the drop)</small>

- Filenames now use the following formats: 000010.95183149.png -- Two files
  produced by the same command (e.g. -n2), 000010.26742632.png -- distinguished
  by a different seed.

  000011.455191342.01.png -- Two files produced by the same command using
  000011.455191342.02.png -- a batch size>1 (e.g. -b2). They have the same seed.

  000011.4160627868.grid#1-4.png -- a grid of four images (-g); the whole grid
  can be regenerated with the indicated key

- It should no longer be possible for one image to overwrite another
- You can use the "cd" and "pwd" commands at the invoke> prompt to set and
  retrieve the path of the output directory.

---

## v1.04 <small>(22 August 2022 - after the drop)</small>

- Updated README to reflect installation of the released weights.
- Suppressed very noisy and inconsequential warning when loading the frozen CLIP
  tokenizer.

---

## v1.03 <small>(22 August 2022)</small>

- The original txt2img and img2img scripts from the CompViz repository have been
  moved into a subfolder named "orig_scripts", to reduce confusion.

---

## v1.02 <small>(21 August 2022)</small>

- A copy of the prompt and all of its switches and options is now stored in the
  corresponding image in a tEXt metadata field named "Dream". You can read the
  prompt using scripts/images2prompt.py, or an image editor that allows you to
  explore the full metadata. **Please run "conda env update" to load the k_lms
  dependencies!!**

---

## v1.01 <small>(21 August 2022)</small>

- added k_lms sampling. **Please run "conda env update" to load the k_lms
  dependencies!!**
- use half precision arithmetic by default, resulting in faster execution and
  lower memory requirements Pass argument --full_precision to invoke.py to get
  slower but more accurate image generation

---

## Links

- **[Read Me](index.md)**
