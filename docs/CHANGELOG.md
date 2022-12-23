---
title: Changelog
---

# :octicons-log-16: **Changelog**

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
  [Manual Installation](installation/INSTALL_MANUAL.md).
- The ability to save frequently-used startup options (model to load, steps,
  sampler, etc) in a `.invokeai` file. See
  [Client](features/CLI.md)
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
- Support for [inpainting](features/INPAINTING.md) and
  [outpainting](features/OUTPAINTING.md)
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
  [larger images to be created without duplicating elements](features/CLI.md#this-is-an-example-of-txt2img),
  at the cost of some performance.
- New `--perlin` and `--threshold` options allow you to add and control
  variation during image generation (see
  [Thresholding and Perlin Noise Initialization](features/OTHER.md#thresholding-and-perlin-noise-initialization-options))
- Extensive metadata now written into PNG files, allowing reliable regeneration
  of images and tweaking of previous settings.
- Command-line completion in `invoke.py` now works on Windows, Linux and Mac
  platforms.
- Improved [command-line completion behavior](features/CLI.md) New commands
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
