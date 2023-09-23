---
title: Command-line Utilities
---

# :material-file-document: Utilities

# Command-line Utilities

InvokeAI comes with several scripts that are accessible via the
command line. To access these commands, start the "developer's
console" from the launcher (`invoke.bat` menu item [8]). Users who are
familiar with Python can alternatively activate InvokeAI's virtual
environment (typically, but not necessarily `invokeai/.venv`).

In the developer's console, type the script's name to run it. To get a
synopsis of what a utility does and the command-line arguments it
accepts, pass it the `-h` argument, e.g.

```bash
invokeai-merge -h
```
## **invokeai-web**

This script launches the web server and is effectively identical to
selecting option [1] in the launcher. An advantage of launching the
server from the command line is that you can override any setting
configuration option in `invokeai.yaml` using like-named command-line
arguments. For example, to temporarily change the size of the RAM
cache to 7 GB, you can launch as follows:

```bash
invokeai-web --ram 7
```

## **invokeai-merge**

This is the model merge script, the same as launcher option [4]. Call
it with the `--gui` command-line argument to start the interactive
console-based GUI. Alternatively, you can run it non-interactively
using command-line arguments as illustrated in the example below which
merges models named `stable-diffusion-1.5` and `inkdiffusion` into a new model named
`my_new_model`:

```bash
invokeai-merge --force --base-model sd-1 --models stable-diffusion-1.5 inkdiffusion --merged_model_name my_new_model
```

## **invokeai-ti**

This is the textual inversion training script that is run by launcher
option [3]. Call it with `--gui` to run the interactive console-based
front end. It can also be run non-interactively. It has about a
zillion arguments, but a typical training session can be launched
with:

```bash
invokeai-ti --model stable-diffusion-1.5 \
            --placeholder_token 'jello' \
            --learnable_property object \
			--num_train_epochs 50 \
			--train_data_dir /path/to/training/images \
			--output_dir /path/to/trained/model
```

(Note that \\ is the Linux/Mac long-line continuation character. Use ^
in Windows).

## **invokeai-install**

This is the console-based model install script that is run by launcher
option [5]. If called without arguments, it will launch the
interactive console-based interface. It can also be used
non-interactively to list, add and remove models as shown by these
examples:

* This will download and install three models from CivitAI, HuggingFace,
and local disk:

```bash
invokeai-install --add https://civitai.com/api/download/models/161302 ^
                  gsdf/Counterfeit-V3.0  ^
				  D:\Models\merge_model_two.safetensors
```
(Note that ^ is the Windows long-line continuation character. Use \\ on
Linux/Mac).

* This will list installed models of type `main`:

```bash
invokeai-model-install --list-models main
```

* This will delete the models named `voxel-ish` and `realisticVision`:

```bash
invokeai-model-install --delete voxel-ish realisticVision
```

## **invokeai-configure**

This is the console-based configure script that ran when InvokeAI was
first installed. You can run it again at any time to change the
configuration, repair a broken install.

Called without any arguments, `invokeai-configure` enters interactive
mode with two screens. The first screen is a form that provides access
to most of InvokeAI's configuration options. The second screen lets
you download, add, and delete models interactively. When you exit the
second screen, the script will add any missing "support models"
needed for core functionality, and any selected "sd weights" which are
the model checkpoint/diffusers files.

This behavior can be changed via a series of command-line
arguments. Here are some of the useful ones:

* `invokeai-configure --skip-sd-weights --skip-support-models`
This will run just the configuration part of the utility, skipping
downloading of support models and stable diffusion weights.

* `invokeai-configure --yes`
This will run the configure script non-interactively. It will set the
configuration options to their default values, install/repair support
models, and download the "recommended" set of SD models.

* `invokeai-configure --yes --default_only` 
This will run the configure script non-interactively. In contrast to
the previous command, it will only download the default SD model,
Stable Diffusion v1.5

* `invokeai-configure --yes --default_only --skip-sd-weights` 
This is similar to the previous command, but will not download any
SD models at all. It is usually used to repair a broken install.

By default, `invokeai-configure` runs on the currently active InvokeAI
root folder. To run it against a different root, pass it the `--root
</path/to/root>` argument.

Lastly, you can use `invokeai-configure` to create a working root
directory entirely from scratch. Assuming you wish to make a root directory
named `InvokeAI-New`, run this command:

```bash
invokeai-configure --root InvokeAI-New --yes --default_only
```
This will create a minimally functional root directory. You can now
launch the web server against it with `invokeai-web --root InvokeAI-New`.

## **invokeai-update**

This is the interactive console-based script that is run by launcher
menu item [9] to update to a new version of InvokeAI. It takes no
command-line arguments.

## **invokeai-metadata**

This is a script which takes a list of InvokeAI-generated images and
outputs their metadata in the same JSON format that you get from the
`</>` button in the Web GUI. For example:

```bash
$ invokeai-metadata ffe2a115-b492-493c-afff-7679aa034b50.png
ffe2a115-b492-493c-afff-7679aa034b50.png:
{
    "app_version": "3.1.0",
    "cfg_scale": 8.0,
    "clip_skip": 0,
    "controlnets": [],
    "generation_mode": "sdxl_txt2img",
    "height": 1024,
    "loras": [],
    "model": {
        "base_model": "sdxl",
        "model_name": "stable-diffusion-xl-base-1.0",
        "model_type": "main"
    },
    "negative_prompt": "",
    "negative_style_prompt": "",
    "positive_prompt": "military grade sushi dinner for shock troopers",
    "positive_style_prompt": "",
    "rand_device": "cpu",
    "refiner_cfg_scale": 7.5,
    "refiner_model": {
        "base_model": "sdxl-refiner",
        "model_name": "sd_xl_refiner_1.0",
        "model_type": "main"
    },
    "refiner_negative_aesthetic_score": 2.5,
    "refiner_positive_aesthetic_score": 6.0,
    "refiner_scheduler": "euler",
    "refiner_start": 0.8,
    "refiner_steps": 20,
    "scheduler": "euler",
    "seed": 387129902,
    "steps": 25,
    "width": 1024
}
```

You may list multiple files on the command line.

## **invokeai-import-images**

InvokeAI uses a database to store information about images it
generated, and just copying the image files from one InvokeAI root
directory to another does not automatically import those images into
the destination's gallery. This script allows you to bulk import
images generated by one instance of InvokeAI into a gallery maintained
by another. It also works on images generated by older versions of
InvokeAI, going way back to version 1.

This script has an interactive mode only. The following example shows
it in action:

```bash
$ invokeai-import-images
===============================================================================
This script will import images generated by earlier versions of
InvokeAI into the currently installed root directory:
   /home/XXXX/invokeai-main
If this is not what you want to do, type ctrl-C now to cancel.
===============================================================================
= Configuration & Settings
Found invokeai.yaml file at /home/XXXX/invokeai-main/invokeai.yaml:
  Database : /home/XXXX/invokeai-main/databases/invokeai.db
  Outputs  : /home/XXXX/invokeai-main/outputs/images
 
Use these paths for import (yes) or choose different ones (no) [Yn]:
Inputs: Specify absolute path containing InvokeAI .png images to import: /home/XXXX/invokeai-2.3/outputs/images/
Include files from subfolders recursively [yN]?

Options for board selection for imported images:
1) Select an existing board name. (found 4)
2) Specify a board name to create/add to.
3) Create/add to board named 'IMPORT'.
4) Create/add to board named 'IMPORT' with the current datetime string appended (.e.g IMPORT_20230919T203519Z).
5) Create/add to board named 'IMPORT' with a the original file app_version appended (.e.g IMPORT_2.2.5).
Specify desired board option: 3

===============================================================================
= Import Settings Confirmation

Database File Path               : /home/XXXX/invokeai-main/databases/invokeai.db
Outputs/Images Directory         : /home/XXXX/invokeai-main/outputs/images
Import Image Source Directory    : /home/XXXX/invokeai-2.3/outputs/images/
  Recurse Source SubDirectories  : No
Count of .png file(s) found      : 5785
Board name option specified      : IMPORT
Database backup will be taken at : /home/XXXX/invokeai-main/databases/backup

Notes about the import process:
- Source image files will not be modified, only copied to the outputs directory.
- If the same file name already exists in the destination, the file will be skipped.
- If the same file name already has a record in the database, the file will be skipped.
- Invoke AI metadata tags will be updated/written into the imported copy only.
- On the imported copy, only Invoke AI known tags (latest and legacy) will be retained (dream, sd-metadata, invokeai, invokeai_metadata)
- A property 'imported_app_version' will be added to metadata that can be viewed in the UI's metadata viewer.
- The new 3.x InvokeAI outputs folder structure is flat so recursively found source imges will all be placed into the single outputs/images folder.
 
Do you wish to continue with the import [Yn] ?

Making DB Backup at /home/lstein/invokeai-main/databases/backup/backup-20230919T203519Z-invokeai.db...Done!

===============================================================================
Importing /home/XXXX/invokeai-2.3/outputs/images/17d09907-297d-4db3-a18a-60b337feac66.png
... (5785 more lines) ...
===============================================================================
= Import Complete - Elpased Time: 0.28 second(s)

Source File(s)                          : 5785
Total Imported                          : 5783
Skipped b/c file already exists on disk : 1
Skipped b/c file already exists in db   : 0
Errors during import                    : 1
```
## **invokeai-db-maintenance**

This script helps maintain the integrity of your InvokeAI database by
finding and fixing three problems that can arise over time:

1. An image was manually deleted from the outputs directory, leaving a
   dangling image record in the InvokeAI database. This will cause a
   black image to appear in the gallery. This is an "orphaned database
   image record." The script can fix this by running a "clean"
   operation on the database, removing the orphaned entries.
   
2. An image is present in the outputs directory but there is no
   corresponding entry in the database. This can happen when the image
   is added manually to the outputs directory, or if a crash occurred
   after the image was generated but before the database was
   completely updated. The symptom is that the image is present in the
   outputs folder but doesn't appear in the InvokeAI gallery. This is
   called an "orphaned image file." The script can fix this problem by
   running an "archive" operation in which orphaned files are moved
   into a directory named `outputs/images-archive`. If you wish, you
   can then run `invokeai-image-import` to reimport these images back
   into the database.
   
3. The thumbnail for an image is missing, again causing a black
   gallery thumbnail. This is fixed by running the "thumbnaiils"
   operation, which simply regenerates and re-registers the missing
   thumbnail.
   
You can find and fix all three of these problems in a single go by
executing this command:

```bash
invokeai-db-maintenance --operation all
```

Or you can run just the clean and thumbnail operations like this:

```bash
invokeai-db-maintenance -operation clean, thumbnail
```

If called without any arguments, the script will ask you which
operations you wish to perform.

## **invokeai-migrate3**

This script will migrate settings and models (but not images!) from an
InvokeAI v2.3 root folder to an InvokeAI 3.X folder. Call it with the
source and destination root folders like this:

```bash
invokeai-migrate3 --from ~/invokeai-2.3 --to invokeai-3.1.1
```

Both directories must previously have been properly created and
initialized by `invokeai-configure`. If you wish to migrate the images
contained in the older root as well, you can use the
`invokeai-image-migrate` script described earlier.

---

Copyright (c) 2023, Lincoln Stein and the InvokeAI Development Team
