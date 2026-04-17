---
title: Canvas Projects
---

# :material-folder-zip: Canvas Projects

## Save and Restore Your Canvas Work

Canvas Projects let you save your entire canvas setup to a file and load it back later. This is useful when you want to:

- **Switch between tasks** without losing your current canvas arrangement
- **Back up complex setups** with multiple layers, masks, and reference images
- **Share canvas layouts** with others or transfer them between machines
- **Recover from deleted images** — all images are embedded in the project file

## What Gets Saved

A canvas project file (`.invk`) captures everything about your current canvas session:

- **All layers** — raster layers, control layers, inpaint masks, regional guidance
- **All drawn content** — brush strokes, pasted images, eraser marks
- **Reference images** — global IP-Adapter / FLUX Redux images with crop settings
- **Regional guidance** — per-region prompts and reference images
- **Bounding box** — position, size, aspect ratio, and scale settings
- **All generation parameters** — prompts, seed, steps, CFG scale, guidance, scheduler, model, VAE, dimensions, img2img strength, infill settings, canvas coherence, refiner settings, FLUX/Z-Image specific parameters, and more
- **LoRAs** — all added LoRA models with their weights and enabled/disabled state

## How to Save a Project

You can save from two places:

1. **Toolbar** — Click the **Archive icon** in the canvas toolbar, then select **Save Canvas Project**
2. **Context menu** — Right-click the canvas, open the **Project** submenu, then select **Save Canvas Project**

A dialog will ask you to enter a **project name**. This name is used as the filename (e.g., entering "My Portrait" saves as `My Portrait.invk`) and is stored inside the project file.

## How to Load a Project

1. **Toolbar** — Click the **Archive icon**, then select **Load Canvas Project**
2. **Context menu** — Right-click the canvas, open the **Project** submenu, then select **Load Canvas Project**

A file dialog will open. Select your `.invk` file. You will see a confirmation dialog warning that loading will replace your current canvas. Click **Load** to proceed.

### What Happens on Load

- Your current canvas is **completely replaced** — all existing layers, masks, reference images, and parameters are overwritten
- Images that are already present on your InvokeAI server are reused automatically (no duplicate uploads)
- Images that were deleted from the server are re-uploaded from the project file
- If the saved model is not installed on your system, the model identifier is still restored — you will need to select an available model manually

## Good to Know

- **No undo** — Loading a project replaces your canvas entirely. There is no way to undo this action, so save your current project first if you want to keep it.
- **Image deduplication** — When loading, images already on your server are not re-uploaded. Only missing images are uploaded from the project file.
- **File size** — The `.invk` file size depends on the number and resolution of images in your canvas. A project with many high-resolution layers can be large.
- **Model availability** — The project saves which model was selected, but does not include the model itself. If the model is not installed when you load the project, you will need to select a different one.
