# Canvas Projects — Technical Documentation

## Overview

Canvas Projects provide a save/load mechanism for the entire canvas state. The feature serializes all canvas entities, generation parameters, reference images, and their associated image files into a ZIP-based `.invk` file. On load, it restores the full state, handling image deduplication and re-uploading as needed.

## File Format

The `.invk` file is a standard ZIP archive with the following structure:

```
project.invk
├── manifest.json
├── canvas_state.json
├── params.json
├── ref_images.json
├── loras.json
└── images/
    ├── {image_name_1}.png
    ├── {image_name_2}.png
    └── ...
```

### manifest.json

Schema version and metadata. Validated on load with Zod.

```json
{
  "version": 1,
  "appVersion": "5.12.0",
  "createdAt": "2026-02-26T12:00:00.000Z",
  "name": "My Canvas Project"
}
```

| Field | Type | Description |
|---|---|---|
| `version` | `number` | Schema version, currently `1`. Used for migration logic on load. |
| `appVersion` | `string` | InvokeAI version that created the file. Informational only. |
| `createdAt` | `string` | ISO 8601 timestamp. |
| `name` | `string` | User-provided project name. Also used as the download filename. |

### canvas_state.json

The serialized canvas entity tree. Type: `CanvasProjectState`.

```typescript
type CanvasProjectState = {
  rasterLayers: CanvasRasterLayerState[];
  controlLayers: CanvasControlLayerState[];
  inpaintMasks: CanvasInpaintMaskState[];
  regionalGuidance: CanvasRegionalGuidanceState[];
  bbox: CanvasState['bbox'];
  selectedEntityIdentifier: CanvasState['selectedEntityIdentifier'];
  bookmarkedEntityIdentifier: CanvasState['bookmarkedEntityIdentifier'];
};
```

Each entity contains its full state including all canvas objects (brush lines, eraser lines, rect shapes, images). Image objects reference files by `image_name` which correspond to files in the `images/` folder.

### params.json

The complete generation parameters state (`ParamsState`). Optional on load (older files may not have it). This includes all fields from the params Redux slice:

- Prompts (positive, negative, prompt history)
- Core generation settings (seed, steps, CFG scale, guidance, scheduler, iterations)
- Model selections (main model, VAE, FLUX VAE, T5 encoder, CLIP embed models, refiner, Z-Image models, Klein models)
- Dimensions (width, height, aspect ratio)
- Img2img strength
- Infill settings (method, tile size, patchmatch downscale, color)
- Canvas coherence settings (mode, edge size, min denoise)
- Refiner parameters (steps, CFG scale, scheduler, aesthetic scores, start)
- FLUX-specific settings (scheduler, DyPE preset/scale/exponent)
- Z-Image-specific settings (scheduler, seed variance)
- Upscale settings (scheduler, CFG scale)
- Seamless tiling, mask blur, CLIP skip, VAE precision, CPU noise, color compensation

### ref_images.json

Global reference image entities (`RefImageState[]`). These are IP-Adapter / FLUX Redux configs with `CroppableImageWithDims` containing both original and cropped image references. Optional on load.

### loras.json

Array of LoRA configurations (`LoRA[]`). Each entry contains:

```typescript
type LoRA = {
  id: string;
  isEnabled: boolean;
  model: ModelIdentifierField;
  weight: number;
};
```

Optional on load. Like models, LoRA identifiers are stored as-is — if a LoRA is not installed when loading, the entry is restored but may not be usable.

### images/

All image files referenced anywhere in the state. Keyed by their original `image_name`. On save, each image is fetched from the backend via `GET /api/v1/images/i/{name}/full` and stored as-is.

## Key Source Files

| File | Purpose |
|---|---|
| `features/controlLayers/util/canvasProjectFile.ts` | Types, constants, image name collection, remapping, existence checking |
| `features/controlLayers/hooks/useCanvasProjectSave.ts` | Save hook — collects Redux state, fetches images, builds ZIP |
| `features/controlLayers/hooks/useCanvasProjectLoad.ts` | Load hook — parses ZIP, deduplicates images, dispatches state |
| `features/controlLayers/components/SaveCanvasProjectDialog.tsx` | Save name dialog + `useSaveCanvasProjectWithDialog` hook |
| `features/controlLayers/components/LoadCanvasProjectConfirmationAlertDialog.tsx` | Load confirmation dialog + `useLoadCanvasProjectWithDialog` hook |
| `features/controlLayers/components/Toolbar/CanvasToolbarProjectMenuButton.tsx` | Toolbar dropdown UI |
| `features/controlLayers/store/canvasSlice.ts` | `canvasProjectRecalled` Redux action |

## Save Flow

1. User clicks "Save Canvas Project" → `SaveCanvasProjectDialog` opens asking for a project name
2. On confirm, `saveCanvasProject(name)` is called
3. Read Redux state via selectors: `selectCanvasSlice()`, `selectParamsSlice()`, `selectRefImagesSlice()`, `selectLoRAsSlice()`
4. Build `CanvasProjectState` from the canvas slice; use `paramsState` directly for params
5. Walk all entities to collect every `image_name` reference via `collectImageNames()`:
   - `CanvasImageState.image.image_name` in layer/mask objects
   - `CroppableImageWithDims.original.image.image_name` in global ref images
   - `CroppableImageWithDims.crop.image.image_name` in cropped ref images
   - `ImageWithDims.image_name` in regional guidance ref images
6. Fetch each image from the backend API
7. Build ZIP with JSZip: add `manifest.json` (including `name`), `canvas_state.json`, `params.json`, `ref_images.json`, and all images into `images/`
8. Sanitize the name for filesystem use and generate blob, trigger download as `{name}.invk`

## Load Flow

1. User selects `.invk` file → confirmation dialog opens
2. On confirm, parse ZIP with JSZip
3. Validate manifest version via Zod schema
4. Read `canvas_state.json`, `params.json` (optional), `ref_images.json` (optional)
5. Collect all `image_name` references from the loaded state
6. **Deduplicate images**: for each referenced image, check if it exists on the server via `getImageDTOSafe(image_name)`
   - Already exists → skip (no upload)
   - Missing → upload from ZIP via `uploadImage()`, record `oldName → newName` mapping
7. Remap all `image_name` values in the loaded state using the mapping (only for re-uploaded images whose names changed)
8. Dispatch Redux actions:
   - `canvasProjectRecalled()` — restores all canvas entities, bbox, selected/bookmarked entity
   - `refImagesRecalled()` — restores global reference images
   - `paramsRecalled()` — replaces the entire params state in one action
   - `loraAllDeleted()` + `loraRecalled()` — restores LoRAs
9. Show success/error toast

## Image Name Collection & Remapping

The `canvasProjectFile.ts` utility provides two parallel sets of functions:

**Collection** (`collectImageNames`): Walks the entire state tree and returns a `Set<string>` of all referenced `image_name` values. This is used by both save (to know which images to fetch) and load (to know which images to check/upload).

**Remapping** (`remapCanvasState`, `remapRefImages`): Deep-clones state objects and replaces `image_name` values using a `Map<string, string>` mapping. Only images that were re-uploaded with a different name are remapped. Images that already existed on the server are left unchanged.

Both walk the same paths through the state tree:
- Layer/mask objects → `CanvasImageState.image.image_name`
- Regional guidance ref images → `ImageWithDims.image_name`
- Global ref images → `CroppableImageWithDims.original.image.image_name` and `.crop.image.image_name`

## Extending the Format

### Adding new optional data (non-breaking)

Add a new JSON file to the ZIP. No version bump needed.

1. **Save**: Add `zip.file('new_data.json', JSON.stringify(data))` in `useCanvasProjectSave.ts`
2. **Load**: Read with `zip.file('new_data.json')` in `useCanvasProjectLoad.ts` — check for `null` so older project files without it still load
3. **Dispatch**: Add the appropriate Redux action to restore the data

### Adding new entity types with images

1. Extend `CanvasProjectState` type in `canvasProjectFile.ts`
2. Add collection logic in `collectImageNames()` to walk the new entity's objects
3. Add remapping logic in `remapCanvasState()` to update image names
4. Include the new entity array in both save and load hooks
5. Handle it in the `canvasProjectRecalled` reducer in `canvasSlice.ts`

### Breaking schema changes

1. Bump `CANVAS_PROJECT_VERSION` in `canvasProjectFile.ts`
2. Update the Zod manifest schema: `version: z.union([z.literal(1), z.literal(2)])`
3. Add migration logic in the load hook: check version, transform v1 → v2 before dispatching

## UI Architecture

### Save dialog

The save flow uses a **nanostore atom** (`$isOpen`) to control the `SaveCanvasProjectDialog`:

1. `useSaveCanvasProjectWithDialog()` — returns a callback that sets `$isOpen` to `true`
2. `SaveCanvasProjectDialog` (singleton in `GlobalModalIsolator`) — renders an `AlertDialog` with a name input
3. On save → calls `saveCanvasProject(name)` and closes the dialog
4. On cancel → closes the dialog

### Load dialog

The load flow uses a **nanostore atom** (`$pendingFile`) to decouple the file dialog from the confirmation dialog:

1. `useLoadCanvasProjectWithDialog()` — opens a programmatic file input (`document.createElement('input')`)
2. On file selection → sets `$pendingFile` atom
3. `LoadCanvasProjectConfirmationAlertDialog` (singleton in `GlobalModalIsolator`) — subscribes to `$pendingFile` via `useStore()`
4. On accept → calls `loadCanvasProject(file)` and clears the atom
5. On cancel → clears the atom

The programmatic file input approach was chosen because the context menu component uses `isLazy: true`, which unmounts the DOM tree when the menu closes — a hidden `<input>` element inside the menu would be destroyed before the file dialog returns.
