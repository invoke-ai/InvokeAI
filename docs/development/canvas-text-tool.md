# Canvas Text Tool

## Overview

The canvas text workflow is split between a Konva module that owns tool state and a React overlay that handles text entry.

- `invokeai/frontend/web/src/features/controlLayers/konva/CanvasTool/CanvasTextToolModule.ts`
  - Owns the tool, cursor preview, and text session state.
  - Manages dynamic cursor contrast, starts sessions on pointer down, and commits sessions by rasterizing the active text block into a new raster layer.
- `invokeai/frontend/web/src/features/controlLayers/components/Text/CanvasTextOverlay.tsx`
  - Renders the on-canvas editor as a `contentEditable` overlay positioned in canvas space.
  - Syncs keyboard input, suppresses app hotkeys, and forwards commits/cancels to the Konva module.
- `invokeai/frontend/web/src/features/controlLayers/components/Text/TextToolOptions.tsx`
  - Provides the font dropdown, size slider/input, formatting toggles, and alignment buttons that appear when the Text tool is active.

## Rasterization pipeline

`renderTextToCanvas()` (`invokeai/frontend/web/src/features/controlLayers/text/textRenderer.ts`) converts the editor contents into a transparent canvas. The Text tool module configures the renderer with the active font stack, weight, styling flags, alignment, and the active canvas color. The resulting canvas is encoded to a PNG data URL and stored in a new raster layer (`image` object) with a transparent background.

Layer placement preserves the original click location:

- The session stores the anchor coordinate (where the user clicked) and current alignment.
- `calculateLayerPosition()` calculates the top-left position for the raster layer after applying the configured padding and alignment offsets.
- New layers are inserted directly above the currently-selected raster layer (when present) and selected automatically.

## Font stacks

Font definitions live in `invokeai/frontend/web/src/features/controlLayers/text/textConstants.ts` as five deterministic stacks covering sans, serif, mono, rounded, and script styles. Each stack lists system-safe fallbacks so the editor can choose the first available font per platform.

To add or adjust fonts:

1. Update `TEXT_FONT_STACKS` with the new `id`, `label`, and CSS `font-family` stack.
2. If you add a new stack, extend the `TEXT_FONT_IDS` tuple and update the `canvasTextSlice` schema default (`TEXT_DEFAULT_FONT_ID`).
3. Provide translation strings for any new labels in `public/locales/*`.
4. The editor and renderer will automatically pick up the new stack via `getFontStackById()`.
