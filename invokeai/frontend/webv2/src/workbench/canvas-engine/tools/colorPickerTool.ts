/**
 * The color-picker tool: while held down (usually via the alt-hold temp-tool
 * switch the pointer pipeline already drives — see `input/pointerPipeline.ts`),
 * it samples the composited document color under the cursor and writes it into
 * the brush color option, so releasing alt drops the user right back into
 * painting with the picked color. Sampling reads the layer cache directly
 * through {@link sampleDocumentColor} — this tool never dispatches and never
 * touches pixels.
 *
 * Each engine builds its own instance so cursor-ring state is per-engine.
 * Zero React, zero import-time side effects.
 */

import type { PointerInput } from '@workbench/canvas-engine/types';

import { rgbaToHex } from '@workbench/canvas-engine/color';
import { sampleDocumentColor } from '@workbench/canvas-engine/render/colorSample';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Fixed on-screen radius (px) for the picker's ring cursor, independent of zoom. */
const CURSOR_SCREEN_RADIUS_PX = 8;

const updateCursorRing = (ctx: ToolContext, input: PointerInput): void => {
  const zoom = ctx.viewport.getZoom();
  ctx.setOverlayCursor({ point: input.documentPoint, radiusDoc: CURSOR_SCREEN_RADIUS_PX / Math.max(zoom, 1e-6) });
  ctx.invalidate({ overlay: true });
};

/** Samples the composited color under `input` and writes it into the brush color option, if pickable. */
const pickColorAt = (ctx: ToolContext, input: PointerInput): void => {
  const doc = ctx.getDocument();
  if (!doc) {
    return;
  }
  const sample = sampleDocumentColor(doc, ctx.layers, ctx.backend, input.documentPoint);
  if (!sample) {
    return;
  }
  const hex = rgbaToHex(sample.r, sample.g, sample.b);
  const opts = ctx.stores.brushOptions.get();
  if (hex !== opts.color) {
    ctx.stores.brushOptions.set({ ...opts, color: hex });
  }
};

/** Creates a fresh color-picker tool. */
export const createColorPickerTool = (): Tool => ({
  cursor: () => 'crosshair',
  id: 'colorPicker',
  onDeactivate: (ctx) => {
    ctx.setOverlayCursor(null);
    ctx.invalidate({ overlay: true });
  },
  onPointerDown: (ctx, input) => {
    if ((input.buttons & PRIMARY_BUTTON) === 0) {
      return;
    }
    updateCursorRing(ctx, input);
    pickColorAt(ctx, input);
  },
  onPointerMove: (ctx, input) => {
    updateCursorRing(ctx, input);
    if (input.buttons & PRIMARY_BUTTON) {
      pickColorAt(ctx, input);
    }
  },
  onPointerUp: (ctx, input) => {
    updateCursorRing(ctx, input);
  },
});
