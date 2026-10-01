/**
 * Samples composited cached color while pressed and routes it to the active color target. Alt-hold restoration
 * returns to the prior tool; the picker neither dispatches nor edits pixels. State is per engine.
 */

import type { PointerInput } from '@workbench/canvas-engine/types';

import { rgbaToHex } from '@workbench/canvas-engine/color';
import { createColorSampler, type ColorSampler } from '@workbench/canvas-engine/render/colorSample';

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

/** Offers sampled color to the one-shot claim then workbench router; standalone fallback updates brush color. */
const pickColorAt = (ctx: ToolContext, sampler: ColorSampler, input: PointerInput): void => {
  const doc = ctx.getDocument();
  if (!doc) {
    return;
  }
  const sample = sampler.sample(doc, ctx.layers, input.documentPoint, ctx.sampleProviders);
  if (!sample) {
    return;
  }
  const hex = rgbaToHex(sample.r, sample.g, sample.b);
  if (ctx.resolveColorSample?.(hex)) {
    return;
  }
  const opts = ctx.stores.brushOptions.get();
  if (hex !== opts.color) {
    ctx.stores.brushOptions.set({ ...opts, color: hex });
  }
};

/** Creates a fresh color-picker tool. */
export const createColorPickerTool = (): Tool => {
  // One engine owns the tool and its backend, so the sampler can outlive a gesture.
  let sampler: ColorSampler | null = null;
  const pick = (ctx: ToolContext, input: PointerInput): void => {
    sampler ??= createColorSampler(ctx.backend);
    pickColorAt(ctx, sampler, input);
  };
  return {
    cursor: () => 'crosshair',
    id: 'colorPicker',
    onDeactivate: (ctx) => {
      ctx.setOverlayCursor(null);
      ctx.invalidate({ overlay: true });
    },
    onPointerCancel: (ctx) => {
      ctx.discardColorSample?.();
    },
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      ctx.discardColorSample?.();
      updateCursorRing(ctx, input);
      pick(ctx, input);
    },
    onPointerMove: (ctx, input) => {
      updateCursorRing(ctx, input);
      if (input.buttons & PRIMARY_BUTTON) {
        pick(ctx, input);
      }
    },
    onPointerUp: (ctx, input) => {
      updateCursorRing(ctx, input);
      ctx.commitColorSample?.();
    },
  };
};
