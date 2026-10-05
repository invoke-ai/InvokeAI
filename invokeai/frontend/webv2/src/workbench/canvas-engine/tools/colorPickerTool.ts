/**
 * Shows a loupe of the composited pixels under the pointer, samples while pressed and routes the color to the active
 * color target. Alt-hold restoration returns to the prior tool; the picker neither dispatches nor edits pixels.
 * State is per engine.
 */

import type { PointerInput } from '@workbench/canvas-engine/types';

import { rgbaToHex } from '@workbench/canvas-engine/color';
import { createColorSampler, type ColorSampler } from '@workbench/canvas-engine/render/colorSample';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** The loupe follows the pointer, hovering or pressed; the engine samples it once per overlay frame. */
const updateLoupe = (ctx: ToolContext): void => {
  ctx.showColorLoupe?.(true);
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
    // The loupe's boxed center pixel is the target; a system cursor would cover it.
    cursor: () => 'none',
    id: 'colorPicker',
    // A pointer already over the canvas (an Alt hold, a pick request) gets the loupe without moving.
    onActivate: (ctx) => updateLoupe(ctx),
    onDeactivate: (ctx) => {
      ctx.showColorLoupe?.(false);
    },
    onPointerCancel: (ctx) => {
      ctx.discardColorSample?.();
    },
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      ctx.discardColorSample?.();
      updateLoupe(ctx);
      pick(ctx, input);
    },
    onPointerMove: (ctx, input) => {
      updateLoupe(ctx);
      if (input.buttons & PRIMARY_BUTTON) {
        pick(ctx, input);
      }
    },
    onPointerUp: (ctx) => {
      updateLoupe(ctx);
      ctx.commitColorSample?.();
    },
  };
};
