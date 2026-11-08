/**
 * Shows a loupe of the composited pixels under the pointer, previews samples while pressed and commits the last one
 * on release (or cancels it when the press ends any other way) through the active color target. Alt-hold restoration
 * returns to the prior tool; the picker neither dispatches nor edits pixels. State is per engine.
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

/** Samples the composite under the pointer as `#rrggbb`, or null off the document. */
const sampleAt = (ctx: ToolContext, sampler: ColorSampler, input: PointerInput): string | null => {
  const doc = ctx.getDocument();
  if (!doc) {
    return null;
  }
  const sample = sampler.sample(doc, ctx.layers, input.documentPoint, ctx.sampleProviders);
  return sample ? rgbaToHex(sample.r, sample.g, sample.b) : null;
};

/** Previews a sample through the one-shot claim then workbench router; standalone fallback updates brush color. */
const previewColor = (ctx: ToolContext, hex: string): void => {
  if (ctx.resolveColorSample?.(hex, 'preview')) {
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
  // The last color this press previewed; a pick commits or cancels only after previewing something.
  let pickedHex: string | null = null;
  const pick = (ctx: ToolContext, input: PointerInput): void => {
    sampler ??= createColorSampler(ctx.backend);
    const hex = sampleAt(ctx, sampler, input);
    if (hex !== null) {
      pickedHex = hex;
      previewColor(ctx, hex);
    }
  };
  /** Ends the pick once: release commits the last preview, anything else cancels it. */
  const settle = (ctx: ToolContext, phase: 'commit' | 'cancel'): void => {
    const hex = pickedHex;
    pickedHex = null;
    if (hex !== null) {
      ctx.resolveColorSample?.(hex, phase);
    }
  };
  return {
    // The loupe's boxed center pixel is the target; a system cursor would cover it.
    cursor: () => 'none',
    id: 'colorPicker',
    // A pointer already over the canvas (an Alt hold, a pick request) gets the loupe without moving.
    onActivate: (ctx) => updateLoupe(ctx),
    onDeactivate: (ctx) => {
      settle(ctx, 'cancel');
      ctx.showColorLoupe?.(false);
    },
    onPointerCancel: (ctx) => {
      settle(ctx, 'cancel');
    },
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      settle(ctx, 'cancel');
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
      settle(ctx, 'commit');
    },
  };
};
