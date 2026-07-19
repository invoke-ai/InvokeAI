/**
 * Rasterizes a `text` layer source. Text layers are PARAMETRIC and
 * EDITABLE-FOREVER: their pixels are derived from the source params (`content`,
 * `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `align`, `color`) rather
 * than a persisted bitmap, so a style/content edit re-renders for free — the
 * headline upgrade over legacy's rasterized (frozen) text.
 *
 * Layout is deliberately SIMPLE (Risk 8): manual line breaks on `\n`, a
 * `lineHeight` multiplier over `fontSize`, and per-line horizontal alignment
 * within the measured block width. There is NO shaping engine — no BiDi, no
 * complex-script clustering, no automatic word wrap, no kerning beyond what the
 * platform `fillText` applies. Metrics come through the {@link RasterSurface}
 * `ctx` seam (`measureText`), so node tests run on the stub's deterministic
 * `width = chars × fontSizePx × 0.6` (see `raster.testStub.ts`).
 *
 * Extent semantics: like a shape, the surface is sized to the text BLOCK's own
 * measured `width`×`height` (its layer-local extent, top-left origin); the
 * compositor applies the layer transform (position/scale) when drawing. A layer
 * with an empty `content` still produces a minimal 1×lineHeight surface so its
 * transform/anchor stays meaningful.
 *
 * Font loading (late web-font availability) is handled by the ENGINE, not here:
 * this rasterizer draws with whatever metrics `ctx` currently reports, and the
 * engine re-rasterizes when a pending font resolves (see `render/fontLoader.ts`).
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

/** A `text` layer source. */
export type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;
type Ctx = RasterSurface['ctx'];

/**
 * Per-character horizontal advance as a fraction of the font's pixel size, used
 * by the pure {@link estimateTextExtent}. It matches the test stub's
 * `measureText` factor so a text layer's estimated extent and its
 * stub-measured surface size agree exactly in node tests. In the browser the
 * true `measureText` governs the rendered surface; the estimate is only a
 * pre-measure cache-size / bounds approximation.
 */
export const TEXT_CHAR_WIDTH_FACTOR = 0.6;

/** The CSS `font` shorthand for a text source (`"<weight> <size>px <family>"`). */
export const textFontString = (source: TextSource): string =>
  `${source.fontWeight} ${source.fontSize}px ${source.fontFamily}`;

/** Splits a text source's content into lines on `\n`; always at least one (possibly empty) line. */
export const textLines = (content: string): string[] => content.split('\n');

/** The line-box height in document px for a source (`fontSize × lineHeight`). */
const lineHeightPx = (source: TextSource): number => source.fontSize * source.lineHeight;

/**
 * A pure, DOM-free estimate of a text block's unscaled pixel extent, matching
 * the stub's deterministic metrics (`TEXT_CHAR_WIDTH_FACTOR`). Used by the pure
 * geometry helpers (`sources.ts`) to size caches / bounds before a real
 * `measureText` runs; the rasterizer resizes the surface to the precise
 * measured size, so a browser mismatch only affects the initial cache size.
 */
export const estimateTextExtent = (source: TextSource): { width: number; height: number } => {
  const lines = textLines(source.content);
  const widest = lines.reduce((max, line) => Math.max(max, line.length), 0);
  return {
    height: Math.max(1, Math.ceil(lines.length * lineHeightPx(source))),
    width: Math.max(1, Math.ceil(widest * source.fontSize * TEXT_CHAR_WIDTH_FACTOR)),
  };
};

/** Measures the block extent through the seam's `ctx` (font must be set first). */
const measureBlock = (ctx: Ctx, source: TextSource, lines: string[]): { width: number; height: number } => {
  let widest = 0;
  for (const line of lines) {
    widest = Math.max(widest, ctx.measureText(line).width);
  }
  return {
    height: Math.max(1, Math.ceil(lines.length * lineHeightPx(source))),
    width: Math.max(1, Math.ceil(widest)),
  };
};

/**
 * Draws a text source into a surface sized to the measured text block, reusing
 * `target` if provided (resizing it to the measured extent), matching the
 * paint/image/shape rasterizer contract. Synchronous work wrapped in a resolved
 * promise so it shares the `rasterizeSource` dispatch signature.
 */
export const rasterizeTextSource = (
  source: TextSource,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  const lines = textLines(source.content);
  const font = textFontString(source);

  // Measure on the target's own ctx (or a fresh surface) with the font applied.
  const surface = target ?? deps.backend.createSurface(1, 1);
  surface.ctx.font = font;
  const { height, width } = measureBlock(surface.ctx, source, lines);

  if (surface.width !== width || surface.height !== height) {
    surface.resize(width, height);
  }
  const { ctx } = surface;
  // A resize resets the browser context state, so (re)apply the transform, font,
  // and paint state AFTER resizing.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = font;
  ctx.textBaseline = 'top';
  ctx.fillStyle = source.color;

  // Horizontal alignment within the block width, via the canvas text anchor.
  let anchorX = 0;
  if (source.align === 'center') {
    ctx.textAlign = 'center';
    anchorX = width / 2;
  } else if (source.align === 'right') {
    ctx.textAlign = 'right';
    anchorX = width;
  } else {
    ctx.textAlign = 'left';
    anchorX = 0;
  }

  const step = lineHeightPx(source);
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i] ?? '', anchorX, i * step);
  }

  return Promise.resolve({ rect: { height, width, x: 0, y: 0 }, surface });
};
