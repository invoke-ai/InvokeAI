/**
 * Editable parametric text uses manual newlines, font-size line-height and per-line alignment within measured
 * bounds. Browser measureText/fillText supply metrics and shaping; there is no custom layout or automatic
 * wrapping. Empty text retains a minimal 1xlineHeight extent. The engine owns late-font rerasterization.
 */

import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

/** A `text` layer source. */
export type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;
type Ctx = RasterSurface['ctx'];

/**
 * Pure estimated character advance matches node stub metrics. Browser measureText determines final surfaces;
 * estimates only seed bounds/cache sizing.
 */
export const TEXT_CHAR_WIDTH_FACTOR = 0.6;

/** The CSS `font` shorthand for a text source (`"<style> <weight> <size>px <family>"`). */
export const textFontString = (source: TextSource, family = source.fontFamily): string => {
  const style = source.fontStyle && source.fontStyle !== 'normal' ? `${source.fontStyle} ` : '';
  return `${style}${source.fontWeight} ${source.fontSize}px ${family}`;
};

/** A stable CSS/OpenType representation of the exact coordinates carried by a text source. */
export const textFontVariationSettings = (source: TextSource): string =>
  Object.entries(source.fontVariations ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([tag, value]) => `"${tag}" ${value}`)
    .join(', ');

/** Splits a text source's content into lines on `\n`; always at least one (possibly empty) line. */
export const textLines = (content: string): string[] => content.split('\n');

/** The line-box height in document px for a source (`fontSize × lineHeight`). */
const lineHeightPx = (source: TextSource): number => source.fontSize * source.lineHeight;

/**
 * DOM-free text extent estimate for pre-measure geometry. Rasterization resizes to actual measured bounds; node
 * stubs use matching metrics.
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
 * Draws into measured text bounds, resizing an optional target. Returns a resolved promise to match source
 * dispatch.
 */
export const rasterizeTextSource = (
  source: TextSource,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  const lines = textLines(source.content);
  const font = textFontString(source, deps.resolveFontFamily?.(source));
  const context = (ctx: Ctx): Ctx & { fontVariationSettings?: string } =>
    ctx as Ctx & { fontVariationSettings?: string };
  const variationSettings = textFontVariationSettings(source);

  // Measure on the target's own ctx (or a fresh surface) with the font applied.
  const surface = target ?? deps.backend.createSurface(1, 1);
  surface.ctx.font = font;
  context(surface.ctx).fontVariationSettings = variationSettings || 'normal';
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
  context(ctx).fontVariationSettings = variationSettings || 'normal';
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
