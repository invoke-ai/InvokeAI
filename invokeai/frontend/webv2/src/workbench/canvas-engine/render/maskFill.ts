/**
 * Mask fill rendering: turns a mask layer's alpha stencil into a tinted,
 * translucent overlay coloured by its {@link CanvasMaskFillContract}.
 *
 * Mirrors legacy (`features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer.ts`):
 * the mask objects are drawn as an opaque alpha stencil, then a fill (a solid
 * colour, or one of five line PATTERNS) is composited over them with
 * `globalCompositeOperation: 'source-in'` so the colour/pattern shows ONLY where
 * the mask is painted. The whole overlay is drawn above all raster/control
 * content at the layer's opacity (legacy default `1`).
 *
 * Pattern tiles are small, colour-specific, cached surfaces built through the
 * {@link RasterBackend} seam (like the transparency checkerboard). Legacy anchors
 * its pattern to the screen; here the colorized surface is drawn through the
 * layer transform, so the pattern is OBJECT-anchored (it scales/pans with the
 * mask content) — a deliberate, documented simplification of the screen-anchored
 * legacy behaviour that keeps the whole pipeline node-testable through the seam.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasMaskFillContract } from '@workbench/types';

import type { RasterBackend, RasterSurface } from './raster';

/** Line specs (tile size, line spacing, stroke width) for each non-solid fill style. */
const PATTERN_SPECS: Record<
  Exclude<CanvasMaskFillContract['style'], 'solid'>,
  { size: number; spacing: number; width: number }
> = {
  // 6px tile, two parallel 45° lines 3px apart (legacy `pattern-diagonal.svg`).
  diagonal: { size: 6, spacing: 3, width: 1.3 },
  // 6px tile, 45° lines in BOTH directions (legacy `pattern-crosshatch.svg`).
  crosshatch: { size: 6, spacing: 3, width: 1.2 },
  // 12px tile, 3 vertical + 3 horizontal lines 4px apart (legacy `pattern-grid.svg`).
  grid: { size: 12, spacing: 4, width: 1 },
  // 9px tile, 3 horizontal lines 3px apart (legacy `pattern-horizontal.svg`).
  horizontal: { size: 9, spacing: 3, width: 1 },
  // 9px tile, 3 vertical lines 3px apart (legacy `pattern-vertical.svg`).
  vertical: { size: 9, spacing: 3, width: 1 },
};

/** Draws the anti-diagonal (`x + y = c`) or main-diagonal (`x - y = c`) line family across the tile. */
const drawDiagonalLines = (
  ctx: RasterSurface['ctx'],
  size: number,
  spacing: number,
  direction: 'anti' | 'main'
): void => {
  // Sweep `c` well beyond the tile so every crossing line is drawn; the surface
  // clips to its bounds. `spacing` divides `size`, so the family tiles seamlessly.
  for (let c = -size; c <= 2 * size; c += spacing) {
    ctx.beginPath();
    if (direction === 'anti') {
      // x + y = c: a long segment crossing the tile.
      ctx.moveTo(c + size, -size);
      ctx.lineTo(c - 2 * size, 2 * size);
    } else {
      // x - y = c.
      ctx.moveTo(c - size, -size);
      ctx.lineTo(c + 2 * size, 2 * size);
    }
    ctx.stroke();
  }
};

/**
 * Builds a colour-specific repeat tile for a non-solid fill `style`, or returns
 * `null` for `'solid'` (no pattern — the colour is filled directly). The tile is
 * drawn in `color` through the backend seam.
 */
export const createMaskPatternTile = (
  backend: RasterBackend,
  style: CanvasMaskFillContract['style'],
  color: string
): RasterSurface | null => {
  if (style === 'solid') {
    return null;
  }
  const spec = PATTERN_SPECS[style];
  const tile = backend.createSurface(spec.size, spec.size);
  const ctx = tile.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, spec.size, spec.size);
  ctx.strokeStyle = color;
  ctx.lineWidth = spec.width;

  switch (style) {
    case 'diagonal':
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'anti');
      break;
    case 'crosshatch':
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'anti');
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'main');
      break;
    case 'grid':
    case 'horizontal':
    case 'vertical': {
      // Pixel-centred lines (0.5 offset) so 1px strokes stay crisp, matching
      // the legacy SVG tiles (lines at 0.5 / 3.5 / 6.5, etc.).
      for (let p = 0.5; p < spec.size; p += spec.spacing) {
        if (style !== 'horizontal') {
          ctx.beginPath();
          ctx.moveTo(p, 0);
          ctx.lineTo(p, spec.size);
          ctx.stroke();
        }
        if (style !== 'vertical') {
          ctx.beginPath();
          ctx.moveTo(0, p);
          ctx.lineTo(spec.size, p);
          ctx.stroke();
        }
      }
      break;
    }
  }
  return tile;
};

/**
 * Produces a colorized RGBA surface the size of the mask cache: the mask's alpha
 * is the stencil, the `fill` supplies the colour (solid) or a repeat `tile`
 * (pattern), composited `source-in`. The caller blits the result through the
 * layer transform.
 */
export const colorizeMask = (
  backend: RasterBackend,
  mask: RasterSurface,
  width: number,
  height: number,
  fill: CanvasMaskFillContract,
  tile: RasterSurface | null,
  target: RasterSurface | null = null
): RasterSurface => {
  const out = target ?? backend.createSurface(width, height);
  if (out.width !== width || out.height !== height) {
    out.resize(width, height);
  }
  const ctx = out.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  // 1. The mask alpha stencil.
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(mask.canvas, 0, 0);
  // 2. Colour/pattern kept only where the stencil is opaque (`source-in`).
  ctx.globalCompositeOperation = 'source-in';
  if (tile) {
    const pattern = ctx.createPattern(tile.canvas, 'repeat');
    ctx.fillStyle = pattern ?? fill.color;
  } else {
    ctx.fillStyle = fill.color;
  }
  ctx.fillRect(0, 0, width, height);
  return out;
};
