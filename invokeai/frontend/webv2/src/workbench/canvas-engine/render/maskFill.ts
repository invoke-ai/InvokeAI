import type { CanvasMaskFillContract } from '@workbench/canvas-engine/contracts';

import type { RasterBackend, RasterSurface } from './raster';

/** Line specs (tile size, line spacing, stroke width) for each non-solid fill style. */
const PATTERN_SPECS: Record<
  Exclude<CanvasMaskFillContract['style'], 'solid'>,
  { size: number; spacing: number; width: number }
> = {
  // 6px tile, two parallel 45° lines 3px apart (legacy `pattern-diagonal.svg`).
  diagonal: { size: 6, spacing: 6, width: 1.3 },
  diagonal_left: { size: 6, spacing: 6, width: 1.3 },
  // 6px tile, 45° lines in BOTH directions (legacy `pattern-crosshatch.svg`).
  crosshatch: { size: 6, spacing: 6, width: 1.2 },
  // 12px tile, 3 vertical + 3 horizontal lines 4px apart (legacy `pattern-grid.svg`).
  grid: { size: 12, spacing: 4, width: 1 },
  horizontal: { size: 4, spacing: 4, width: 1 },
  vertical: { size: 4, spacing: 4, width: 1 },
};

/** Draws the anti-diagonal (`x + y = c`) or main-diagonal (`x - y = c`) line family across the tile. */
const drawDiagonalLines = (
  ctx: RasterSurface['ctx'],
  size: number,
  spacing: number,
  direction: 'anti' | 'main',
  phase: number
): void => {
  // Sweep `c` well beyond the tile so every crossing line is drawn; the surface clips to its bounds. `spacing` divides
  // `size`, so the family tiles seamlessly.
  for (let c = phase - size; c <= 2 * size; c += spacing) {
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

/** Builds a color-specific repeating tile; solid fills return null for direct color drawing. */
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
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'anti', 4);
      break;
    case 'diagonal_left':
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'main', 2);
      break;
    case 'crosshatch':
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'anti', 4);
      drawDiagonalLines(ctx, spec.size, spec.spacing, 'main', 3);
      break;
    case 'grid':
    case 'horizontal':
    case 'vertical': {
      // Half-pixel alignment keeps 1px strokes crisp, matching legacy SVG tiles.
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

/** Colorizes cache alpha with solid fill or repeat tile via source-in; the caller applies the layer transform. */
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
