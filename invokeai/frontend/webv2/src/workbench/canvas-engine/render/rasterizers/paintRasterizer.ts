/**
 * Rasterizes a `paint` layer source. Paint layers are CONTENT-SIZED: the
 * persisted bitmap covers only the painted region, positioned in the layer's
 * local space by the source `offset` (default `{ x: 0, y: 0 }` for legacy
 * document-sized bitmaps). When the source has a bitmap it is decoded and
 * blitted onto a surface sized to the bitmap; when it is `null` the layer is
 * empty and the result is a zero-rect surface (a brand-new / cleared layer).
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

import { blitBitmap, resolveBitmap } from './imageRasterizer';

/** A `paint` source (its optional layer-local bitmap offset). */
type PaintSource = { type: 'paint'; bitmap: CanvasImageRef | null; offset?: { x: number; y: number } };

/** Rasterizes a `paint` source into a content-sized surface placed at its offset. */
export const rasterizePaintSource = async (
  source: PaintSource,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  if (source.bitmap) {
    const { height, width } = source.bitmap;
    const offset = source.offset ?? { x: 0, y: 0 };
    const lease = await resolveBitmap(source.bitmap, deps);
    try {
      const surface = blitBitmap(lease.bitmap, width, height, deps, target);
      return { rect: { height, width, x: offset.x, y: offset.y }, surface };
    } finally {
      lease.release();
    }
  }

  // No bitmap yet: an empty (zero-rect) layer. Collapse any reused target to 0×0
  // so the empty-layer invariants (skip in composite/hit-test/flush) hold.
  const surface = target ?? deps.backend.createSurface(0, 0);
  if (surface.width !== 0 || surface.height !== 0) {
    surface.resize(0, 0);
  }
  return { rect: { height: 0, width: 0, x: 0, y: 0 }, surface };
};
