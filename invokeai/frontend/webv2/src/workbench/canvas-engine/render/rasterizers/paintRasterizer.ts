/**
 * Paint surfaces match persisted bitmap dimensions at the source's local offset (legacy zero). Null bitmaps yield
 * empty zero-rect surfaces.
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
