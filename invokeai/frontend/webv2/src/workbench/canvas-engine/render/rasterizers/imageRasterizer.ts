/**
 * Rasterizes an `image` layer source: decodes the referenced asset to an
 * `ImageBitmap` (cached per image name in the store) and blits it onto a
 * surface sized to the bitmap.
 *
 * Decoding is async and goes entirely through injected seams (the resolver
 * for bytes, the backend for `createImageBitmap`), so it runs in node tests
 * with fakes. Zero React, zero import-time side effects.
 */

import type { CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { DecodedBitmapLease } from '@workbench/canvas-engine/render/decodedBitmapPool';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { createDecodedBitmapPool } from '@workbench/canvas-engine/render/decodedBitmapPool';

import type { RasterizeDeps, RasterizeResult } from './types';

/** Acquires a short-lived decoded bitmap lease, coalescing concurrent decodes by image name. */
export const resolveBitmap = async (image: CanvasImageRef, deps: RasterizeDeps): Promise<DecodedBitmapLease> => {
  deps.signal?.throwIfAborted();
  const pool = deps.bitmapPool ?? createDecodedBitmapPool();
  const lease = await pool.acquire(
    image.imageName,
    async (decodeSignal) => {
      const blob = await deps.resolver(image.imageName, decodeSignal);
      decodeSignal?.throwIfAborted();
      const bitmap = await deps.backend.createImageBitmap(blob);
      if (decodeSignal?.aborted) {
        bitmap.close();
        decodeSignal.throwIfAborted();
      }
      return bitmap;
    },
    deps.signal
  );
  if (deps.signal?.aborted) {
    lease.release();
    deps.signal.throwIfAborted();
  }
  return lease;
};

/** Draws a bitmap onto a surface sized to `width`x`height`, reusing `target` if given. */
export const blitBitmap = (
  bitmap: ImageBitmap,
  width: number,
  height: number,
  deps: RasterizeDeps,
  target?: RasterSurface
): RasterSurface => {
  const surface = target ?? deps.backend.createSurface(width, height);
  if (surface.width !== width || surface.height !== height) {
    surface.resize(width, height);
  }
  surface.ctx.clearRect(0, 0, width, height);
  surface.ctx.drawImage(bitmap, 0, 0);
  return surface;
};

/** Rasterizes an `image` source into a surface sized to the image ref. */
export const rasterizeImageSource = async (
  source: { type: 'image'; image: CanvasImageRef },
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  const lease = await resolveBitmap(source.image, deps);
  try {
    const { height, width } = source.image;
    const surface = blitBitmap(lease.bitmap, width, height, deps, target);
    return { rect: { height, width, x: 0, y: 0 }, surface };
  } finally {
    lease.release();
  }
};
