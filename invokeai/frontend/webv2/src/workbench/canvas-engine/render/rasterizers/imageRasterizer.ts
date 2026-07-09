/**
 * Rasterizes an `image` layer source: decodes the referenced asset to an
 * `ImageBitmap` (cached per image name in the store) and blits it onto a
 * surface sized to the bitmap.
 *
 * Decoding is async and goes entirely through injected seams (the resolver
 * for bytes, the backend for `createImageBitmap`), so it runs in node tests
 * with fakes. Zero React, zero import-time side effects.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasImageRef } from '@workbench/types';

import type { RasterizeDeps, RasterizeResult } from './types';

/** Decodes and caches a bitmap for the given image reference, reusing the store cache. */
export const resolveBitmap = async (image: CanvasImageRef, deps: RasterizeDeps): Promise<ImageBitmap> => {
  const cached = deps.store.getBitmap(image.imageName);
  if (cached) {
    return cached;
  }
  const blob = await deps.resolver(image.imageName);
  const bitmap = await deps.backend.createImageBitmap(blob);
  // Another concurrent decode may have populated the cache while we awaited;
  // prefer the already-cached bitmap and close ours to avoid a leak.
  const raced = deps.store.getBitmap(image.imageName);
  if (raced) {
    bitmap.close();
    return raced;
  }
  deps.store.setBitmap(image.imageName, bitmap);
  return bitmap;
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
  const bitmap = await resolveBitmap(source.image, deps);
  const { height, width } = source.image;
  const surface = blitBitmap(bitmap, width, height, deps, target);
  return { rect: { height, width, x: 0, y: 0 }, surface };
};
