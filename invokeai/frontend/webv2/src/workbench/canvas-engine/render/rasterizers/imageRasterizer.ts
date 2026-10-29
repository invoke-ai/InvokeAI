import type { CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { DecodedBitmapLease } from '@workbench/canvas-engine/render/decodedBitmapPool';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { createDecodedBitmapPool } from '@workbench/canvas-engine/render/decodedBitmapPool';

import type { RasterizeDeps, RasterizeResult } from './types';

/**
 * Thrown by an `ImageResolver` for an image that is missing (deleted, or never stored) or not readable by this user;
 * either lasts, so the layer reads as unreadable.
 */
export class CanvasImageMissingError extends Error {
  constructor(readonly imageName: string) {
    super(`Canvas image "${imageName}" is missing or not readable.`);
    this.name = 'CanvasImageMissingError';
  }
}

/**
 * The resolver could not fetch an image that may still be readable (network, server, an expired session): a
 * rasterization that fails with it can succeed later, unlike a missing image or one that does not decode.
 */
export class ImageUnavailableError extends Error {
  constructor(
    readonly imageName: string,
    options: { cause: unknown }
  ) {
    super(`Canvas image "${imageName}" could not be fetched.`, options);
    this.name = 'ImageUnavailableError';
  }
}

/**
 * Acquires a short-lived decoded bitmap lease, coalescing concurrent decodes by image name. A resolver failure other
 * than {@link CanvasImageMissingError} or cancellation rejects as {@link ImageUnavailableError}.
 */
export const resolveBitmap = async (image: CanvasImageRef, deps: RasterizeDeps): Promise<DecodedBitmapLease> => {
  deps.signal?.throwIfAborted();
  const pool = deps.bitmapPool ?? createDecodedBitmapPool();
  const lease = await pool.acquire(
    image.imageName,
    async (decodeSignal) => {
      let blob: Blob;
      try {
        blob = await deps.resolver(image.imageName, decodeSignal);
      } catch (error) {
        if (error instanceof CanvasImageMissingError || decodeSignal?.aborted) {
          throw error;
        }
        throw new ImageUnavailableError(image.imageName, { cause: error });
      }
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
