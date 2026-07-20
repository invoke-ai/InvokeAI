/**
 * The RasterBackend seam: every place the engine needs a 2D canvas surface
 * or an ImageBitmap, it goes through an injected `RasterBackend` instead of
 * calling `document.createElement('canvas')` / `new OffscreenCanvas(...)`
 * directly. This keeps the engine testable in node — tests inject
 * `render/raster.testStub.ts`'s stub backend instead of `createDomRasterBackend()`.
 */

/** A single drawable/resizable 2D surface, backed by either an OffscreenCanvas or an HTMLCanvasElement. */
export interface RasterSurface {
  readonly canvas: OffscreenCanvas | HTMLCanvasElement;
  readonly ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  readonly width: number;
  readonly height: number;
  resize(w: number, h: number): void;
}

export interface RasterSurfaceOptions {
  /** Optimize the backing context for repeated pixel readback (for example brush-history caches). */
  willReadFrequently?: boolean;
}

/** Injectable factory for raster surfaces and image bitmaps. */
export interface RasterBackend {
  createSurface(width: number, height: number, options?: RasterSurfaceOptions): RasterSurface;
  createImageBitmap(source: ImageBitmapSource): Promise<ImageBitmap>;
  /**
   * Encodes a surface's pixels to an image `Blob` (PNG by default). Used by the
   * bitmap store to persist painted layers as content-hashed server images.
   */
  encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
}

const isOffscreenCanvasSupported = (): boolean => typeof OffscreenCanvas !== 'undefined';

/** Encodes a DOM/Offscreen surface's canvas to an image `Blob`. */
const encodeDomSurface = (surface: RasterSurface, type: string): Promise<Blob> => {
  const { canvas } = surface;
  if (typeof (canvas as OffscreenCanvas).convertToBlob === 'function') {
    return (canvas as OffscreenCanvas).convertToBlob({ type });
  }
  const htmlCanvas = canvas as HTMLCanvasElement;
  return new Promise((resolve, reject) => {
    htmlCanvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('Failed to encode canvas surface to a Blob'));
      }
    }, type);
  });
};

class OffscreenRasterSurface implements RasterSurface {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
  width: number;
  height: number;

  constructor(width: number, height: number, options?: RasterSurfaceOptions) {
    this.canvas = new OffscreenCanvas(width, height);
    const ctx = this.canvas.getContext('2d', { willReadFrequently: options?.willReadFrequently });
    if (!ctx) {
      throw new Error('Failed to acquire a 2D context from OffscreenCanvas');
    }
    this.ctx = ctx;
    this.width = width;
    this.height = height;
  }

  resize(w: number, h: number): void {
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
  }
}

class DomCanvasRasterSurface implements RasterSurface {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;

  constructor(width: number, height: number, options?: RasterSurfaceOptions) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    const ctx = this.canvas.getContext('2d', { willReadFrequently: options?.willReadFrequently });
    if (!ctx) {
      throw new Error('Failed to acquire a 2D context from HTMLCanvasElement');
    }
    this.ctx = ctx;
    this.width = width;
    this.height = height;
  }

  resize(w: number, h: number): void {
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
  }
}

/**
 * Creates a `RasterBackend` backed by the DOM/browser: `OffscreenCanvas`
 * when available, falling back to `HTMLCanvasElement` otherwise (notably
 * Safari < 16.4, which lacks `OffscreenCanvas` support).
 */
export const createDomRasterBackend = (): RasterBackend => {
  const useOffscreen = isOffscreenCanvasSupported();
  return {
    createSurface(width: number, height: number, options?: RasterSurfaceOptions): RasterSurface {
      return useOffscreen
        ? new OffscreenRasterSurface(width, height, options)
        : new DomCanvasRasterSurface(width, height, options);
    },
    createImageBitmap(source: ImageBitmapSource): Promise<ImageBitmap> {
      return createImageBitmap(source);
    },
    encodeSurface(surface: RasterSurface, type = 'image/png'): Promise<Blob> {
      return encodeDomSurface(surface, type);
    },
  };
};
