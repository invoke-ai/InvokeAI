/**
 * Injected RasterBackend owns all canvas/ImageBitmap creation, allowing node tests to use recording surfaces
 * instead of browser globals.
 */

/** A single drawable/resizable 2D surface, backed by either an OffscreenCanvas or an HTMLCanvasElement. */
export interface RasterSurface {
  readonly canvas: OffscreenCanvas | HTMLCanvasElement;
  readonly ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  readonly width: number;
  readonly height: number;
  resize(w: number, h: number): void;
  /**
   * Resize preserves pixels at (dx,dy) by adopting a fresh canvas and blitting the old backing store, avoiding CPU
   * readback and a second temporary allocation. Surface identity remains stable, but canvas/context references are
   * replaced and all context state resets.
   */
  resizePreserving(w: number, h: number, dx: number, dy: number): void;
}

export interface RasterSurfaceOptions {
  /** Optimize the backing context for repeated pixel readback (for example brush-history caches). */
  willReadFrequently?: boolean;
}

/** Injectable factory for raster surfaces and image bitmaps. */
export interface RasterBackend {
  createSurface(width: number, height: number, options?: RasterSurfaceOptions): RasterSurface;
  createImageBitmap(source: ImageBitmapSource): Promise<ImageBitmap>;
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
  private readonly options: RasterSurfaceOptions | undefined;

  constructor(width: number, height: number, options?: RasterSurfaceOptions) {
    this.options = options;
    this.canvas = new OffscreenCanvas(width, height);
    this.ctx = this.acquire();
    this.width = width;
    this.height = height;
  }

  private acquire(): OffscreenCanvasRenderingContext2D {
    const ctx = this.canvas.getContext('2d', { willReadFrequently: this.options?.willReadFrequently });
    if (!ctx) {
      throw new Error('Failed to acquire a 2D context from OffscreenCanvas');
    }
    return ctx;
  }

  resize(w: number, h: number): void {
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
  }

  resizePreserving(w: number, h: number, dx: number, dy: number): void {
    const previous = this.canvas;
    this.canvas = new OffscreenCanvas(w, h);
    this.ctx = this.acquire();
    this.ctx.drawImage(previous, dx, dy);
    this.width = w;
    this.height = h;
  }
}

class DomCanvasRasterSurface implements RasterSurface {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  private readonly options: RasterSurfaceOptions | undefined;

  constructor(width: number, height: number, options?: RasterSurfaceOptions) {
    this.options = options;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.acquire();
    this.width = width;
    this.height = height;
  }

  private acquire(): CanvasRenderingContext2D {
    const ctx = this.canvas.getContext('2d', { willReadFrequently: this.options?.willReadFrequently });
    if (!ctx) {
      throw new Error('Failed to acquire a 2D context from HTMLCanvasElement');
    }
    return ctx;
  }

  resize(w: number, h: number): void {
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
  }

  resizePreserving(w: number, h: number, dx: number, dy: number): void {
    const previous = this.canvas;
    this.canvas = document.createElement('canvas');
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx = this.acquire();
    this.ctx.drawImage(previous, dx, dy);
    this.width = w;
    this.height = h;
  }
}

/** Uses OffscreenCanvas when available, otherwise HTMLCanvasElement. */
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
