/**
 * Node-safe RasterBackend records drawing calls, context properties and gradient stops without real pixels.
 * Synthetic getImageData buffers use {@link StubRasterBackendOptions.readbackAlpha}; browser tests verify actual
 * raster results.
 */

import type { RasterBackend, RasterSurface } from './raster';

/** A single recorded call made against a stub context. */
export interface RasterCallLogEntry {
  op: string;
  args: unknown[];
}

/** A `RasterSurface` created by the test stub backend, with its call log exposed. */
export interface StubRasterSurface extends RasterSurface {
  readonly callLog: RasterCallLogEntry[];
}

/** A `RasterBackend` whose `createSurface` returns the call-log-bearing `StubRasterSurface`. */
export interface StubRasterBackend extends RasterBackend {
  createSurface(width: number, height: number): StubRasterSurface;
}

/** Options for {@link createTestStubRasterBackend}. */
export interface StubRasterBackendOptions {
  /**
   * Synthetic readback alpha defaults to visible 255. Set zero explicitly for empty-cache tests because trimming
   * interprets transparent pixels as no content.
   */
  readbackAlpha?: number;
}

const createStubImageData = (width: number, height: number, alpha = 0): ImageData => {
  const data = new Uint8ClampedArray(Math.max(0, width) * Math.max(0, height) * 4);
  if (alpha !== 0) {
    for (let index = 3; index < data.length; index += 4) {
      data[index] = alpha;
    }
  }
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

/**
 * Stub text advance matches estimateTextExtent's 0.6*font-size factor, keeping node cache estimates and measured
 * extents consistent.
 */
const STUB_CHAR_WIDTH_FACTOR = 0.6;

/** Extracts the `px` size from a CSS `font` shorthand (e.g. `"700 48px Inter"` → 48), defaulting to 10. */
const fontSizeFromShorthand = (font: unknown): number => {
  const match = typeof font === 'string' ? /(\d+(?:\.\d+)?)px/.exec(font) : null;
  return match ? parseFloat(match[1] ?? '10') : 10;
};

const createStubCtx = (
  callLog: RasterCallLogEntry[],
  readbackAlpha: number
): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D => {
  const log = (op: string, args: unknown[]): void => {
    callLog.push({ args, op });
  };

  const props: Record<string, unknown> = {};

  const methods: Record<string, (...args: unknown[]) => unknown> = {
    arc: (...args: unknown[]) => log('arc', args),
    beginPath: (...args: unknown[]) => log('beginPath', args),
    clearRect: (...args: unknown[]) => log('clearRect', args),
    clip: (...args: unknown[]) => log('clip', args),
    closePath: (...args: unknown[]) => log('closePath', args),
    fillText: (...args: unknown[]) => log('fillText', args),
    measureText: (...args: unknown[]) => {
      log('measureText', args);
      const text = String(args[0] ?? '');
      const width = text.length * fontSizeFromShorthand(props.font) * STUB_CHAR_WIDTH_FACTOR;
      return { width } as unknown as TextMetrics;
    },
    createLinearGradient: (...args: unknown[]) => {
      log('createLinearGradient', args);
      return {
        addColorStop: (...stopArgs: unknown[]) => log('addColorStop', stopArgs),
      } as unknown as CanvasGradient;
    },
    createPattern: (...args: unknown[]) => {
      log('createPattern', args);
      // Non-null pattern marker lets guarded fill paths execute.
      return { __stubPattern: true } as unknown as CanvasPattern;
    },
    createRadialGradient: (...args: unknown[]) => {
      log('createRadialGradient', args);
      return {
        addColorStop: (...stopArgs: unknown[]) => log('addColorStop', stopArgs),
      } as unknown as CanvasGradient;
    },
    createImageData: (...args: unknown[]) => {
      const [width, height] = args as [number, number];
      log('createImageData', [width, height]);
      return createStubImageData(width, height, readbackAlpha);
    },
    drawImage: (...args: unknown[]) => log('drawImage', args),
    ellipse: (...args: unknown[]) => log('ellipse', args),
    fill: (...args: unknown[]) => log('fill', args),
    fillRect: (...args: unknown[]) => log('fillRect', args),
    getImageData: (...args: unknown[]) => {
      const [sx, sy, sw, sh] = args as [number, number, number, number];
      log('getImageData', [sx, sy, sw, sh]);
      return createStubImageData(sw, sh, readbackAlpha);
    },
    lineTo: (...args: unknown[]) => log('lineTo', args),
    moveTo: (...args: unknown[]) => log('moveTo', args),
    putImageData: (...args: unknown[]) => log('putImageData', args),
    rect: (...args: unknown[]) => log('rect', args),
    restore: (...args: unknown[]) => log('restore', args),
    save: (...args: unknown[]) => log('save', args),
    setLineDash: (...args: unknown[]) => log('setLineDash', args),
    setTransform: (...args: unknown[]) => log('setTransform', args),
    stroke: (...args: unknown[]) => log('stroke', args),
    strokeRect: (...args: unknown[]) => log('strokeRect', args),
  };

  const proxy = new Proxy(methods, {
    get(target, prop: string) {
      if (prop in target) {
        return target[prop];
      }
      return props[prop];
    },
    set(_target, prop: string, value: unknown) {
      props[prop] = value;
      log('set', [prop, value]);
      return true;
    },
  });

  return proxy as unknown as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
};

class StubRasterSurfaceImpl implements StubRasterSurface {
  readonly callLog: RasterCallLogEntry[] = [];
  readonly canvas: OffscreenCanvas | HTMLCanvasElement;
  readonly ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  width: number;
  height: number;

  constructor(width: number, height: number, readbackAlpha = 255) {
    this.width = width;
    this.height = height;
    this.canvas = { height, width } as unknown as OffscreenCanvas | HTMLCanvasElement;
    this.ctx = createStubCtx(this.callLog, readbackAlpha);
  }

  resize(w: number, h: number): void {
    this.callLog.push({ args: [w, h], op: 'resize' });
    this.width = w;
    this.height = h;
  }

  /**
   * Records resize/offset without real pixels. Preserve the context log across growth so tests can inspect the
   * surface's full history.
   */
  resizePreserving(w: number, h: number, dx: number, dy: number): void {
    this.callLog.push({ args: [w, h, dx, dy], op: 'resizePreserving' });
    this.width = w;
    this.height = h;
  }
}

export const createTestStubRasterBackend = (options: StubRasterBackendOptions = {}): StubRasterBackend => ({
  createImageBitmap: (source: ImageBitmapSource): Promise<ImageBitmap> => {
    void source;
    return Promise.resolve({ close: () => {}, height: 0, width: 0 } as unknown as ImageBitmap);
  },
  createSurface: (width: number, height: number): StubRasterSurface =>
    new StubRasterSurfaceImpl(width, height, options.readbackAlpha ?? 255),
  // Size-keyed fake blobs keep node encoding deterministic.
  encodeSurface: (surface: RasterSurface, type = 'image/png'): Promise<Blob> =>
    Promise.resolve(new Blob([`stub-surface-${surface.width}x${surface.height}`], { type })),
});
