/**
 * A node-safe `RasterBackend` stub for vitest. Surfaces are backed by a
 * fake 2D context that records every call it receives instead of touching
 * a real canvas, so engine tests can assert on what was drawn without a
 * DOM or `OffscreenCanvas`.
 *
 * The subset of `CanvasRenderingContext2D` the engine currently uses is
 * implemented as recorded methods (save/restore/clearRect/fillRect/
 * strokeRect/drawImage/path building/fill/stroke/clip/createPattern/
 * createLinearGradient/createRadialGradient/setTransform/
 * getImageData/putImageData/setLineDash). Gradients returned by the
 * `create*Gradient` factories record their `addColorStop` calls on the same
 * surface call log. Property assignments (fillStyle,
 * globalAlpha, globalCompositeOperation, ...) are recorded too, via a
 * `{ op: 'set', args: [prop, value] }` entry, so tests can assert that
 * opacity/blend/style state was applied in order. Extend the method table in
 * `createStubCtx` below as later tasks need more of the API surface.
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

const createStubImageData = (width: number, height: number): ImageData => {
  const data = new Uint8ClampedArray(Math.max(0, width) * Math.max(0, height) * 4);
  return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
};

/**
 * Deterministic per-character advance the stub's {@link measureText} multiplies
 * by the current font's pixel size — the same `0.6` factor the text rasterizer's
 * pure `estimateTextExtent` uses (`TEXT_CHAR_WIDTH_FACTOR`), so a text layer's
 * measured surface size in node tests exactly matches its estimated extent.
 */
const STUB_CHAR_WIDTH_FACTOR = 0.6;

/** Extracts the `px` size from a CSS `font` shorthand (e.g. `"700 48px Inter"` → 48), defaulting to 10. */
const fontSizeFromShorthand = (font: unknown): number => {
  const match = typeof font === 'string' ? /(\d+(?:\.\d+)?)px/.exec(font) : null;
  return match ? parseFloat(match[1] ?? '10') : 10;
};

const createStubCtx = (callLog: RasterCallLogEntry[]): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D => {
  const log = (op: string, args: unknown[]): void => {
    callLog.push({ args, op });
  };

  // Stored non-method property values (fillStyle, globalAlpha, font, etc.).
  // Declared before the method table so `measureText` can read the current
  // `font` to produce font-size-dependent metrics.
  const props: Record<string, unknown> = {};

  const methods: Record<string, (...args: unknown[]) => unknown> = {
    arc: (...args: unknown[]) => log('arc', args),
    beginPath: (...args: unknown[]) => log('beginPath', args),
    clearRect: (...args: unknown[]) => log('clearRect', args),
    clip: (...args: unknown[]) => log('clip', args),
    closePath: (...args: unknown[]) => log('closePath', args),
    fillText: (...args: unknown[]) => log('fillText', args),
    // Deterministic, font-size-aware metrics: width = chars × fontSizePx × 0.6.
    // No DOM/real canvas needed, so text measurement is reproducible in node.
    measureText: (...args: unknown[]) => {
      log('measureText', args);
      const text = String(args[0] ?? '');
      const width = text.length * fontSizeFromShorthand(props.font) * STUB_CHAR_WIDTH_FACTOR;
      return { width } as unknown as TextMetrics;
    },
    createLinearGradient: (...args: unknown[]) => {
      log('createLinearGradient', args);
      // A recording CanvasGradient stand-in: every addColorStop is logged on
      // the surface's own call log (as `addColorStop`), so tests can assert the
      // stop offsets/colors that were applied to the gradient.
      return {
        addColorStop: (...stopArgs: unknown[]) => log('addColorStop', stopArgs),
      } as unknown as CanvasGradient;
    },
    createPattern: (...args: unknown[]) => {
      log('createPattern', args);
      // A non-null marker standing in for a CanvasPattern, so callers that
      // guard on a null return (e.g. the checkerboard fill) proceed.
      return { __stubPattern: true } as unknown as CanvasPattern;
    },
    createRadialGradient: (...args: unknown[]) => {
      log('createRadialGradient', args);
      return {
        addColorStop: (...stopArgs: unknown[]) => log('addColorStop', stopArgs),
      } as unknown as CanvasGradient;
    },
    drawImage: (...args: unknown[]) => log('drawImage', args),
    ellipse: (...args: unknown[]) => log('ellipse', args),
    fill: (...args: unknown[]) => log('fill', args),
    fillRect: (...args: unknown[]) => log('fillRect', args),
    getImageData: (...args: unknown[]) => {
      const [sx, sy, sw, sh] = args as [number, number, number, number];
      log('getImageData', [sx, sy, sw, sh]);
      return createStubImageData(sw, sh);
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

  // The proxy records every property assignment (into `props`, declared above)
  // so tests can assert on applied state.
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

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.canvas = { height, width } as unknown as OffscreenCanvas | HTMLCanvasElement;
    this.ctx = createStubCtx(this.callLog);
  }

  resize(w: number, h: number): void {
    this.callLog.push({ args: [w, h], op: 'resize' });
    this.width = w;
    this.height = h;
  }
}

/**
 * Creates a `RasterBackend` whose surfaces are backed by a fake, node-safe
 * 2D context that records draw calls instead of executing them.
 */
export const createTestStubRasterBackend = (): StubRasterBackend => ({
  createImageBitmap: (source: ImageBitmapSource): Promise<ImageBitmap> => {
    void source;
    return Promise.resolve({ close: () => {}, height: 0, width: 0 } as unknown as ImageBitmap);
  },
  createSurface: (width: number, height: number): StubRasterSurface => new StubRasterSurfaceImpl(width, height),
  // Deterministic fake blob keyed on the surface size, so encode calls are
  // reproducible in node without touching a real canvas.
  encodeSurface: (surface: RasterSurface, type = 'image/png'): Promise<Blob> =>
    Promise.resolve(new Blob([`stub-surface-${surface.width}x${surface.height}`], { type })),
});
