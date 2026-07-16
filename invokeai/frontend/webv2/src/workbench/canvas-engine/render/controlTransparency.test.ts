import { describe, expect, it } from 'vitest';

import type { RasterBackend, RasterSurface } from './raster';

import { applyLightnessToAlpha, renderControlTransparency } from './controlTransparency';

/** Builds a flat RGBA `Uint8ClampedArray` from a list of [r, g, b, a] pixels. */
const pixels = (...rgba: [number, number, number, number][]): Uint8ClampedArray => new Uint8ClampedArray(rgba.flat());

describe('applyLightnessToAlpha', () => {
  it('sets alpha to the HSL lightness of each pixel, clamped to the existing alpha', () => {
    const data = pixels(
      [255, 255, 255, 255], // white: lightness 255 -> alpha stays 255
      [0, 0, 0, 255], // black: lightness 0 -> alpha becomes 0
      [128, 128, 128, 255], // mid-gray: lightness 128 -> alpha 128
      [100, 50, 200, 255], // colored: min 50, max 200 -> lightness 125 -> alpha 125
      [255, 255, 255, 100] // opaque-white but a=100: lightness 255, min(100, 255) keeps 100
    );

    applyLightnessToAlpha(data);

    // Assert the resulting alpha channel for each pixel.
    expect(data[3]).toBe(255);
    expect(data[7]).toBe(0);
    expect(data[11]).toBe(128);
    expect(data[15]).toBe(125);
    expect(data[19]).toBe(100);
  });

  it('leaves the RGB channels untouched', () => {
    const data = pixels(
      [255, 255, 255, 255],
      [0, 0, 0, 255],
      [128, 128, 128, 255],
      [100, 50, 200, 255],
      [255, 255, 255, 100]
    );

    applyLightnessToAlpha(data);

    // Only the alpha byte of every quad may change; the RGB bytes are preserved.
    expect([data[0], data[1], data[2]]).toEqual([255, 255, 255]);
    expect([data[4], data[5], data[6]]).toEqual([0, 0, 0]);
    expect([data[8], data[9], data[10]]).toEqual([128, 128, 128]);
    expect([data[12], data[13], data[14]]).toEqual([100, 50, 200]);
    expect([data[16], data[17], data[18]]).toEqual([255, 255, 255]);
  });

  it('mutates the buffer in place and returns void', () => {
    const data = pixels([0, 0, 0, 255]);
    const same = data;

    const result = applyLightnessToAlpha(data);

    // Same reference, mutated in place; the function returns nothing.
    expect(result).toBeUndefined();
    expect(data).toBe(same);
    expect(same[3]).toBe(0);
  });

  it('handles an empty buffer without throwing', () => {
    const data = new Uint8ClampedArray(0);
    expect(() => applyLightnessToAlpha(data)).not.toThrow();
    expect(data).toHaveLength(0);
  });
});

/** A minimal `RasterSurface` recording drawImage/getImageData/putImageData against a fed bitmap. */
interface FakeSurface extends RasterSurface {
  readonly drawImageArgs: unknown[][];
  readonly getImageDataArgs: unknown[][];
  readonly putImageDataCalls: ImageData[];
}

/**
 * A minimal but faithful `RasterBackend` for `renderControlTransparency`: every
 * surface's `getImageData` returns a fresh copy of `sourcePixels`, and its
 * `putImageData` is recorded so the transform's output can be inspected.
 */
const createFakeBackend = (sourcePixels: Uint8ClampedArray): RasterBackend & { readonly created: FakeSurface[] } => {
  const created: FakeSurface[] = [];

  const createSurface = (width: number, height: number): FakeSurface => {
    const drawImageArgs: unknown[][] = [];
    const getImageDataArgs: unknown[][] = [];
    const putImageDataCalls: ImageData[] = [];

    const ctx = {
      clearRect: () => {},
      drawImage: (...args: unknown[]) => {
        drawImageArgs.push(args);
      },
      getImageData: (...args: unknown[]): ImageData => {
        getImageDataArgs.push(args);
        // Return a fresh copy so callers mutate their own buffer, not the source.
        const data = new Uint8ClampedArray(sourcePixels);
        return { colorSpace: 'srgb', data, height, width } as unknown as ImageData;
      },
      globalAlpha: 1,
      globalCompositeOperation: 'source-over' as GlobalCompositeOperation,
      putImageData: (imageData: ImageData) => {
        putImageDataCalls.push(imageData);
      },
      setTransform: () => {},
    } as unknown as OffscreenCanvasRenderingContext2D;

    const surface: FakeSurface = {
      canvas: { height, width } as unknown as OffscreenCanvas,
      ctx,
      drawImageArgs,
      getImageDataArgs,
      height,
      putImageDataCalls,
      resize: () => {},
      width,
    };
    created.push(surface);
    return surface;
  };

  return {
    createImageBitmap: () => Promise.resolve({ close: () => {}, height: 0, width: 0 } as unknown as ImageBitmap),
    createSurface,
    created,
    encodeSurface: (surface: RasterSurface, type = 'image/png') =>
      Promise.resolve(new Blob([`fake-${surface.width}x${surface.height}`], { type })),
  };
};

describe('renderControlTransparency', () => {
  it('allocates the output surface at the requested dimensions', () => {
    const backend = createFakeBackend(pixels([0, 0, 0, 255]));
    const cache = backend.createSurface(64, 48);

    const out = renderControlTransparency(backend, cache, 64, 48);

    expect(out.width).toBe(64);
    expect(out.height).toBe(48);
    // The returned surface is a NEW allocation, not the cache itself.
    expect(out).not.toBe(cache);
  });

  it('draws the cache canvas into the output surface', () => {
    const backend = createFakeBackend(pixels([0, 0, 0, 255]));
    const cache = backend.createSurface(10, 10);

    const out = renderControlTransparency(backend, cache, 10, 10) as FakeSurface;

    // The cache's canvas is blitted into the output at the origin.
    expect(out.drawImageArgs).toHaveLength(1);
    expect(out.drawImageArgs[0]![0]).toBe(cache.canvas);
    expect(out.drawImageArgs[0]!.slice(1)).toEqual([0, 0]);
  });

  it('applies the lightness->alpha transform: opaque black pixels become fully transparent', () => {
    // Feed a bitmap of opaque black pixels; after the transform each alpha is 0.
    const backend = createFakeBackend(pixels([0, 0, 0, 255], [0, 0, 0, 255]));
    const cache = backend.createSurface(2, 1);

    const out = renderControlTransparency(backend, cache, 2, 1) as FakeSurface;

    // getImageData was read over the full surface rect, and the result put back.
    expect(out.getImageDataArgs).toHaveLength(1);
    expect(out.getImageDataArgs[0]).toEqual([0, 0, 2, 1]);
    expect(out.putImageDataCalls).toHaveLength(1);

    const result = out.putImageDataCalls[0]!.data;
    // Black -> lightness 0 -> alpha 0; RGB channels are preserved.
    expect(result[3]).toBe(0);
    expect(result[7]).toBe(0);
    expect([result[0], result[1], result[2]]).toEqual([0, 0, 0]);
  });

  it('applies the lightness->alpha transform to a mixed bitmap', () => {
    const backend = createFakeBackend(
      pixels(
        [255, 255, 255, 255], // white -> alpha 255
        [0, 0, 0, 255], // black -> alpha 0
        [100, 50, 200, 255] // colored -> lightness 125 -> alpha 125
      )
    );
    const cache = backend.createSurface(3, 1);

    const out = renderControlTransparency(backend, cache, 3, 1) as FakeSurface;

    const result = out.putImageDataCalls[0]!.data;
    expect(result[3]).toBe(255);
    expect(result[7]).toBe(0);
    expect(result[11]).toBe(125);
    // RGB untouched by the transform.
    expect([result[8], result[9], result[10]]).toEqual([100, 50, 200]);
  });
});
