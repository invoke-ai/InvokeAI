import type {
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import type { RasterBackend, RasterSurface } from './raster';

import { sampleDocumentColor } from './colorSample';
import { createLayerCacheStore } from './layerCache';

/** A fake surface that records `drawImage`/`setTransform` calls, for asserting traversal order and translation math. */
interface FakeSurface extends RasterSurface {
  drawnCanvases: unknown[];
  transforms: number[][];
}

/** A {@link RasterBackend} test double that also exposes every surface it created, for assertions. */
interface FixedPixelBackend extends RasterBackend {
  __surfaces: FakeSurface[];
}

/**
 * A minimal `RasterBackend` whose scratch surfaces report a single fixed pixel
 * for every `getImageData` call — enough to test `sampleDocumentColor`'s
 * bounds/alpha/traversal logic without modeling real canvas compositing.
 */
const createFixedPixelBackend = (pixel: readonly [number, number, number, number]): FixedPixelBackend => {
  const createdSurfaces: FakeSurface[] = [];

  return {
    createImageBitmap: () => Promise.resolve({} as ImageBitmap),
    createSurface: (width: number, height: number): FakeSurface => {
      const drawnCanvases: unknown[] = [];
      const transforms: number[][] = [];
      const canvas = { height, width } as unknown as OffscreenCanvas;
      const ctx = {
        clearRect: () => {},
        drawImage: (image: unknown) => drawnCanvases.push(image),
        getImageData: () => ({ data: Uint8ClampedArray.from(pixel), height: 1, width: 1 }) as unknown as ImageData,
        restore: () => {},
        save: () => {},
        setTransform: (...args: number[]) => transforms.push(args),
      } as unknown as OffscreenCanvasRenderingContext2D;
      const surface: FakeSurface = {
        canvas,
        ctx,
        drawnCanvases,
        height,
        resize: () => {},
        transforms,
        width,
      };
      createdSurfaces.push(surface);
      return surface;
    },
    encodeSurface: () => Promise.resolve(new Blob()),
    __surfaces: createdSurfaces,
  };
};

const rasterLayer = (id: string, overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: null,
  version: 2,
  width: 100,
});

describe('sampleDocumentColor', () => {
  it('returns null for a point outside the document bounds (never allocates a scratch surface)', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 255]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([]);

    expect(sampleDocumentColor(doc, layers, backend, { x: -1, y: 5 })).toBeNull();
    expect(sampleDocumentColor(doc, layers, backend, { x: 100, y: 5 })).toBeNull();
    expect(sampleDocumentColor(doc, layers, backend, { x: 5, y: -1 })).toBeNull();
    expect(sampleDocumentColor(doc, layers, backend, { x: 5, y: 100 })).toBeNull();
    expect(backend.__surfaces).toHaveLength(0);
  });

  it('returns null when the composited pixel is fully transparent', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 0]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([rasterLayer('a')]);
    layers.getOrCreate('a', 100, 100);

    expect(sampleDocumentColor(doc, layers, backend, { x: 5, y: 5 })).toBeNull();
  });

  it('returns the composited rgba when the sampled pixel has non-zero alpha', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 128]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([rasterLayer('a')]);
    layers.getOrCreate('a', 100, 100);

    expect(sampleDocumentColor(doc, layers, backend, { x: 5, y: 5 })).toEqual({ a: 128, b: 30, g: 20, r: 10 });
  });

  it('draws renderable layers bottom-to-top, skipping disabled and uncached layers', () => {
    const backend = createFixedPixelBackend([1, 2, 3, 255]);
    const layers = createLayerCacheStore(backend);
    const topEntry = layers.getOrCreate('top', 100, 100);
    const bottomEntry = layers.getOrCreate('bottom', 100, 100);
    // 'disabled' and 'nocache' are deliberately excluded from compositing.
    const doc = makeDoc([
      rasterLayer('top'),
      rasterLayer('disabled', { isEnabled: false }),
      rasterLayer('nocache'),
      rasterLayer('bottom'),
    ]);

    sampleDocumentColor(doc, layers, backend, { x: 5, y: 5 });

    const scratch = backend.__surfaces.at(-1)!;
    expect(scratch.drawnCanvases).toEqual([bottomEntry.surface.canvas, topEntry.surface.canvas]);
  });

  it('translates the view so the floored sample point lands at the scratch origin', () => {
    const backend = createFixedPixelBackend([1, 2, 3, 255]);
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('a', 100, 100);
    const doc = makeDoc([rasterLayer('a')]);

    sampleDocumentColor(doc, layers, backend, { x: 12.7, y: 34.2 });

    const scratch = backend.__surfaces.at(-1)!;
    // Identity layer transform composed with the sample-point translation: e/f
    // carry the floored, negated point (no per-layer offset/scale/rotation).
    // The last `setTransform` is the per-layer draw (the first is the initial reset).
    expect(scratch.transforms.at(-1)).toEqual([1, 0, 0, 1, -12, -34]);
  });
});
