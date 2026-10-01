import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { describe, expect, it } from 'vitest';

import type { RasterBackend, RasterSurface } from './raster';

import { createColorSampler } from './colorSample';
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

/** Fixed-pixel scratch surfaces isolate sampling bounds, alpha and traversal without simulating compositing. */
const createFixedPixelBackend = (pixel: readonly [number, number, number, number]): FixedPixelBackend => {
  const createdSurfaces: FakeSurface[] = [];

  return {
    createImageBitmap: () => Promise.resolve({} as ImageBitmap),
    createSurface: (width: number, height: number): FakeSurface => {
      const drawnCanvases: unknown[] = [];
      const transforms: number[][] = [];
      let hasDrawnPixels = false;
      const canvas = { height, width } as unknown as OffscreenCanvas;
      const ctx = {
        clearRect: () => {
          hasDrawnPixels = false;
        },
        drawImage: (image: unknown) => {
          drawnCanvases.push(image);
          hasDrawnPixels = true;
        },
        getImageData: () =>
          ({
            data: Uint8ClampedArray.from(hasDrawnPixels ? pixel : [0, 0, 0, 0]),
            height: 1,
            width: 1,
          }) as unknown as ImageData,
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
        resizePreserving: () => {},
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

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId: null,
  version: 3,
  width: 100,
});

describe('createColorSampler', () => {
  it('returns null for non-finite points without allocating a scratch surface', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 255]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([]);

    expect(createColorSampler(backend).sample(doc, layers, { x: Number.NaN, y: 5 })).toBeNull();
    expect(createColorSampler(backend).sample(doc, layers, { x: 5, y: Number.POSITIVE_INFINITY })).toBeNull();
    expect(backend.__surfaces).toHaveLength(0);
  });

  it('returns null when no layer covers a point beyond the document bounds', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 255]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([]);

    const sampler = createColorSampler(backend);
    expect(sampler.sample(doc, layers, { x: -1, y: 5 })).toBeNull();
    expect(sampler.sample(doc, layers, { x: 100, y: 5 })).toBeNull();
    expect(sampler.sample(doc, layers, { x: 5, y: -1 })).toBeNull();
    expect(sampler.sample(doc, layers, { x: 5, y: 100 })).toBeNull();
    // One scratch serves every sample.
    expect(backend.__surfaces).toHaveLength(1);
  });

  it('returns null when the composited pixel is fully transparent', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 0]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([rasterLayer('a')]);
    layers.getOrCreate('a', 100, 100);

    expect(createColorSampler(backend).sample(doc, layers, { x: 5, y: 5 })).toBeNull();
  });

  it('returns the composited rgba when the sampled pixel has non-zero alpha', () => {
    const backend = createFixedPixelBackend([10, 20, 30, 128]);
    const layers = createLayerCacheStore(backend);
    const doc = makeDoc([rasterLayer('a')]);
    layers.getOrCreate('a', 100, 100);

    expect(createColorSampler(backend).sample(doc, layers, { x: 5, y: 5 })).toEqual({ a: 128, b: 30, g: 20, r: 10 });
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

    createColorSampler(backend).sample(doc, layers, { x: 5, y: 5 });

    const scratch = backend.__surfaces.at(-1)!;
    expect(scratch.drawnCanvases).toEqual([bottomEntry.surface.canvas, topEntry.surface.canvas]);
  });

  it('skips zero-sized cached layer surfaces', () => {
    const backend = createFixedPixelBackend([1, 2, 3, 0]);
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('empty', 0, 0);

    expect(createColorSampler(backend).sample(makeDoc([rasterLayer('empty')]), layers, { x: 5, y: 5 })).toBeNull();
    expect(backend.__surfaces.at(-1)?.drawnCanvases).toEqual([]);
  });

  it('translates the view so the floored sample point lands at the scratch origin', () => {
    const backend = createFixedPixelBackend([1, 2, 3, 255]);
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('a', 100, 100);
    const doc = makeDoc([rasterLayer('a')]);

    createColorSampler(backend).sample(doc, layers, { x: 12.7, y: 34.2 });

    const scratch = backend.__surfaces.at(-1)!;
    // Identity-layer sampling translates by the negated floored point; inspect the final per-layer transform after
    // reset.
    expect(scratch.transforms.at(-1)).toEqual([1, 0, 0, 1, -12, -34]);
  });

  it('reuses a sample of the same pixel until the document or cached pixels change', () => {
    const backend = createFixedPixelBackend([1, 2, 3, 255]);
    const layers = createLayerCacheStore(backend);
    layers.getOrCreate('a', 100, 100);
    const doc = makeDoc([rasterLayer('a')]);
    const sampler = createColorSampler(backend);
    const draws = (): number => backend.__surfaces.at(-1)!.drawnCanvases.length;

    const first = sampler.sample(doc, layers, { x: 5.2, y: 5.9 });
    expect(draws()).toBe(1);
    expect(sampler.sample(doc, layers, { x: 5.8, y: 5.1 })).toBe(first);
    expect(draws()).toBe(1);

    sampler.sample(doc, layers, { x: 6, y: 5 });
    expect(draws()).toBe(2);
    layers.publishPixels('a');
    sampler.sample(doc, layers, { x: 6, y: 5 });
    expect(draws()).toBe(3);
    sampler.sample({ ...doc }, layers, { x: 6, y: 5 });
    expect(draws()).toBe(4);
  });
});
