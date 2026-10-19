import type {
  CanvasAdjustmentEntry,
  CanvasDocumentContractV3,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { applyAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { ThumbnailController } from './thumbnailController';

describe('ThumbnailController budget', () => {
  it('returns over-budget before rasterizing a missing thumbnail cache', async () => {
    const layer = {
      blendMode: 'normal' as const,
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 100, imageName: 'layer.png', width: 100 }, type: 'image' as const },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 100,
    };
    const rasterize = vi.fn();
    const controller = new ThumbnailController({
      backend: createTestStubRasterBackend(),
      getActiveProjectId: () => 'p',
      getCheckerboard: vi.fn(),
      getDocument: () => document,
      getEntry: () => undefined,
      getMaskPattern: () => null,
      isDisposed: () => false,
      isSupportedSource: () => true,
      peekAdjustedSurface: () => null,
      projectId: 'p',
      rasterize,
      reportError: vi.fn(),
      reserve: () => ({ availableBytes: 0, requestedBytes: 80_000, status: 'over-budget' }),
      setStatus: vi.fn(),
    });

    await expect(controller.request('layer')).resolves.toBe('over-budget');
    expect(rasterize).not.toHaveBeenCalled();
  });
});

describe('ThumbnailController adjusted rasters', () => {
  const grain: CanvasAdjustmentEntry = {
    amount: 0.4,
    distribution: 'gaussian',
    id: 'n',
    isEnabled: true,
    monochrome: false,
    seed: 3,
    type: 'noise',
  };
  const invert: CanvasAdjustmentEntry = { id: 'i', isEnabled: true, type: 'invert' };

  /** A 200×100 cache drawn into a 50×25 thumbnail (scale ¼), with the given display-copy peek. */
  const drawThumbnail = (peek: (surface: RasterSurface) => RasterSurface | null) => {
    const backend = createTestStubRasterBackend();
    const created: StubRasterSurface[] = [];
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('layer', 200, 100);
    entry.hasPublishedPixels = true;
    const layer: CanvasRasterLayerContractV2 = {
      adjustments: [invert, grain],
      blendMode: 'normal',
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 100, imageName: 'layer.png', width: 200 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 200, x: 0, y: 0 },
      height: 100,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 200,
    };
    const controller = new ThumbnailController({
      backend: {
        ...backend,
        createSurface: (width, height) => {
          const surface = backend.createSurface(width, height);
          created.push(surface);
          return surface;
        },
      },
      getActiveProjectId: () => 'p',
      getCheckerboard: () => backend.createSurface(8, 8),
      getDocument: () => document,
      getEntry: () => entry,
      getMaskPattern: () => null,
      isDisposed: () => false,
      isSupportedSource: () => true,
      peekAdjustedSurface: (_layer, peeked) => peek(peeked.surface),
      projectId: 'p',
      rasterize: vi.fn(),
      reportError: vi.fn(),
      setStatus: vi.fn(),
    });
    const ctx = { clearRect: vi.fn(), createPattern: () => null, drawImage: vi.fn(), globalAlpha: 1 };
    const target = { getContext: () => ctx, height: 0, width: 0 } as unknown as HTMLCanvasElement;
    expect(controller.draw('layer', target, 50)).toBe(true);
    const thumbnail = created.find((surface) => surface.width === 50 && surface.height === 25)!;
    return { cache: entry.surface, thumbnail };
  };

  it("downscales the display's current adjusted copy without applying the stack again", () => {
    const displayed = createTestStubRasterBackend().createSurface(200, 100);
    const { thumbnail } = drawThumbnail(() => displayed);

    const drawn = thumbnail.callLog.find((call) => call.op === 'drawImage');
    expect(drawn?.args).toEqual([displayed.canvas, 0, 0, 50, 25]);
    expect(thumbnail.callLog.some((call) => call.op === 'getImageData' || call.op === 'putImageData')).toBe(false);
  });

  it('without a display copy adjusts the thumbnail itself, scaling noise σ by the thumbnail scale', () => {
    const { cache, thumbnail } = drawThumbnail(() => null);

    expect(thumbnail.callLog.find((call) => call.op === 'drawImage')?.args[0]).toBe(cache.canvas);
    const written = thumbnail.callLog.find((call) => call.op === 'putImageData')!.args[0] as ImageData;
    // The stub reads back opaque black; a quarter-size thumbnail carries a quarter of the grain's σ.
    const expected = { data: new Uint8ClampedArray(50 * 25 * 4), height: 25, width: 50 } as ImageData;
    for (let i = 3; i < expected.data.length; i += 4) {
      expected.data[i] = 255;
    }
    applyAdjustments(expected, [invert, { ...grain, amount: 0.1 } as CanvasAdjustmentEntry], { x: 0, y: 0 });
    expect([...written.data]).toEqual([...expected.data]);
  });
});
