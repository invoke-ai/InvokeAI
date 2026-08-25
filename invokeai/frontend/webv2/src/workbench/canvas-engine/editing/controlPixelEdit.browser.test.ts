import type { CanvasDocumentContractV2, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { BitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';

import { PixelEditController } from '@workbench/canvas-engine/controllers/controlPixelController';
import { createBitmapStore } from '@workbench/canvas-engine/document/bitmapStore';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { compositeDocument } from '@workbench/canvas-engine/render/compositor';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it, vi } from 'vitest';

import { bakePixelEditSurface } from './controlPixelEdit';

const IDENTITY = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 } as const;

const imageLayer = (overrides: Partial<CanvasRasterLayerContractV2> = {}): CanvasRasterLayerContractV2 => ({
  adjustments: { brightness: 0.5, contrast: 0, saturation: 0 },
  blendMode: 'normal',
  id: 'image',
  isEnabled: true,
  isLocked: false,
  name: 'Image',
  opacity: 1,
  source: { image: { height: 1, imageName: 'two-pixels', width: 2 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1.5, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
  ...overrides,
});

const inertBitmapStore = (): BitmapStore => ({
  discardLayer: vi.fn(),
  dispose: vi.fn(),
  flushPendingUploads: vi.fn(() => Promise.resolve()),
  isSelfEcho: vi.fn(() => false),
  markLayerDirty: vi.fn(),
  reset: vi.fn(),
  suspendLayer: vi.fn(() => vi.fn()),
});

const setup = (
  initialLayer: CanvasRasterLayerContractV2,
  bitmapStore: BitmapStore = inertBitmapStore(),
  resources: {
    backend?: RasterBackend;
    layers?: LayerCacheStore;
    onReplacement?: (layer: CanvasRasterLayerContractV2) => void;
  } = {}
) => {
  const backend = resources.backend ?? createDomRasterBackend();
  const layers = resources.layers ?? createLayerCacheStore(backend);
  const adjusted = createAdjustedSurfaceCache(backend);
  const transformOverrides = new Map<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >();
  const entry = layers.getOrCreate(initialLayer.id, 2, 1);
  entry.surface.ctx.putImageData(new ImageData(new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]), 2, 1), 0, 0);
  layers.publishPixels(initialLayer.id);
  let layer = initialLayer;
  const document: CanvasDocumentContractV2 = {
    background: 'transparent',
    bbox: { height: 4, width: 8, x: 0, y: 0 },
    height: 4,
    layers: [layer],
    selectedLayerId: layer.id,
    version: 2,
    width: 8,
  };
  const history = createHistory();
  const controller = new PixelEditController({
    applyImagePatch: vi.fn(),
    backend,
    bitmapStore,
    canEdit: () => true,
    deleteDerived: (layerId) => adjusted.delete(layerId),
    dispatchReplacement: (replacement) => {
      if (replacement.type !== 'raster') {
        throw new Error('expected raster replacement');
      }
      layer = replacement;
      document.layers = [replacement];
      resources.onReplacement?.(replacement);
    },
    endBurst: vi.fn(),
    getActiveProjectId: () => 'project',
    getAdjustedSurface: (candidate, candidateEntry) =>
      candidate.type === 'raster' ? adjusted.get(candidate.id, candidateEntry, candidate.adjustments) : null,
    getDocument: () => document,
    getTransformSession: () => null,
    history,
    installPrepared: (prepared) => {
      layers.installReplacement(prepared);
    },
    invalidate: vi.fn(),
    isCacheReady: () => true,
    isOperationIdle: () => true,
    layers,
    notifyPainted: (layerId) => {
      layers.publishPixels(layerId);
    },
    preparePixels: (layerId, rect, pixels) => layers.prepareReplacement(layerId, rect, pixels),
    projectId: 'project',
    publishStroke: vi.fn(),
    setTransformOverride: (layerId, transform) => {
      if (transform) {
        transformOverrides.set(layerId, transform);
      } else {
        transformOverrides.delete(layerId);
      }
    },
  });
  const composite = (): RasterSurface => {
    const target = backend.createSurface(document.width, document.height);
    compositeDocument(target, document, layers, IDENTITY, {
      adjustedSurface: (candidate, candidateEntry) =>
        controller.isOpenFor([candidate.id])
          ? null
          : candidate.type === 'raster'
            ? adjusted.get(candidate.id, candidateEntry, candidate.adjustments)
            : null,
      imageSmoothing: false,
      transformOverrides,
    });
    return target;
  };
  return { adjusted, backend, bitmapStore, composite, controller, document, getLayer: () => layer, history, layers };
};

const pixels = (surface: RasterSurface): number[] => [
  ...surface.ctx.getImageData(0, 0, surface.width, surface.height).data,
];

describe('image-layer erasing with real browser pixels', () => {
  it('preserves adjustment-before-transform rendering for untouched pixels', () => {
    const original = imageLayer();
    const h = setup(original);
    const originalEntry = h.layers.get(original.id)!;
    const adjusted = h.adjusted.get(original.id, originalEntry, original.adjustments)!;
    const expected = bakePixelEditSurface({
      backend: h.backend,
      source: adjusted,
      sourceRect: originalEntry.rect,
      transform: original.transform,
    });
    const wrongOrder = bakePixelEditSurface({
      backend: h.backend,
      source: originalEntry.surface,
      sourceRect: originalEntry.rect,
      transform: original.transform,
    });
    const expectedPixels = pixels(expected.surface);
    expect(expectedPixels).not.toEqual(pixels(wrongOrder.surface));
    const displayBefore = pixels(h.composite());

    const transaction = h.controller.begin(original.id);
    expect(transaction).not.toBeNull();
    const preview = h.layers.get(original.id)!;
    const beforeStroke = preview.surface.ctx.getImageData(0, 0, preview.rect.width, preview.rect.height);
    expect([...beforeStroke.data]).toEqual(expectedPixels);
    expect(pixels(h.composite())).toEqual(displayBefore);

    // Erase only the final transformed pixel. The preceding pixels must stay
    // byte-identical to the pre-edit display, not a raw-then-adjusted variant.
    preview.surface.ctx.clearRect(preview.rect.width - 1, 0, 1, 1);
    const afterStroke = preview.surface.ctx.getImageData(0, 0, preview.rect.width, preview.rect.height);
    transaction!.commitStroke({
      afterImageData: afterStroke,
      beforeImageData: beforeStroke,
      dirtyRect: { ...preview.rect },
      layerId: original.id,
      tool: 'eraser',
    });

    expect(Array.from(afterStroke.data.slice(0, -4))).toEqual(expectedPixels.slice(0, -4));
    expect(Array.from(afterStroke.data.slice(-4))).toEqual([0, 0, 0, 0]);
    expect(h.getLayer()).toMatchObject({ source: { type: 'paint' }, transform: { scaleX: 1, scaleY: 1 } });
    expect(h.getLayer()).not.toHaveProperty('adjustments');

    h.history.undo();
    expect(h.getLayer()).toEqual(original);
    expect(pixels(h.layers.get(original.id)!.surface)).toEqual([0, 0, 0, 255, 255, 255, 255, 255]);
    h.controller.dispose();
  });

  it('collapses a fully erased image materialization instead of uploading transparent padding', async () => {
    const original = imageLayer({
      adjustments: undefined,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    const uploadImage = vi.fn(() => Promise.resolve({ height: 1, imageName: 'unexpected', width: 2 }));
    const backend = createDomRasterBackend();
    const layers = createLayerCacheStore(backend);
    let currentSource = original.source;
    const bitmapStore = createBitmapStore({
      debounceMs: 60_000,
      dispatch: vi.fn(() => true),
      encodeSurface: (surface) => backend.encodeSurface(surface),
      getLayerSource: () => currentSource,
      getLayerSurface: (layerId) => {
        const entry = layers.get(layerId);
        return entry && entry.rect.width > 0 && entry.rect.height > 0
          ? { offset: { x: entry.rect.x, y: entry.rect.y }, surface: entry.surface }
          : null;
      },
      trimLayerPixels: (layerId) => trimPaintCacheToAlpha({ isLayerBusy: () => false, layers }, layerId),
      uploadImage,
    });

    // Use the same real layer store in the controller and persistence seams.
    const h = setup(original, bitmapStore, {
      backend,
      layers,
      onReplacement: (replacement) => {
        currentSource = replacement.source;
      },
    });
    const transaction = h.controller.begin(original.id)!;
    const preview = layers.get(original.id)!;
    const beforeStroke = preview.surface.ctx.getImageData(0, 0, preview.rect.width, preview.rect.height);
    preview.surface.ctx.clearRect(0, 0, preview.rect.width, preview.rect.height);
    const afterStroke = preview.surface.ctx.getImageData(0, 0, preview.rect.width, preview.rect.height);
    transaction.commitStroke({
      afterImageData: afterStroke,
      beforeImageData: beforeStroke,
      dirtyRect: { ...preview.rect },
      layerId: original.id,
      tool: 'eraser',
    });

    await bitmapStore.flushPendingUploads();

    expect(layers.get(original.id)!.rect).toEqual({ height: 0, width: 0, x: 0, y: 0 });
    expect(currentSource).toEqual({ bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' });
    expect(uploadImage).not.toHaveBeenCalled();
    h.controller.dispose();
    bitmapStore.dispose();
  });
});
