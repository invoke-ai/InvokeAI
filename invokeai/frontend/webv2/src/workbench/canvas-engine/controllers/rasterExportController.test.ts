import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { RasterExportController } from './rasterExportController';
import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';

const pin = () => ({ release: () => undefined });

describe('RasterExportController budget', () => {
  it('returns over-budget before allocating a baked raster export', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('layer', 100, 100);
    entry.hasPublishedPixels = true;
    entry.stale = false;
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
    const reserve = vi.fn(() => ({ availableBytes: 0, requestedBytes: 40_000, status: 'over-budget' as const }));
    const controller = new RasterExportController({
      backend,
      captureGuard: () => ({ cacheVersion: 1, documentGeneration: 1, layer, layerId: 'layer', projectId: 'p' }),
      getDocument: () => document,
      getOrStartRasterization: () => Promise.resolve('published'),
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin,
      reserve,
    });

    await expect(controller.baked('layer')).resolves.toEqual({ status: 'over-budget' });
    expect(reserve).toHaveBeenCalledWith(40_000);
  });

  it('holds the baked-surface reservation until blob encoding settles', async () => {
    const stub = createTestStubRasterBackend();
    let resolveEncode!: (blob: Blob) => void;
    const encodeSurface = vi.fn(
      () =>
        new Promise<Blob>((resolve) => {
          resolveEncode = resolve;
        })
    );
    const backend = { ...stub, encodeSurface };
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('layer', 100, 100);
    entry.hasPublishedPixels = true;
    entry.stale = false;
    const layer = {
      adjustments: [
        { brightness: 0.1, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      ],
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
    const release = vi.fn();
    const reserve = vi.fn(() => ({ lease: { release }, status: 'ok' as const }));
    const controller = new RasterExportController({
      backend,
      captureGuard: () => ({ cacheVersion: 1, documentGeneration: 1, layer, layerId: 'layer', projectId: 'p' }),
      getDocument: () => document,
      getOrStartRasterization: () => Promise.resolve('published'),
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin,
      reserve,
    });

    const pending = controller.blob('layer');
    await vi.waitFor(() => expect(encodeSurface).toHaveBeenCalledOnce());
    expect(release).not.toHaveBeenCalled();

    resolveEncode(new Blob(['png']));
    await expect(pending).resolves.toMatchObject({ status: 'ok' });
    expect(release).toHaveBeenCalledOnce();
  });

  it('transfers the baked-surface reservation to the caller until idempotent release', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('layer', 100, 100);
    entry.hasPublishedPixels = true;
    entry.stale = false;
    const layer = {
      adjustments: [
        { brightness: 0.1, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      ],
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
    const release = vi.fn();
    const reserve = vi.fn(() => ({ lease: { release }, status: 'ok' as const }));
    const controller = new RasterExportController({
      backend,
      captureGuard: () => ({ cacheVersion: 1, documentGeneration: 1, layer, layerId: 'layer', projectId: 'p' }),
      getDocument: () => document,
      getOrStartRasterization: () => Promise.resolve('published'),
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin,
      reserve,
    });

    const result = await controller.baked('layer');
    expect(result.status).toBe('ok');
    expect(reserve).toHaveBeenCalledWith(80_000);
    expect(release).not.toHaveBeenCalled();
    if (result.status === 'ok') {
      result.release();
      result.release();
    }
    expect(release).toHaveBeenCalledOnce();
  });

  it('waits for custom font readiness before returning cached text pixels', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('text-layer', 80, 32);
    entry.hasPublishedPixels = true;
    entry.renderedFontFamily = '__invoke_font_1';
    entry.stale = false;
    const source = {
      align: 'left' as const,
      color: '#fff',
      content: 'custom',
      fontFamily: 'Catalog Family',
      fontRef: { contentHash: 'hash-1', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
      fontSize: 24,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text' as const,
    };
    const layer = {
      blendMode: 'normal' as const,
      id: 'text-layer',
      isEnabled: true,
      isLocked: false,
      name: 'Text',
      opacity: 1,
      source,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 32, width: 80, x: 0, y: 0 },
      height: 32,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 80,
    };
    let resolveFont!: (family: string) => void;
    const waitForFont = vi.fn(
      () =>
        new Promise<string>((resolve) => {
          resolveFont = resolve;
        })
    );
    const getOrStartRasterization = vi.fn(() => Promise.resolve<'published'>('published'));
    const controller = new RasterExportController({
      backend,
      captureGuard: () => ({ cacheVersion: 1, documentGeneration: 1, layer, layerId: 'text-layer', projectId: 'p' }),
      getDocument: () => document,
      getOrStartRasterization,
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin,
      waitForFont,
    });

    const pending = controller.rasterize('text-layer');
    expect(waitForFont).toHaveBeenCalledWith(source, undefined);
    expect(getOrStartRasterization).not.toHaveBeenCalled();
    resolveFont('__invoke_font_1');
    await expect(pending).resolves.toMatchObject({ status: 'ok' });
  });

  it('re-rasterizes fallback text once when a missing font recovers before export', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const entry = layers.getOrCreate('text-layer', 80, 32);
    entry.hasPublishedPixels = true;
    entry.renderedFontFamily = 'Catalog Family';
    entry.stale = false;
    const source = {
      align: 'left' as const,
      color: '#fff',
      content: 'custom',
      fontFamily: 'Catalog Family',
      fontRef: { contentHash: 'hash-1', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
      fontSize: 24,
      fontWeight: 400,
      lineHeight: 1.2,
      type: 'text' as const,
    };
    const layer = {
      blendMode: 'normal' as const,
      id: 'text-layer',
      isEnabled: true,
      isLocked: false,
      name: 'Text',
      opacity: 1,
      source,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 32, width: 80, x: 0, y: 0 },
      height: 32,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 80,
    };
    const waitForFont = vi
      .fn()
      .mockRejectedValueOnce(new Error('font unavailable'))
      .mockResolvedValue('__invoke_font_1');
    const invalidateLayerCache = vi.fn((layerId: string) => layers.invalidate(layerId));
    const rasterizedCustomPixels = vi.fn(() => {
      entry.renderedFontFamily = '__invoke_font_1';
      entry.stale = false;
      entry.hasPublishedPixels = true;
      entry.surface.ctx.fillText('custom pixels', 0, 0);
    });
    const getOrStartRasterization = vi.fn(() => {
      rasterizedCustomPixels();
      return Promise.resolve('published' as const);
    });
    const controller = new RasterExportController({
      backend,
      captureGuard: () => ({
        cacheVersion: entry.version,
        documentGeneration: 1,
        layer,
        layerId: 'text-layer',
        projectId: 'p',
      }),
      getDocument: () => document,
      getOrStartRasterization,
      invalidateLayerCache,
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin,
      waitForFont,
    });

    await expect(controller.rasterize('text-layer')).resolves.toEqual({ status: 'not-ready' });
    expect(getOrStartRasterization).not.toHaveBeenCalled();

    await expect(controller.rasterize('text-layer')).resolves.toMatchObject({ status: 'ok' });
    expect(invalidateLayerCache).toHaveBeenCalledOnce();
    expect(getOrStartRasterization).toHaveBeenCalledOnce();
    expect(rasterizedCustomPixels).toHaveBeenCalledOnce();
    expect(entry.renderedFontFamily).toBe('__invoke_font_1');

    await expect(controller.rasterize('text-layer')).resolves.toMatchObject({ status: 'ok' });
    expect(invalidateLayerCache).toHaveBeenCalledOnce();
    expect(getOrStartRasterization).toHaveBeenCalledOnce();
  });

  it('pins the source for the whole read lease, on cached and freshly rasterized paths', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const memory = new RasterMemoryBudgetController();
    const layer = {
      blendMode: 'normal' as const,
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 10, imageName: 'layer.png', width: 10 }, type: 'image' as const },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 10, width: 10, x: 0, y: 0 },
      height: 10,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 10,
    };
    let pinnedDuringRasterization = false;
    const controller = new RasterExportController({
      backend,
      captureGuard: (captured, entry) => ({
        cacheVersion: entry.version,
        documentGeneration: 1,
        layer: captured,
        layerId: captured.id,
        projectId: 'p',
      }),
      getDocument: () => document,
      getOrStartRasterization: () => {
        pinnedDuringRasterization = memory.isPinned('layer');
        layers.getOrCreateRect('layer', { height: 10, width: 10, x: 0, y: 0 });
        layers.publishPixels('layer');
        return Promise.resolve('published');
      },
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin: (layerId) => memory.pin(layerId),
    });

    const rasterized = await controller.rasterize('layer');
    expect(pinnedDuringRasterization).toBe(true);
    expect(rasterized.status).toBe('ok');
    expect(memory.isPinned('layer')).toBe(true);
    if (rasterized.status === 'ok') {
      rasterized.release();
      rasterized.release();
    }
    expect(memory.isPinned('layer')).toBe(false);

    const cached = await controller.rasterize('layer');
    expect(memory.isPinned('layer')).toBe(true);
    if (cached.status === 'ok') {
      cached.release();
    }
    expect(memory.isPinned('layer')).toBe(false);
  });

  it.each([
    ['aborted', 'aborted', { height: 10, width: 10, x: 0, y: 0 }],
    ['stale', 'not-ready', { height: 10, width: 10, x: 0, y: 0 }],
    ['published', 'empty', { height: 0, width: 0, x: 0, y: 0 }],
  ] as const)('releases the pin when a %s rasterization reads as %s', async (outcome, status, publishedRect) => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const memory = new RasterMemoryBudgetController();
    const layer = {
      blendMode: 'normal' as const,
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 10, imageName: 'layer.png', width: 10 }, type: 'image' as const },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 10, width: 10, x: 0, y: 0 },
      height: 10,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 10,
    };
    const controller = new RasterExportController({
      backend,
      captureGuard: (captured, entry) => ({
        cacheVersion: entry.version,
        documentGeneration: 1,
        layer: captured,
        layerId: captured.id,
        projectId: 'p',
      }),
      getDocument: () => document,
      getOrStartRasterization: () => {
        layers.publishRasterized(
          'layer',
          publishedRect,
          backend.createSurface(publishedRect.width, publishedRect.height)
        );
        return Promise.resolve(outcome);
      },
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin: (layerId) => memory.pin(layerId),
    });

    await expect(controller.rasterize('layer')).resolves.toEqual({ status });
    expect(memory.isPinned('layer')).toBe(false);
  });

  it('releases the source pin once an adjusted copy is made, keeping its reservation until release', async () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const memory = new RasterMemoryBudgetController();
    const layer = {
      adjustments: [
        { brightness: 0.1, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
      ],
      blendMode: 'normal' as const,
      id: 'layer',
      isEnabled: true,
      isLocked: false,
      name: 'Layer',
      opacity: 1,
      source: { image: { height: 10, imageName: 'layer.png', width: 10 }, type: 'image' as const },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster' as const,
    };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 10, width: 10, x: 0, y: 0 },
      height: 10,
      stacks: stacksFrom([layer]),
      selectedLayerId: null,
      version: 3,
      width: 10,
    };
    layers.publishRasterized('layer', { height: 10, width: 10, x: 0, y: 0 }, backend.createSurface(10, 10));
    const controller = new RasterExportController({
      backend,
      captureGuard: (captured, entry) => ({
        cacheVersion: entry.version,
        documentGeneration: 1,
        layer: captured,
        layerId: captured.id,
        projectId: 'p',
      }),
      getDocument: () => document,
      getOrStartRasterization: () => Promise.resolve('published'),
      isGuardCurrent: () => true,
      isRasterizing: () => false,
      isSupportedSource: () => true,
      layers,
      pin: (layerId) => memory.pin(layerId),
      reserve: (bytes) => memory.reserveOperation(bytes, { purpose: 'raster-export' }),
    });

    const adjusted = await controller.rasterize('layer', { applyAdjustments: true });

    expect(adjusted.status).toBe('ok');
    expect(memory.isPinned('layer')).toBe(false);
    expect(memory.snapshot().reservedBytes).toBe(800);
    if (adjusted.status === 'ok') {
      expect(adjusted.surface).not.toBe(layers.peek('layer')?.surface);
      adjusted.release();
    }
    expect(memory.snapshot().reservedBytes).toBe(0);
  });
});
