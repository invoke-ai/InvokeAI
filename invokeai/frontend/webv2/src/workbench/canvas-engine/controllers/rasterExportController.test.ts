import type { CanvasDocumentContractV2 } from '@workbench/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { RasterExportController } from './rasterExportController';

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
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [layer],
      selectedLayerId: null,
      version: 2,
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
      adjustments: { brightness: 0.1, contrast: 0, saturation: 0 },
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
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [layer],
      selectedLayerId: null,
      version: 2,
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
      adjustments: { brightness: 0.1, contrast: 0, saturation: 0 },
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
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      layers: [layer],
      selectedLayerId: null,
      version: 2,
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
});
