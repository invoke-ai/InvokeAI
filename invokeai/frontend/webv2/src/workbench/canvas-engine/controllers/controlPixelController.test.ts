import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';

import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { PixelEditController } from './controlPixelController';

const imageData = (data: readonly number[], width = 1, height = 1): ImageData =>
  ({ colorSpace: 'srgb', data: new Uint8ClampedArray(data), height, width }) as ImageData;

describe('PixelEditController', () => {
  it('can be instantiated with narrow fakes and rejects edits without a document', () => {
    const controller = new PixelEditController({
      applyImagePatch: vi.fn(),
      backend: createTestStubRasterBackend(),
      bitmapStore: { discardLayer: vi.fn(), markLayerDirty: vi.fn(), suspendLayer: vi.fn() } as never,
      canEdit: () => true,
      deleteDerived: vi.fn(),
      dispatchReplacement: vi.fn(),
      endBurst: vi.fn(),
      getActiveProjectId: () => 'project-1',
      getAdjustedSurface: vi.fn(),
      getDocument: () => null,
      getTransformSession: () => null,
      history: {} as never,
      installPrepared: vi.fn(),
      invalidate: vi.fn(),
      isCacheReady: () => false,
      isOperationIdle: () => true,
      layers: {} as never,
      notifyPainted: vi.fn(),
      preparePixels: vi.fn(),
      projectId: 'project-1',
      publishStroke: vi.fn(),
      setTransformOverride: vi.fn(),
    });

    expect(controller.begin('control-1')).toBeNull();
    expect(controller.isOpenFor(['control-1'])).toBe(false);
    expect(() => controller.dispose()).not.toThrow();
  });

  it('dirties pixels and releases persistence when direct-edit cleanup throws', () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const release = vi.fn();
    const markLayerDirty = vi.fn();
    const layer: CanvasControlLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: 'm', weight: 1 },
      blendMode: 'normal',
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Control',
      opacity: 1,
      source: { bitmap: null, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: false,
    };
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 8, width: 8, x: 0, y: 0 },
      height: 8,
      layers: [layer],
      selectedLayerId: layer.id,
      version: 2,
      width: 8,
    };
    const history = createHistory();
    const controller = new PixelEditController({
      applyImagePatch: vi.fn(),
      backend,
      bitmapStore: { discardLayer: vi.fn(), markLayerDirty, suspendLayer: vi.fn(() => release) },
      canEdit: () => true,
      deleteDerived: vi.fn(),
      dispatchReplacement: vi.fn(),
      endBurst: () => {
        throw new Error('end burst failed');
      },
      getActiveProjectId: () => 'project-1',
      getAdjustedSurface: vi.fn(),
      getDocument: () => document,
      getTransformSession: () => null,
      history,
      installPrepared: vi.fn(),
      invalidate: vi.fn(),
      isCacheReady: () => true,
      isOperationIdle: () => true,
      layers,
      notifyPainted: vi.fn(),
      preparePixels: (layerId, rect, pixels) => layers.prepareReplacement(layerId, rect, pixels),
      projectId: 'project-1',
      publishStroke: vi.fn(),
      setTransformOverride: vi.fn(),
    });
    const transaction = controller.begin(layer.id)!;
    layers.getOrCreateRect(layer.id, { height: 1, width: 1, x: 0, y: 0 });

    expect(() =>
      transaction.commitPatch('Direct edit', {
        after: imageData([255, 0, 0, 255]),
        before: imageData([0, 0, 0, 0]),
        rect: { height: 1, width: 1, x: 0, y: 0 },
      })
    ).toThrow('end burst failed');
    expect(history.canUndo()).toBe(true);
    expect(markLayerDirty).toHaveBeenCalledWith(layer.id);
    expect(release).toHaveBeenCalledOnce();
    expect(controller.isOpenFor([layer.id])).toBe(false);
  });

  it('keeps an accepted materialization dirty and releases persistence when transform cleanup throws', () => {
    const backend = createTestStubRasterBackend();
    const layers = createLayerCacheStore(backend);
    const release = vi.fn();
    const markLayerDirty = vi.fn();
    const layer: CanvasRasterLayerContractV2 = {
      blendMode: 'normal',
      id: 'image',
      isEnabled: true,
      isLocked: false,
      name: 'Image',
      opacity: 1,
      source: { image: { height: 1, imageName: 'image', width: 1 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const document: CanvasDocumentContractV2 = {
      background: 'transparent',
      bbox: { height: 8, width: 8, x: 0, y: 0 },
      height: 8,
      layers: [layer],
      selectedLayerId: layer.id,
      version: 2,
      width: 8,
    };
    const entry = layers.getOrCreate(layer.id, 1, 1);
    entry.surface.ctx.putImageData(imageData([255, 255, 255, 255]), 0, 0);
    layers.publishPixels(layer.id);
    const history = createHistory();
    const controller = new PixelEditController({
      applyImagePatch: vi.fn(),
      backend,
      bitmapStore: { discardLayer: vi.fn(), markLayerDirty, suspendLayer: vi.fn(() => release) },
      canEdit: () => true,
      deleteDerived: vi.fn(),
      dispatchReplacement: (replacement) => {
        document.layers = [replacement];
      },
      endBurst: vi.fn(),
      getActiveProjectId: () => 'project-1',
      getAdjustedSurface: () => null,
      getDocument: () => document,
      getTransformSession: () => null,
      history,
      installPrepared: vi.fn(),
      invalidate: vi.fn(),
      isCacheReady: () => true,
      isOperationIdle: () => true,
      layers,
      notifyPainted: vi.fn(),
      preparePixels: (layerId, rect, pixels) => layers.prepareReplacement(layerId, rect, pixels),
      projectId: 'project-1',
      publishStroke: vi.fn(),
      setTransformOverride: (_layerId, transform) => {
        if (transform === null) {
          throw new Error('transform cleanup failed');
        }
      },
    });
    const transaction = controller.begin(layer.id)!;

    expect(() =>
      transaction.commitPatch('Materialized edit', {
        after: imageData([0, 0, 0, 0]),
        before: imageData([255, 255, 255, 255]),
        rect: { height: 1, width: 1, x: 0, y: 0 },
      })
    ).toThrow('transform cleanup failed');
    expect(document.layers[0]).toMatchObject({ source: { type: 'paint' }, type: 'raster' });
    expect(history.canUndo()).toBe(true);
    expect(markLayerDirty).toHaveBeenCalledWith(layer.id);
    expect(release).toHaveBeenCalledOnce();
    expect(controller.isOpenFor([layer.id])).toBe(false);
  });
});
