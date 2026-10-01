import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { describe, expect, it, vi } from 'vitest';

import { LayerController } from './layerController';
import { StructuralLayerController } from './structuralLayerController';

describe('LayerController', () => {
  const mask = {
    applyImagePatch: vi.fn(),
    ctx: {} as never,
    deleteDerived: vi.fn(),
    discardPersisted: vi.fn(),
    isCacheReady: () => false,
    layers: {} as never,
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    restoreCache: vi.fn(),
  };
  const thumbnail = {
    backend: {} as never,
    getActiveProjectId: () => null,
    getCheckerboard: vi.fn(),
    getDocument: () => null,
    getEntry: () => undefined,
    getMaskPattern: () => null,
    isDisposed: () => false,
    isSupportedSource: () => true,
    projectId: 'p1',
    rasterize: vi.fn(),
    reportError: vi.fn(),
    setStatus: vi.fn(),
  };
  const structural = new StructuralLayerController({
    ctx: {
      applyStep: vi.fn(),
      begin: () => ({ status: 'not-ready' }),
      canEdit: () => true,
      dispatch: vi.fn(() => true),
      getDocument: () => null,
      getEditRevision: () => 0,
      getReducerDocument: () => null,
      historyTop: () => null,
      isGestureActive: () => false,
      projectId: 'p',
    },
  });
  // The layer operations are exercised by their own tests; here they only need to construct.
  const rasterize = { backend: {} as never, ctx: {} as never, rasterizeDeps: vi.fn() };
  const merge = {
    backend: {} as never,
    ctx: {} as never,
    exportBaked: vi.fn(),
    hasExportableContent: () => false,
    isCacheReady: () => true,
    layers: {} as never,
    needsPixelPersistence: () => false,
    publishSelectedLayerIds: vi.fn(),
  };
  const booleanMerge = {
    backend: {} as never,
    ctx: {} as never,
    exportBaked: vi.fn(),
    isCacheReady: () => true,
    isGuardCurrent: () => true,
  };
  const extractMaskedArea = {
    backend: {} as never,
    ctx: {} as never,
    derived: {} as never,
    diagnostics: {} as never,
    exportBaked: vi.fn(),
    getAdjustedSurface: vi.fn(),
    getMaskPattern: () => null,
    hasExportableContent: () => false,
    isCacheReady: () => true,
    isGuardCurrent: () => true,
    layers: {} as never,
    rasterize: vi.fn(),
  };
  const crop = {
    backend: {} as never,
    captureCache: vi.fn(),
    ctx: {} as never,
    discardPersisted: vi.fn(),
    exportBaked: vi.fn(),
    isGuardCurrent: () => true,
    isSupportedSource: () => true,
  };
  const copy = { backend: {} as never, ctx: {} as never, exportBaked: vi.fn(), isGuardCurrent: () => true };
  const newRasterLayer = { backend: {} as never, ctx: {} as never, layers: {} as never, selection: {} as never };
  it('exposes only declared layer and preview ports', async () => {
    const forward: CanvasProjectMutation = { id: 'layer', type: 'setCanvasSelectedLayer' };
    const inverse: CanvasProjectMutation = { id: null, type: 'setCanvasSelectedLayer' };
    const deps = {
      commitGeneratedImageResult: vi.fn(() => Promise.resolve({ layerId: 'copy', status: 'committed' as const })),
      mask,
      booleanMerge,
      extractMaskedArea,
      newRasterLayer,
      crop,
      copy,
      merge,
      thumbnail,
      structural,
      rasterize,
    };
    const controller = new LayerController(deps);

    expect(
      controller.layers
        .beginStructuralPreview()
        ?.apply({ id: 'layer', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' })
    ).toBe(true);
    controller.layers.commitStructural('edit', forward, inverse);
    expect(controller.previews.drawLayerThumbnail('layer', {} as HTMLCanvasElement, 96)).toBe(false);
    await expect(controller.previews.requestLayerThumbnail('layer')).resolves.toBe('stale');
  });

  it('disposes idempotently and rejects later mutations', () => {
    const deps = {
      commitGeneratedImageResult: vi.fn(() => Promise.resolve({ layerId: 'copy', status: 'committed' as const })),
      mask,
      booleanMerge,
      extractMaskedArea,
      newRasterLayer,
      crop,
      copy,
      merge,
      thumbnail,
      structural,
      rasterize,
    };
    const controller = new LayerController(deps);
    controller.dispose();
    controller.dispose();

    expect(controller.layers.beginStructuralPreview()).toBeNull();
    controller.layers.commitStructural('late', {} as CanvasProjectMutation, {} as CanvasProjectMutation);
    expect(controller.previews.drawLayerThumbnail('layer', {} as HTMLCanvasElement, 96)).toBe(false);
  });
});
