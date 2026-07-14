import type { WorkbenchAction } from '@workbench/workbenchState';

import { describe, expect, it, vi } from 'vitest';

import { LayerController } from './layerController';
import { StructuralLayerController } from './structuralLayerController';

describe('LayerController', () => {
  const mask = {
    applyImagePatch: vi.fn(),
    canEdit: () => true,
    deleteDerived: vi.fn(),
    discardPersisted: vi.fn(),
    dispatch: vi.fn(),
    endBurst: vi.fn(),
    getDocument: () => null,
    history: {} as never,
    isCacheReady: () => false,
    isGestureActive: () => false,
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
    canEdit: () => true,
    dispatch: vi.fn(),
    getDocument: () => null,
    history: { push: vi.fn() } as never,
    isGestureActive: () => false,
  });
  const rasterize = {
    backend: {} as never,
    canEdit: () => true,
    dispatch: vi.fn(),
    endBurst: vi.fn(),
    getDocument: () => null,
    history: {} as never,
    isGestureActive: () => false,
    layers: {} as never,
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    rasterizeDeps: vi.fn(),
  };
  const merge = {
    backend: {} as never,
    canEdit: () => true,
    dispatch: vi.fn(),
    endBurst: vi.fn(),
    getDocument: () => null,
    isCacheReady: () => true,
    isGestureActive: () => false,
    layers: {} as never,
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
  };
  const booleanMerge = {
    backend: {} as never,
    capturePermit: () => null,
    createLayerId: () => 'result',
    dispatchPrepared: vi.fn(),
    endBurst: vi.fn(),
    exportBaked: vi.fn(),
    getDocument: () => null,
    getReducerDocument: () => null,
    history: {} as never,
    installPrepared: vi.fn(),
    isCacheReady: () => true,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    isPermitCurrent: () => true,
    preparePixels: vi.fn(),
  };
  const extractMaskedArea = {
    backend: {} as never,
    capturePermit: () => null,
    createLayerId: () => 'result',
    derived: {} as never,
    diagnostics: {} as never,
    dispatchPrepared: vi.fn(),
    endBurst: vi.fn(),
    exportBaked: vi.fn(),
    getAdjustedSurface: vi.fn(),
    getDocument: () => null,
    getMaskPattern: () => null,
    getReducerDocument: () => null,
    hasExportableContent: () => false,
    history: {} as never,
    installPrepared: vi.fn(),
    isCacheReady: () => true,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    isPermitCurrent: () => true,
    layers: {} as never,
    preparePixels: vi.fn(),
    rasterize: vi.fn(),
  };
  const crop = {
    backend: {} as never,
    captureCache: vi.fn(),
    capturePermit: () => null,
    discardPersisted: vi.fn(),
    dispatchPrepared: vi.fn(),
    endBurst: vi.fn(),
    exportBaked: vi.fn(),
    getDocument: () => null,
    getReducerDocument: () => null,
    history: {} as never,
    installPrepared: vi.fn(),
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    isPermitCurrent: () => true,
    isSupportedSource: () => true,
    preparePixels: vi.fn(),
  };
  const copy = {
    capturePermit: () => null,
    createLayerId: () => 'copy',
    dispatchPrepared: vi.fn(),
    endBurst: vi.fn(),
    exportBaked: vi.fn(),
    getDocument: () => null,
    getReducerDocument: () => null,
    history: {} as never,
    installPrepared: vi.fn(),
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    isPermitCurrent: () => true,
    preparePixels: vi.fn(),
  };
  it('exposes only declared layer and preview ports', async () => {
    const forward: WorkbenchAction = { id: 'layer', type: 'setCanvasSelectedLayer' };
    const inverse: WorkbenchAction = { id: null, type: 'setCanvasSelectedLayer' };
    const deps = {
      commitGeneratedImageResult: vi.fn(() => Promise.resolve({ layerId: 'copy', status: 'committed' as const })),
      mask,
      booleanMerge,
      extractMaskedArea,
      crop,
      copy,
      merge,
      thumbnail,
      structural,
      rasterize,
    };
    const controller = new LayerController(deps);

    expect(controller.layers.applyStructuralPreview(forward)).toBe(true);
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

    expect(controller.layers.applyStructuralPreview({} as WorkbenchAction)).toBe(false);
    controller.layers.commitStructural('late', {} as WorkbenchAction, {} as WorkbenchAction);
    expect(controller.previews.drawLayerThumbnail('layer', {} as HTMLCanvasElement, 96)).toBe(false);
  });
});
