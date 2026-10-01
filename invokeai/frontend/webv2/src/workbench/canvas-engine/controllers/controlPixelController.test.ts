import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createHistory, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { PixelEditController } from './controlPixelController';

const imageData = (data: readonly number[], width = 1, height = 1): ImageData =>
  ({ colorSpace: 'srgb', data: new Uint8ClampedArray(data), height, width }) as ImageData;

const PATCH = {
  after: imageData([255, 0, 0, 255]),
  before: imageData([0, 0, 0, 0]),
  rect: { height: 1, width: 1, x: 0, y: 0 },
};

const control = layerContract('control', 'control');
const image = layerContract('image', 'raster', {
  source: { image: { height: 1, imageName: 'image', width: 1 }, type: 'image' },
} as Partial<CanvasLayerContract>);

/** The controller over a real reducer, mutation context and history. */
const createHarness = (
  layer: CanvasLayerContract,
  options: { byteBudget?: number; gestureActive?: boolean; notifyPainted?: () => void } = {}
) => {
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: documentFrom([layer], layer.id),
    type: 'replaceCanvasDocument',
  });
  const history = createHistory({ byteBudget: options.byteBudget });
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  layers.getOrCreateRect(layer.id, { height: 1, width: 1, x: 0, y: 0 });
  const dispatch = (action: CanvasProjectMutation): boolean => {
    project = applyCanvasProjectMutation(project, action);
    return true;
  };
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => project.canvas.document,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: () => undefined,
    isGestureActive: () => options.gestureActive ?? false,
    isGuardCurrent: () => true,
    preparePixels: () => {
      throw new Error('unused');
    },
    projectId: project.id,
    refreshMirror: () => undefined,
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: () => () => undefined,
  });
  const release = vi.fn();
  const calls = {
    installPrepared: vi.fn(),
    markLayerDirty: vi.fn(),
    release,
    reportRefusal: vi.fn(),
    suspendLayer: vi.fn(() => release),
  };
  const controller = new PixelEditController({
    applyImagePatch: vi.fn(() => Promise.resolve()),
    backend,
    bitmapStore: { discardLayer: vi.fn(), markLayerDirty: calls.markLayerDirty, suspendLayer: calls.suspendLayer },
    canEdit: () => true,
    ctx,
    deleteDerived: vi.fn(),
    getActiveProjectId: () => project.id,
    getAdjustedSurface: () => null,
    getDocument: () => project.canvas.document,
    getTransformSession: () => null,
    installPrepared: calls.installPrepared,
    invalidate: vi.fn(),
    isCacheReady: () => true,
    isOperationIdle: () => true,
    layers,
    notifyPainted: options.notifyPainted ?? vi.fn(),
    preparePixels: (layerId, rect, pixels) => layers.prepareReplacement(layerId, rect, pixels),
    projectId: project.id,
    publishStroke: vi.fn(),
    reportRefusal: calls.reportRefusal,
    setTransformOverride: vi.fn(),
  });
  return {
    calls,
    controller,
    dispatch,
    history,
    layer: () => getDocumentLayer(project.canvas.document, layer.id),
    layers,
  };
};

describe('PixelEditController', () => {
  it('rejects edits on a layer that is gone', () => {
    const h = createHarness(control);
    h.dispatch({ ids: ['control'], type: 'removeCanvasLayers' });

    expect(h.controller.begin('control')).toBeNull();
    expect(h.controller.isOpenFor(['control'])).toBe(false);
  });

  it('records a direct edit as one step, persisting it once the suspension ends', () => {
    const h = createHarness(control);
    const transaction = h.controller.begin('control')!;
    expect(h.calls.suspendLayer).toHaveBeenCalledWith('control');

    expect(transaction.commitPatch('Direct edit', PATCH)).toBe(true);
    expect(h.history.entries().past).toEqual(['Direct edit']);
    expect(h.calls.markLayerDirty).toHaveBeenCalledWith('control');
    expect(h.calls.release).toHaveBeenCalledOnce();
    expect(h.controller.isOpenFor(['control'])).toBe(false);
  });

  it('keeps an accepted edit recorded and persisted when a notification throws', () => {
    const h = createHarness(control, {
      notifyPainted: () => {
        throw new Error('observer failed');
      },
    });
    const transaction = h.controller.begin('control')!;

    expect(transaction.commitPatch('Direct edit', PATCH)).toBe(true);
    expect(h.history.canUndo()).toBe(true);
    expect(h.calls.markLayerDirty).toHaveBeenCalledWith('control');
    expect(h.calls.release).toHaveBeenCalledOnce();
  });

  it('refuses a gesture-time edit unless the gesture itself starts it', () => {
    const h = createHarness(control, { gestureActive: true });

    expect(h.controller.begin('control')).toBeNull();
    expect(h.calls.reportRefusal).toHaveBeenCalledWith('gesture-active');
    expect(h.controller.begin('control', { gesture: true })).not.toBeNull();
  });

  it('records a materialization as one replace step and replays it whole', async () => {
    const h = createHarness(image);
    const transaction = h.controller.begin('image')!;

    expect(transaction.commitPatch('Erase', PATCH)).toBe(true);
    expect(h.layer()).toMatchObject({ source: { type: 'paint' }, type: 'raster' });
    expect(h.calls.release).toHaveBeenCalledOnce();

    expect(await h.history.undo()).toEqual({ status: 'applied' });
    expect(h.layer()).toMatchObject({ source: { type: 'image' } });
    expect(await h.history.redo()).toEqual({ status: 'applied' });
    expect(h.layer()).toMatchObject({ source: { type: 'paint' } });
    expect(h.calls.installPrepared).toHaveBeenCalledTimes(2);
  });

  it('leaves a materialization step in place when its layer is gone', async () => {
    const h = createHarness(image);
    h.controller.begin('image')!.commitPatch('Erase', PATCH);
    h.dispatch({ ids: ['image'], type: 'removeCanvasLayers' });

    expect(await h.history.undo()).toMatchObject({ status: 'failed' });
    expect(h.history.entries().past).toEqual(['Erase']);
    expect(h.calls.installPrepared).not.toHaveBeenCalled();
  });

  it('refuses a materialization too large to undo before baking anything', () => {
    const h = createHarness(image, { byteBudget: 4 });
    const surface = h.layers.get('image')!.surface;

    expect(h.controller.begin('image')).toBeNull();
    expect(h.calls.reportRefusal).toHaveBeenCalledWith('over-budget');
    expect(h.layers.get('image')!.surface).toBe(surface);
    expect(h.calls.suspendLayer).not.toHaveBeenCalled();
  });

  it("admits a materialization's whole-layer snapshots once, not again for the stroke inside it", () => {
    const { rect } = createHarness(image).layers.get('image')!;
    const snapshots = rect.width * rect.height * 4 * 2 + HISTORY_ENTRY_OVERHEAD_BYTES;
    const h = createHarness(image, { byteBudget: snapshots });
    const transaction = h.controller.begin('image')!;

    expect(transaction.grow(rect.width * rect.height * 8)).toBe(true);
    expect(transaction.commitPatch('Erase', PATCH)).toBe(true);
    expect(h.calls.reportRefusal).not.toHaveBeenCalled();
  });

  it('reinstates the unbaked cache when a materialized edit is cancelled', () => {
    const h = createHarness(image);
    const surface = h.layers.get('image')!.surface;
    const transaction = h.controller.begin('image')!;
    expect(h.layers.get('image')!.surface).not.toBe(surface);

    transaction.cancel();
    expect(h.layers.get('image')!.surface).toBe(surface);
    expect(h.layer()).toMatchObject({ source: { type: 'image' } });
    expect(h.calls.release).toHaveBeenCalledOnce();
    expect(h.history.canUndo()).toBe(false);
  });
});
