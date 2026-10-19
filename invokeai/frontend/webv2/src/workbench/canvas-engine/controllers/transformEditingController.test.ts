import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { TransformSession } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { TransformEditingController } from './transformEditingController';

const MOVED = { rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 4 };

/** The controller over a real reducer, mutation context and history. */
const createHarness = (
  layer: CanvasLayerContract,
  options: { byteBudget?: number; commitStatus?: 'committed' | 'busy' } = {}
) => {
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: documentFrom([layer], layer.id),
    type: 'replaceCanvasDocument',
  });
  const history = createHistory({ byteBudget: options.byteBudget });
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  layers.getOrCreateRect(layer.id, { height: 4, width: 4, x: 0, y: 0 });
  const dispatch = (action: CanvasProjectMutation): boolean => {
    project = applyCanvasProjectMutation(project, action);
    return true;
  };
  const rasterFits = { value: true };
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
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    preparePixels: () => {
      throw new Error('unused');
    },
    projectId: project.id,
    refreshMirror: () => undefined,
    reserveRaster: () =>
      rasterFits.value
        ? { lease: { release: () => undefined }, status: 'ok' }
        : { availableBytes: 0, requestedBytes: 1, status: 'over-budget' },
    subscribeReducer: () => () => undefined,
  });
  let session: TransformSession | null = null;
  const calls = {
    commitStructural: vi.fn(() => ({ status: options.commitStatus ?? ('committed' as const) })),
    reportRefusal: vi.fn(),
    restoreCache: vi.fn(),
    setOverride: vi.fn(),
  };
  const controller = new TransformEditingController({
    backend,
    canEdit: () => true,
    commitStructural: calls.commitStructural,
    ctx,
    getCache: (layerId) => layers.get(layerId) ?? null,
    getDocument: () => project.canvas.document,
    invalidate: vi.fn(),
    isGestureActive: () => false,
    reportRefusal: calls.reportRefusal,
    restoreCache: calls.restoreCache,
    session: { get: () => session, set: (value) => (session = value) },
    setOverride: calls.setOverride,
  });
  return {
    calls,
    controller,
    dispatch,
    history,
    layer: () => getDocumentLayer(project.canvas.document, layer.id),
    rasterFits,
    session: () => session,
  };
};

const shape = layerContract('shape', 'raster', {
  source: { fill: '#fff', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 },
} as Partial<CanvasLayerContract>);
const paint = layerContract('paint', 'raster', {
  source: { bitmap: { height: 4, imageName: 'paint.png', width: 4 }, type: 'paint' },
} as Partial<CanvasLayerContract>);

describe('TransformEditingController', () => {
  it('commits a parametric transform as a structural edit and ends the session', () => {
    const h = createHarness(shape);
    h.controller.begin('shape');
    h.controller.update(MOVED);
    h.controller.apply();

    expect(h.calls.commitStructural).toHaveBeenCalledWith(
      'Transform layer',
      expect.objectContaining({ id: 'shape', patch: { transform: MOVED } }),
      expect.objectContaining({ id: 'shape', patch: { transform: expect.objectContaining({ x: 0 }) } })
    );
    expect(h.session()).toBeNull();
    expect(h.calls.setOverride).toHaveBeenLastCalledWith('shape', null);
  });

  it('keeps the session open and reports a refused commit', () => {
    const h = createHarness(shape, { commitStatus: 'busy' });
    h.controller.begin('shape');
    h.controller.update(MOVED);
    h.controller.apply();

    expect(h.calls.reportRefusal).toHaveBeenCalledWith('busy');
    expect(h.session()).toMatchObject({ layerId: 'shape', transform: MOVED });
  });

  it('bakes a paint transform into one step that undo and redo replay as a unit', async () => {
    const h = createHarness(paint);
    h.controller.begin('paint');
    h.controller.update(MOVED);
    h.controller.apply();

    expect(h.session()).toBeNull();
    expect(h.layer()?.transform).toMatchObject({ x: 0, y: 0 });
    expect(h.calls.restoreCache).toHaveBeenLastCalledWith(
      'paint',
      { height: 4, width: 4, x: 12, y: 4 },
      expect.anything()
    );
    expect(h.history.entries().past).toEqual(['Transform layer']);

    expect(await h.history.undo()).toEqual({ status: 'applied' });
    expect(h.calls.restoreCache).toHaveBeenLastCalledWith(
      'paint',
      { height: 4, width: 4, x: 0, y: 0 },
      expect.anything()
    );
    expect(await h.history.redo()).toEqual({ status: 'applied' });
    expect(h.calls.restoreCache).toHaveBeenCalledTimes(3);
  });

  it('refuses a bake too large to undo before changing the layer', () => {
    const h = createHarness(paint, { byteBudget: 10 });
    h.controller.begin('paint');
    h.controller.update(MOVED);
    h.controller.apply();

    expect(h.calls.reportRefusal).toHaveBeenCalledWith('over-budget');
    expect(h.calls.restoreCache).not.toHaveBeenCalled();
    expect(h.session()).toMatchObject({ transform: MOVED });
    expect(h.history.canUndo()).toBe(false);
  });

  it('keeps the step in place when the pixels it restores do not fit', async () => {
    const h = createHarness(paint);
    h.controller.begin('paint');
    h.controller.update(MOVED);
    h.controller.apply();
    h.rasterFits.value = false;

    expect(await h.history.undo()).toMatchObject({ status: 'failed' });
    expect(h.layer()?.transform).toMatchObject({ x: 0, y: 0 });
    expect(h.calls.restoreCache).toHaveBeenCalledOnce();
    h.rasterFits.value = true;
    expect(await h.history.undo()).toEqual({ status: 'applied' });
  });

  it('leaves the step in place when its layer can no longer take the transform', async () => {
    const h = createHarness(paint);
    h.controller.begin('paint');
    h.controller.update(MOVED);
    h.controller.apply();
    h.dispatch({ ids: ['paint'], type: 'removeCanvasLayers' });

    expect(await h.history.undo()).toMatchObject({ status: 'failed' });
    expect(h.history.entries().past).toEqual(['Transform layer']);
    expect(h.calls.restoreCache).toHaveBeenCalledOnce();
  });
});
