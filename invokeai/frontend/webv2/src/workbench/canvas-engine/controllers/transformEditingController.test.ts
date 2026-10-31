import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { TransformSession } from '@workbench/canvas-engine/engineStores';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { ToolId } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { restoreControlOriginal, withDetectedMap } from '@workbench/canvas-engine/document/controlDerivation';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { composeLayerTransforms } from '@workbench/canvas-engine/transform/transformMath';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { TransformEditingController } from './transformEditingController';

const MOVED = { rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 4 };

/**
 * The controller over a real reducer, mutation context and history. The first layer is selected and has a 4×4 cache;
 * `state` holds what the engine would otherwise decide (the tool, locks, a float, a gesture).
 */
const createHarness = (
  layers: CanvasLayerContract | CanvasLayerContract[],
  options: { byteBudget?: number; commitStatus?: 'committed' | 'busy' } = {}
) => {
  const [layer] = Array.isArray(layers) ? layers : [layers];
  if (!layer) {
    throw new Error('expected a layer');
  }
  let project = applyCanvasProjectMutation(createInitialWorkbenchState().projects[0]!, {
    document: documentFrom(Array.isArray(layers) ? layers : [layers], layer.id),
    type: 'replaceCanvasDocument',
  });
  const history = createHistory({ byteBudget: options.byteBudget });
  const backend = createTestStubRasterBackend();
  const caches = createLayerCacheStore(backend);
  caches.getOrCreateRect(layer.id, { height: 4, width: 4, x: 0, y: 0 });
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
  let pending: string | null = null;
  const state = {
    canEdit: true,
    floating: false,
    gesture: false,
    tool: 'brush' as ToolId,
  };
  const calls = {
    commitStructural: vi.fn(() => ({ status: options.commitStatus ?? ('committed' as const) })),
    enterTool: vi.fn(() => {
      state.tool = 'transform';
    }),
    rebaseLineEnd: vi.fn(),
    reportRefusal: vi.fn(),
    restoreCache: vi.fn(),
    select: vi.fn((layerId: string) => dispatch({ id: layerId, type: 'setCanvasSelectedLayer' })),
    setOverride: vi.fn(),
  };
  const controller = new TransformEditingController({
    backend,
    canEdit: () => state.canEdit,
    commitStructural: calls.commitStructural,
    ctx,
    enterTool: calls.enterTool,
    getCache: (layerId) => caches.get(layerId) ?? null,
    getDocument: () => project.canvas.document,
    getUserToolId: () => state.tool,
    hasFloatingSelection: () => state.floating,
    invalidate: vi.fn(),
    isGestureActive: () => state.gesture,
    peekCacheRect: (layerId) => caches.peek(layerId)?.rect,
    pending: { get: () => pending, set: (value) => (pending = value) },
    reportRefusal: calls.reportRefusal,
    rebaseLineEnd: calls.rebaseLineEnd,
    restoreCache: calls.restoreCache,
    select: calls.select,
    session: { get: () => session, set: (value) => (session = value) },
    setOverride: calls.setOverride,
  });
  return {
    calls,
    controller,
    dispatch,
    document: () => project.canvas.document,
    history,
    layer: () => getDocumentLayer(project.canvas.document, layer.id),
    pending: () => pending,
    rasterFits,
    session: () => session,
    state,
  };
};

type HarnessState = ReturnType<typeof createHarness>['state'];

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

  it("moves a detected control layer's original with the baked map, in the same undo step", async () => {
    const depth = { settings: {}, type: 'depth_anything_depth_estimation' };
    const originalTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 2, y: 3 };
    const detected: CanvasControlLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      controlType: 'depth',
      detection: depth,
      filter: depth,
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Depth 1',
      opacity: 1,
      original: { image: { height: 4, imageName: 'photo', width: 4 }, type: 'image' },
      originalTransform,
      source: { bitmap: { height: 4, imageName: 'map.png', width: 4 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const applied = { rotation: 0, scaleX: 2, scaleY: 2, x: 12, y: 4 };
    const h = createHarness(detected);
    h.controller.begin('control');
    h.controller.update(applied);
    h.controller.apply();

    const baked = h.layer() as CanvasControlLayerContract;
    expect(baked.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    expect(baked).toMatchObject({ controlType: 'depth', detection: depth, original: detected.original });
    // Re-detection and "None" both land the original where the applied transform put it.
    const placed = composeLayerTransforms(applied, originalTransform);
    expect(restoreControlOriginal(baked).transform).toEqual(placed);
    const map = { image: { height: 4, imageName: 'next-map', width: 4 }, rect: { height: 4, width: 4, x: 0, y: 0 } };
    expect(withDetectedMap(baked, map, depth)?.transform).toEqual(placed);
    expect(h.history.entries().past).toEqual(['Transform layer']);

    await h.history.undo();
    expect(h.layer()).toEqual(detected);
  });

  it('holds the original a shearing bake detaches, so undo can restore it', async () => {
    const depth = { settings: {}, type: 'depth_anything_depth_estimation' };
    const detected: CanvasControlLayerContract = {
      adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
      blendMode: 'normal',
      controlType: 'depth',
      detection: depth,
      filter: depth,
      id: 'control',
      isEnabled: true,
      isLocked: false,
      name: 'Depth 1',
      opacity: 1,
      original: { image: { height: 4, imageName: 'photo', width: 4 }, type: 'image' },
      originalTransform: { rotation: Math.PI / 4, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      source: { bitmap: { height: 4, imageName: 'map.png', width: 4 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'control',
      withTransparencyEffect: true,
    };
    const h = createHarness(detected);
    h.controller.begin('control');
    h.controller.update({ rotation: 0, scaleX: 2, scaleY: 1, x: 0, y: 0 });
    h.controller.apply();

    expect(h.layer()).not.toHaveProperty('original');
    expect(h.layer()).toMatchObject({ detection: null });
    expect(h.history.heldAssetRefs().images).toContain('photo');

    await h.history.undo();
    expect(h.layer()).toEqual(detected);
  });
});

describe('TransformEditingController.request', () => {
  const persistedPaint = (id: string) =>
    layerContract(id, 'raster', {
      source: { bitmap: { height: 4, imageName: `${id}.png`, width: 4 }, type: 'paint' },
    } as Partial<CanvasLayerContract>);
  const a = persistedPaint('a');
  const b = persistedPaint('b');

  it('selects the layer, enters the tool and opens its session', () => {
    const h = createHarness([a, b]);

    expect(h.controller.request('b')).toBe('started');
    expect(h.document()?.selectedLayerId).toBe('b');
    expect(h.calls.enterTool).toHaveBeenCalledOnce();
    expect(h.session()?.layerId).toBe('b');
    expect(h.pending()).toBeNull();
  });

  it('cancels a session on another layer, dropping its preview, before opening one on the target', () => {
    const h = createHarness([a, b]);
    h.controller.begin('a');
    h.controller.update(MOVED);

    expect(h.controller.request('b')).toBe('started');
    expect(h.calls.setOverride).toHaveBeenCalledWith('a', null);
    expect(h.session()).toMatchObject({ layerId: 'b', transform: { scaleX: 1, x: 0, y: 0 } });
    // The abandoned preview on `a` never landed.
    expect(getDocumentLayer(h.document(), 'a')?.transform.x).toBe(0);
  });

  it.each([
    ['a locked leaf', { ...b, isLocked: true }, 'locked'],
    ['a mask', layerContract('b', 'inpaint_mask'), 'ineligible'],
    ['a missing leaf', null, 'missing'],
  ] as const)('refuses %s without selecting or switching tools', (_, target, status) => {
    const h = createHarness(target ? [a, target] : [a]);

    expect(h.controller.request('b')).toBe(status);
    expect(h.calls.select).not.toHaveBeenCalled();
    expect(h.calls.enterTool).not.toHaveBeenCalled();
    expect(h.session()).toBeNull();
    expect(h.pending()).toBeNull();
  });

  it.each([
    ['a gesture runs', (s: HarnessState) => (s.gesture = true)],
    ['a float is live', (s: HarnessState) => (s.floating = true)],
    ['document edits are locked', (s: HarnessState) => (s.canEdit = false)],
  ])('refuses as busy while %s, leaving the selection and tool unchanged', (_, lock) => {
    const h = createHarness([a, b]);
    lock(h.state);

    expect(h.controller.request('b')).toBe('busy');
    expect(h.calls.select).not.toHaveBeenCalled();
    expect(h.calls.enterTool).not.toHaveBeenCalled();
    expect(h.session()).toBeNull();
  });

  it('refuses as busy once disposed', () => {
    const h = createHarness([a, b]);
    h.controller.dispose();
    h.controller.dispose();

    expect(h.controller.request('b')).toBe('busy');
    expect(h.calls.select).not.toHaveBeenCalled();
  });
});

describe('TransformEditingController pending transform', () => {
  /** A paint layer whose only pixels are a stroke persistence has not acknowledged yet. */
  const unpersisted = layerContract('stroke', 'raster');
  const other = layerContract('other', 'raster', {
    source: { bitmap: { height: 4, imageName: 'other.png', width: 4 }, type: 'paint' },
  } as Partial<CanvasLayerContract>);

  const requestUnpersisted = () => {
    const h = createHarness([unpersisted, other]);
    expect(h.controller.request('stroke')).toBe('started');
    expect(h.session()).toBeNull();
    expect(h.pending()).toBe('stroke');
    const persist = () => {
      h.dispatch({
        id: 'stroke',
        source: { bitmap: { height: 4, imageName: 'stroke.png', width: 4 }, type: 'paint' },
        type: 'updateCanvasLayerSource',
      });
      h.controller.onDocumentChanged();
    };
    return { ...h, persist };
  };

  it('opens the session once the pixels are persisted, and not before', () => {
    const h = requestUnpersisted();
    h.controller.onDocumentChanged();
    expect(h.pending()).toBe('stroke');

    h.persist();
    expect(h.session()?.layerId).toBe('stroke');
    expect(h.pending()).toBeNull();
  });

  it('keeps waiting through a switch to the transform tool itself', () => {
    const h = requestUnpersisted();
    h.controller.onToolSwitch('transform');
    h.persist();
    expect(h.session()?.layerId).toBe('stroke');
  });

  it.each([
    ['a tool switch', (h: ReturnType<typeof requestUnpersisted>) => h.controller.onToolSwitch('brush')],
    ['Escape', (h: ReturnType<typeof requestUnpersisted>) => expect(h.controller.onEscape()).toBe(true)],
    ['a history replay', (h: ReturnType<typeof requestUnpersisted>) => h.controller.onReplay()],
    ['a document replacement', (h: ReturnType<typeof requestUnpersisted>) => h.controller.onDocumentReplaced()],
    [
      'a selection change',
      (h: ReturnType<typeof requestUnpersisted>) => {
        h.dispatch({ id: 'other', type: 'setCanvasSelectedLayer' });
        h.controller.onDocumentChanged();
        h.dispatch({ id: 'stroke', type: 'setCanvasSelectedLayer' });
      },
    ],
    [
      'leaving transform for a tool the user is on',
      (h: ReturnType<typeof requestUnpersisted>) => {
        h.state.tool = 'brush';
        h.controller.onDocumentChanged();
        h.state.tool = 'transform';
      },
    ],
    [
      'another session opening',
      (h: ReturnType<typeof requestUnpersisted>) => {
        h.controller.begin('other');
        h.controller.onDocumentChanged();
        h.controller.cancel();
      },
    ],
    ['disposal', (h: ReturnType<typeof requestUnpersisted>) => h.controller.dispose()],
  ])('stops waiting on %s, silently', (_, end) => {
    const h = requestUnpersisted();
    end(h);
    expect(h.pending()).toBeNull();

    h.persist();
    expect(h.session()).toBeNull();
    expect(h.calls.reportRefusal).not.toHaveBeenCalled();
  });

  it('stops waiting when the layer is removed', () => {
    const h = requestUnpersisted();
    h.dispatch({ ids: ['stroke'], type: 'removeCanvasLayers' });
    h.controller.onDocumentChanged();
    expect(h.pending()).toBeNull();
  });

  it('leaves Escape to the next handler when nothing waits', () => {
    const h = createHarness([other]);
    expect(h.controller.onEscape()).toBe(false);
  });

  it('reports not-ready when the waiting layer’s upload fails with no retry to follow', () => {
    const h = requestUnpersisted();
    h.controller.onUploadFailed('other');
    expect(h.pending()).toBe('stroke');
    expect(h.calls.reportRefusal).not.toHaveBeenCalled();

    h.controller.onUploadFailed('stroke');
    expect(h.pending()).toBeNull();
    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('not-ready');
  });
});
