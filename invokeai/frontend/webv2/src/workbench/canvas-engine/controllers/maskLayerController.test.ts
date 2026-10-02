import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { Project } from '@workbench/projectContracts';

import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { describe, expect, it, vi } from 'vitest';

import { MaskLayerController } from './maskLayerController';
import { createCanvasMutationContext } from './mutationContext';

const PERSISTED = { height: 20, imageName: 'persisted-mask', width: 20 };
const CACHE_RECT = { height: 20, width: 20, x: 0, y: 0 };

/** A real reducer, history, transaction protocol and layer cache around one persisted inpaint mask. */
const createHarness = (options: { byteBudget?: number } = {}) => {
  const base = createInitialWorkbenchState().projects[0]!;
  const mask = layerContract('mask', 'inpaint_mask');
  if (mask.type !== 'inpaint_mask') {
    throw new Error('expected an inpaint mask fixture');
  }
  mask.mask = { ...mask.mask, bitmap: PERSISTED, offset: { x: 0, y: 0 } };
  let project: Project = applyCanvasProjectMutation(base, {
    document: documentFrom([mask]),
    type: 'replaceCanvasDocument',
  });
  let mirrorBroken = false;
  let mirrorDocument = project.canvas.document;
  let reserveFits = true;
  // Fault injection: edits lock while the operation prepares, so its publication is refused.
  let lockWhilePreparing = false;
  let locked = false;
  const dispatched: CanvasProjectMutation[] = [];
  const history = createHistory({ byteBudget: options.byteBudget });
  const layers = createLayerCacheStore(createTestStubRasterBackend());
  layers.getOrCreateRect('mask', CACHE_RECT).stale = false;
  layers.publishPixels('mask');
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: (mutation) => {
      dispatched.push(mutation);
      const next = applyCanvasProjectMutation(project, mutation);
      const changed = next !== project;
      project = next;
      if (!mirrorBroken) {
        mirrorDocument = project.canvas.document;
      }
      return changed;
    },
    editOwner: Symbol('owner'),
    editingLocked: { get: () => locked, subscribe: () => () => undefined },
    getDocument: () => mirrorDocument,
    getReducerDocument: () => project.canvas.document,
    history,
    installPrepared: () => undefined,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    preparePixels: () => ({}) as never,
    projectId: base.id,
    refreshMirror: () => {
      if (mirrorBroken) {
        throw new Error('mirror broken');
      }
      mirrorDocument = project.canvas.document;
    },
    reserveRaster: () => {
      locked ||= lockWhilePreparing;
      return reserveFits
        ? { lease: { release: () => undefined }, status: 'ok' }
        : { availableBytes: 0, requestedBytes: 1, status: 'over-budget' };
    },
    subscribeReducer: () => () => undefined,
  });
  const restoreCache = vi.fn();
  const controller = new MaskLayerController({
    applyImagePatch: () => Promise.resolve(),
    ctx,
    deleteDerived: vi.fn(),
    discardPersisted: vi.fn(),
    isCacheReady: () => true,
    layers,
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    restoreCache,
  });
  const maskBitmap = () => {
    const layer = getDocumentLayer(project.canvas.document, 'mask');
    return layer?.type === 'inpaint_mask' ? layer.mask.bitmap : undefined;
  };
  return {
    breakMirror: (broken: boolean) => {
      mirrorBroken = broken;
    },
    controller,
    dispatched,
    history,
    layers,
    maskBitmap,
    restoreCache,
    lockWhilePreparing: () => {
      lockWhilePreparing = true;
    },
    setReserveFits: (fits: boolean) => {
      reserveFits = fits;
    },
  };
};

describe('MaskLayerController', () => {
  it('refuses an invert whose grown extent does not fit before touching the cache or history', () => {
    const h = createHarness();
    h.setReserveFits(false);
    const before = h.layers.get('mask')!;

    expect(h.controller.invert('mask')).toEqual({ status: 'over-budget' });
    expect(h.layers.get('mask')!.rect).toEqual(CACHE_RECT);
    expect(h.layers.get('mask')!.version).toBe(before.version);
    expect(h.history.canUndo()).toBe(false);
  });

  it('returns the cache to its extent when an invert cannot be recorded', () => {
    const h = createHarness();
    h.lockWhilePreparing();

    expect(h.controller.invert('mask').status).not.toBe('committed');
    expect(h.layers.get('mask')!.rect).toEqual(CACHE_RECT);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses a clear history could never retain before dispatching', () => {
    const h = createHarness({ byteBudget: 64 });

    expect(h.controller.clear('mask')).toEqual({ status: 'over-budget' });
    expect(h.dispatched).toEqual([]);
    expect(h.maskBitmap()).toEqual(PERSISTED);
    expect(h.layers.get('mask')!.rect).toEqual(CACHE_RECT);
  });

  it('rolls the mask back and keeps its pixels when the mirror cannot follow a clear', () => {
    const h = createHarness();
    h.breakMirror(true);

    expect(h.controller.clear('mask')).toEqual({ status: 'stale' });
    expect(h.maskBitmap()).toEqual(PERSISTED);
    expect(h.layers.get('mask')!.rect).toEqual(CACHE_RECT);
    expect(h.history.canUndo()).toBe(false);
  });

  it('restores the cleared pixels on undo, and keeps the step when their copy does not fit', async () => {
    const h = createHarness();
    expect(h.controller.clear('mask')).toEqual({ status: 'committed' });
    expect(h.maskBitmap()).toBeNull();
    expect(h.layers.get('mask')!.rect.width).toBe(0);

    h.setReserveFits(false);
    expect((await h.history.undo()).status).toBe('failed');
    expect(h.maskBitmap()).toBeNull();
    expect(h.history.canUndo()).toBe(true);

    h.setReserveFits(true);
    expect(await h.history.undo()).toEqual({ status: 'applied' });
    expect(h.maskBitmap()).toEqual(PERSISTED);
    expect(h.restoreCache).toHaveBeenCalledWith('mask', CACHE_RECT, expect.anything());
  });
});
