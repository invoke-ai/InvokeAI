import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createHistory, HISTORY_BYTE_BUDGET } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { FloatingSelectionController } from './floatingSelectionController';

const paintLayer = (id: string, transform: Partial<CanvasLayerContract['transform']> = {}): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0, ...transform },
  type: 'raster',
});

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId: layers[0]?.id ?? null,
  version: 3,
  width: 100,
});

const createHarness = (
  options: { layer?: CanvasLayerContract; maskRect?: Rect; cacheRect?: Rect | null; byteBudget?: number } = {}
) => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const history = createHistory({ byteBudget: options.byteBudget });
  const layer = options.layer ?? paintLayer('a');
  let document: CanvasDocumentContractV3 | null = makeDoc([layer]);

  const maskRect = options.maskRect ?? { height: 30, width: 30, x: 20, y: 20 };
  const maskSurface: PlacedSurface = {
    rect: maskRect,
    surface: backend.createSurface(maskRect.width, maskRect.height),
  };
  const replaceMask = vi.fn();
  const selection = {
    antsPaths: () => [],
    bounds: () => maskRect,
    clear: vi.fn(),
    commit: vi.fn(),
    containsPoint: () => true,
    dispose: vi.fn(),
    hasSelection: () => true,
    invert: vi.fn(),
    mask: () => maskSurface,
    replaceMask,
    restore: vi.fn(),
    selectAll: vi.fn(),
    snapshot: vi.fn(() => ({ alpha: null, bounds: null, commits: [], rect: null, selected: false })),
  } as SelectionState;

  if (options.cacheRect !== null) {
    layers.getOrCreateRect(layer.id, options.cacheRect ?? { height: 100, width: 100, x: 0, y: 0 });
  }

  const releasePersistence = vi.fn();
  const calls = {
    applyImagePatch: vi.fn((_layerId: string, _rect: Rect, _pixels: ImageData) => Promise.resolve()),
    invalidateLayer: vi.fn(),
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    onChange: vi.fn(),
    releasePersistence,
    reportRefusal: vi.fn(),
    suspendPersistence: vi.fn(() => releasePersistence),
  };
  let locked = false;
  const lockListeners = new Set<() => void>();
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: () => true,
    editOwner: Symbol('owner'),
    editingLocked: {
      get: () => locked,
      subscribe: (listener) => {
        lockListeners.add(listener);
        return () => lockListeners.delete(listener);
      },
    },
    getDocument: () => document,
    getReducerDocument: () => document,
    history,
    installPrepared: () => undefined,
    isGestureActive: () => false,
    isGuardCurrent: () => true,
    preparePixels: () => {
      throw new Error('unused');
    },
    projectId: 'p',
    refreshMirror: () => undefined,
    reserveRaster: () => ({ lease: { release: () => undefined }, status: 'ok' }),
    subscribeReducer: () => () => undefined,
  });

  const controller = new FloatingSelectionController({
    applyImagePatch: calls.applyImagePatch,
    backend,
    ctx,
    getDocument: () => document,
    invalidateLayer: calls.invalidateLayer,
    layers,
    markDirty: calls.markDirty,
    notifyPainted: calls.notifyPainted,
    onChange: calls.onChange,
    reportRefusal: calls.reportRefusal,
    selection,
    suspendPersistence: calls.suspendPersistence,
  });

  const budget = options.byteBudget ?? HISTORY_BYTE_BUDGET;
  return {
    backend,
    calls,
    controller,
    history,
    layer,
    layers,
    replaceMask,
    selection,
    heldBytes: (): number => {
      let low = 0;
      let high = budget;
      while (low < high) {
        const mid = Math.ceil((low + high) / 2);
        const admission = history.admit(mid);
        admission?.release();
        if (admission) {
          low = mid;
        } else {
          high = mid - 1;
        }
      }
      return budget - low;
    },
    disposeContext: () => ctx.dispose(),
    setLocked: (value: boolean) => {
      locked = value;
      lockListeners.forEach((listener) => listener());
    },
    removeLayer: () => {
      document = makeDoc([]);
    },
  };
};

const move = (controller: FloatingSelectionController, x: number, y: number): boolean =>
  controller.setTransform({ rotation: 0, scaleX: 1, scaleY: 1, x, y });

// Commit bytes for the default 30×30 lift at (20, 20) on an identity layer: before and after RGBA over the hole∪landing
// union, the 30×30 selection alpha before and after, and the entry overhead.
const commitBytes = (patchWidth: number, patchHeight: number, movedMaskPixels = 900): number =>
  patchWidth * patchHeight * 8 + 900 + movedMaskPixels + 256;
const LIFT_BYTES = commitBytes(30, 30);

describe('FloatingSelectionController: lift', () => {
  it('lifts the selection region and reports a live float', () => {
    const h = createHarness();
    expect(h.controller.lift('a')).toBe('lifted');
    expect(h.controller.has()).toBe(true);
    expect(h.controller.get()?.layerId).toBe('a');
    expect(h.controller.get()?.pixels.rect).toEqual({ height: 30, width: 30, x: 20, y: 20 });
    expect(h.controller.get()?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
  });

  it('suspends persistence of the layer until the float commits or cancels', () => {
    const committed = createHarness();
    committed.controller.lift('a');
    expect(committed.calls.suspendPersistence).toHaveBeenCalledWith('a');
    expect(committed.calls.releasePersistence).not.toHaveBeenCalled();
    move(committed.controller, 40, 0);
    committed.controller.commit();
    expect(committed.calls.releasePersistence).toHaveBeenCalledOnce();
    expect(committed.calls.markDirty.mock.invocationCallOrder[0]).toBeLessThan(
      committed.calls.releasePersistence.mock.invocationCallOrder[0]!
    );

    const cancelled = createHarness();
    cancelled.controller.lift('a');
    cancelled.controller.cancel();
    expect(cancelled.calls.releasePersistence).toHaveBeenCalledOnce();
  });

  it('refuses a lift whose commit could never be recorded before copying or cutting anything', () => {
    const h = createHarness({ byteBudget: LIFT_BYTES - 1 });
    const surface = h.layers.get('a')!.surface as unknown as { callLog: unknown[] };
    const drawnBefore = surface.callLog.length;

    expect(h.controller.lift('a')).toBe('refused');

    expect(h.controller.has()).toBe(false);
    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('over-budget');
    expect(surface.callLog).toHaveLength(drawnBefore);
    expect(h.calls.suspendPersistence).not.toHaveBeenCalled();
    expect(h.calls.notifyPainted).not.toHaveBeenCalled();
    expect(h.heldBytes()).toBe(0);
  });

  it('refuses and reports a lift while editing is locked, admitting and cutting nothing', () => {
    const h = createHarness();
    h.setLocked(true);

    expect(h.controller.lift('a')).toBe('refused');

    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('busy');
    expect(h.calls.suspendPersistence).not.toHaveBeenCalled();
    expect(h.heldBytes()).toBe(0);
  });

  it('refuses a lift from content whose cache is not rebuilt yet, so the layer is not moved instead', () => {
    const h = createHarness({
      cacheRect: null,
      layer: {
        ...(paintLayer('a') as Extract<CanvasLayerContract, { type: 'raster' }>),
        source: { bitmap: { height: 100, imageName: 'paint', width: 100 }, offset: { x: 0, y: 0 }, type: 'paint' },
      },
    });

    expect(h.controller.lift('a')).toBe('refused');
    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('not-ready');
    expect(h.heldBytes()).toBe(0);

    expect(createHarness({ cacheRect: null }).controller.lift('a')).toBe('unavailable');
  });

  it('restores the cut, lets the layer persist and returns its admission when the cut throws', () => {
    const h = createHarness();
    const createSurface = h.backend.createSurface.bind(h.backend);
    vi.spyOn(h.backend, 'createSurface').mockImplementation((width, height) => {
      if (h.calls.suspendPersistence.mock.calls.length > 0) {
        throw new Error('allocation failed');
      }
      return createSurface(width, height);
    });

    expect(() => h.controller.lift('a')).toThrow('allocation failed');

    expect(h.controller.has()).toBe(false);
    const ops = (h.layers.get('a')!.surface as unknown as { callLog: { op: string }[] }).callLog.map((call) => call.op);
    expect(ops.at(-1)).toBe('putImageData');
    expect(h.calls.releasePersistence).toHaveBeenCalledOnce();
    expect(h.heldBytes()).toBe(0);
  });
});

describe('FloatingSelectionController: movement admission', () => {
  it('grows the admission to cover a commit at each accepted transform, from geometry alone', () => {
    const h = createHarness();
    h.controller.lift('a');
    const surface = h.layers.get('a')!.surface as unknown as { callLog: unknown[] };
    const drawn = surface.callLog.length;

    expect(move(h.controller, 40, 0)).toBe(true);
    // Hole x ∈ [20,50) ∪ landing x ∈ [60,90) spans 70×30.
    expect(h.heldBytes()).toBe(commitBytes(70, 30));
    expect(surface.callLog).toHaveLength(drawn);
    expect(h.selection.snapshot).not.toHaveBeenCalled();
  });

  it('keeps the last accepted transform and reports once when a move outgrows the budget', () => {
    const h = createHarness({ byteBudget: commitBytes(40, 30) });
    h.controller.lift('a');

    expect(move(h.controller, 5, 0)).toBe(true);
    expect(move(h.controller, 40, 0)).toBe(false);
    expect(move(h.controller, 60, 0)).toBe(false);
    expect(h.controller.get()?.transform).toMatchObject({ x: 5, y: 0 });
    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('over-budget');

    expect(move(h.controller, 10, 0)).toBe(true);
    expect(h.controller.get()?.transform).toMatchObject({ x: 10, y: 0 });
    expect(h.calls.reportRefusal).toHaveBeenCalledOnce();
  });

  it('records one entry for repeated drags and returns the admission once it lands', () => {
    const h = createHarness();
    h.controller.lift('a');
    for (const x of [5, 15, 40, 12, 30]) {
      move(h.controller, x, 0);
    }
    h.controller.commit();
    expect(h.history.entries().past).toEqual(['Move selection']);
    expect(h.heldBytes()).toBe(0);
    // The entry retains the final 60×30 patch twice plus its overhead (the stub selection has no alpha).
    expect(h.history.byteSize()).toBe(60 * 30 * 8 + 256);
  });
});

describe('FloatingSelectionController: display effects', () => {
  const controlLayer = (withTransparencyEffect: boolean): CanvasLayerContract => ({
    adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
    blendMode: 'normal',
    id: 'a',
    isEnabled: true,
    isLocked: false,
    name: 'a',
    opacity: 1,
    source: { image: { height: 100, imageName: 'a', width: 100 }, type: 'image' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'control',
    withTransparencyEffect,
  });

  it('bakes a display copy for a control layer with the transparency effect', () => {
    const h = createHarness({ layer: controlLayer(true) });
    h.controller.lift('a');

    const float = h.controller.get()!;
    expect(float.display).not.toBeNull();
    expect(float.display).not.toBe(float.pixels.surface);
    expect({ height: float.display!.height, width: float.display!.width }).toEqual({
      height: float.pixels.rect.height,
      width: float.pixels.rect.width,
    });
  });

  it('bakes back the RAW pixels, never the display copy', () => {
    const h = createHarness({ layer: controlLayer(true) });
    h.controller.lift('a');
    const float = h.controller.get()!;
    const raw = float.pixels.surface;

    move(h.controller, 40, 0);
    h.controller.commit();

    const entry = h.layers.get('a')!;
    const drawn = (entry.surface as unknown as { callLog: { op: string; args: unknown[] }[] }).callLog.filter(
      (call) => call.op === 'drawImage'
    );
    expect(drawn.some((call) => call.args[0] === raw.canvas)).toBe(true);
    expect(drawn.some((call) => call.args[0] === float.display!.canvas)).toBe(false);
  });
});

describe('FloatingSelectionController: commit', () => {
  it('pushes exactly one undoable entry for the whole move, however many drags', async () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 5, 0);
    move(h.controller, 12, 3);
    move(h.controller, 40, 10);
    h.controller.commit();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(true);
    await h.history.undo();
    expect(h.history.canUndo()).toBe(false);
  });

  it('puts the pixels, extent and ants back when landing fails midway', () => {
    const h = createHarness();
    const before = { alpha: null, bounds: null, commits: [], rect: null, selected: true };
    (h.selection.snapshot as ReturnType<typeof vi.fn>).mockReturnValueOnce(before);
    h.replaceMask.mockImplementation(() => {
      throw new Error('mask allocation failed');
    });
    h.controller.lift('a');
    move(h.controller, 90, 0);

    expect(() => h.controller.commit()).toThrow('mask allocation failed');

    expect(h.layers.get('a')!.rect).toEqual({ height: 100, width: 100, x: 0, y: 0 });
    expect(h.selection.restore).toHaveBeenLastCalledWith(before);
    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(false);
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.calls.releasePersistence).toHaveBeenCalledOnce();
    expect(h.heldBytes()).toBe(0);
  });

  it('puts the pixels back and reports a commit that can no longer be admitted', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.setLocked(true);

    h.controller.commit();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(false);
    expect(h.calls.reportRefusal).toHaveBeenCalledExactlyOnceWith('busy');
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.replaceMask).not.toHaveBeenCalled();
    expect(h.calls.releasePersistence).toHaveBeenCalledOnce();
    expect(h.heldBytes()).toBe(0);
  });

  it('keeps the step and the ants in place when its pixels cannot be restored', async () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.controller.commit();
    h.calls.applyImagePatch.mockRejectedValueOnce(new Error('evicted'));

    expect(await h.history.undo()).toMatchObject({ status: 'failed' });
    expect(h.history.entries()).toEqual({ evicted: 0, future: [], past: ['Move selection'] });
    expect(h.selection.restore).not.toHaveBeenCalled();
  });

  it('drops the float without committing when its layer has gone', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.removeLayer();
    h.controller.commit();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(false);
  });
});

describe('FloatingSelectionController: cancel', () => {
  it('drops the float and pushes no history', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.controller.cancel();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refills the hole before the layer may persist again, then returns the admission', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    const surface = h.layers.get('a')!.surface as unknown as { callLog: { op: string }[] };
    let refilledAtRelease = false;
    h.calls.releasePersistence.mockImplementation(() => {
      refilledAtRelease = surface.callLog.at(-1)?.op === 'putImageData';
    });

    h.controller.cancel();

    expect(refilledAtRelease).toBe(true);
    expect(h.heldBytes()).toBe(0);
  });
});

describe('FloatingSelectionController: dispose', () => {
  let harness: ReturnType<typeof createHarness>;

  beforeEach(() => {
    harness = createHarness();
  });

  it('puts a live float back rather than dropping its pixels on the floor', () => {
    harness.controller.lift('a');
    move(harness.controller, 40, 0);
    harness.controller.dispose();

    expect(harness.controller.has()).toBe(false);
    expect(harness.history.canUndo()).toBe(false);
    expect(harness.calls.notifyPainted).toHaveBeenCalledTimes(2);
    expect(harness.calls.releasePersistence).toHaveBeenCalledOnce();
    expect(harness.heldBytes()).toBe(0);
  });

  it('is idempotent', () => {
    harness.controller.dispose();
    harness.controller.dispose();
    expect(harness.controller.has()).toBe(false);
  });
});
