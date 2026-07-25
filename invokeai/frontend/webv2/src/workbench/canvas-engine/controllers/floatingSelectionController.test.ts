import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { createHistory } from '@workbench/canvas-engine/history/history';
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

const makeDoc = (layers: CanvasLayerContract[]): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId: layers[0]?.id ?? null,
  version: 2,
  width: 100,
});

const createHarness = (options: { layer?: CanvasLayerContract; maskRect?: Rect; cacheRect?: Rect } = {}) => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const history = createHistory();
  const layer = options.layer ?? paintLayer('a');
  let document: CanvasDocumentContractV2 | null = makeDoc([layer]);

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
    selectAll: vi.fn(),
  } as SelectionState;

  // Seed a cache so there is something to lift out of.
  layers.getOrCreateRect(layer.id, options.cacheRect ?? { height: 100, width: 100, x: 0, y: 0 });

  const calls = {
    applyImagePatch: vi.fn(),
    endBurst: vi.fn(),
    invalidateLayer: vi.fn(),
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    onChange: vi.fn(),
  };

  const controller = new FloatingSelectionController({
    applyImagePatch: calls.applyImagePatch,
    backend,
    canEdit: () => true,
    endBurst: calls.endBurst,
    getDocument: () => document,
    history,
    invalidateLayer: calls.invalidateLayer,
    layers,
    markDirty: calls.markDirty,
    notifyPainted: calls.notifyPainted,
    onChange: calls.onChange,
    selection,
  });

  return {
    backend,
    calls,
    controller,
    history,
    layer,
    layers,
    replaceMask,
    removeLayer: () => {
      document = makeDoc([]);
    },
  };
};

/** Moves the float by a layer-local delta. */
const move = (controller: FloatingSelectionController, x: number, y: number): void =>
  controller.setTransform({ rotation: 0, scaleX: 1, scaleY: 1, x, y });

describe('FloatingSelectionController: lift', () => {
  it('lifts the selection region and reports a live float', () => {
    const h = createHarness();
    expect(h.controller.lift('a')).toBe(true);
    expect(h.controller.has()).toBe(true);
    expect(h.controller.get()?.layerId).toBe('a');
    expect(h.controller.get()?.pixels.rect).toEqual({ height: 30, width: 30, x: 20, y: 20 });
    // Identity until something drags it.
    expect(h.controller.get()?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
  });

  it('recomposites the hole but never marks the layer dirty', () => {
    // Persisting mid-float would upload a holed bitmap the user never asked for.
    const h = createHarness();
    h.controller.lift('a');
    expect(h.calls.notifyPainted).toHaveBeenCalledWith('a');
    expect(h.calls.markDirty).not.toHaveBeenCalled();
  });

  it('pushes no history entry — only a commit is undoable', () => {
    const h = createHarness();
    h.controller.lift('a');
    expect(h.history.canUndo()).toBe(false);
  });

  it('closes any coalescing burst before touching pixels', () => {
    const h = createHarness();
    h.controller.lift('a');
    expect(h.calls.endBurst).toHaveBeenCalled();
  });

  it('refuses a second lift while a float is live', () => {
    const h = createHarness();
    expect(h.controller.lift('a')).toBe(true);
    expect(h.controller.lift('a')).toBe(false);
  });

  it('refuses a locked layer', () => {
    const h = createHarness({ layer: { ...paintLayer('a'), isLocked: true } });
    expect(h.controller.lift('a')).toBe(false);
    expect(h.controller.has()).toBe(false);
  });

  it('refuses a hidden layer', () => {
    const h = createHarness({ layer: { ...paintLayer('a'), isEnabled: false } });
    expect(h.controller.lift('a')).toBe(false);
  });

  it('refuses an unknown layer', () => {
    const h = createHarness();
    expect(h.controller.lift('nope')).toBe(false);
  });

  it('refuses when the selection misses the layer content', () => {
    const h = createHarness({ cacheRect: { height: 10, width: 10, x: 0, y: 0 } });
    expect(h.controller.lift('a')).toBe(false);
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
    // Without it the float would show the control map's raw, opaque black
    // background — the effect the layer itself renders away.
    const h = createHarness({ layer: controlLayer(true) });
    h.controller.lift('a');

    const float = h.controller.get()!;
    expect(float.display).not.toBeNull();
    expect(float.display).not.toBe(float.pixels.surface);
    // Same extent as the lifted pixels, so it blits at the same rect.
    expect({ height: float.display!.height, width: float.display!.width }).toEqual({
      height: float.pixels.rect.height,
      width: float.pixels.rect.width,
    });
  });

  it('leaves the display copy null when the layer has no effect', () => {
    const h = createHarness({ layer: controlLayer(false) });
    h.controller.lift('a');
    expect(h.controller.get()!.display).toBeNull();
  });

  it('bakes back the RAW pixels, never the display copy', () => {
    // The effect is display-only; burning it into the document would darken the
    // layer a little more on every move.
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
  it('pushes exactly one undoable entry for the whole move, however many drags', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 5, 0);
    move(h.controller, 12, 3);
    move(h.controller, 40, 10);
    h.controller.commit();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(true);
    h.history.undo();
    expect(h.history.canUndo()).toBe(false);
  });

  it('spans both the hole and the landing region so one undo restores both', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.controller.commit();

    // The commit writes the cache directly; `applyImagePatch` is the history
    // entry's replay bridge, so it first runs on undo.
    expect(h.calls.applyImagePatch).not.toHaveBeenCalled();
    h.history.undo();

    const [layerId, rect] = h.calls.applyImagePatch.mock.calls[0] as [string, Rect];
    expect(layerId).toBe('a');
    // The hole at x ∈ [20,50) unioned with the landing region at x ∈ [60,90).
    expect(rect.x).toBe(20);
    expect(rect.x + rect.width).toBe(90);
  });

  it('marks the layer dirty so the baked pixels persist', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.controller.commit();
    expect(h.calls.markDirty).toHaveBeenCalledWith('a');
  });

  it('carries the selection mask along so the ants land on the moved pixels', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 10);
    h.controller.commit();

    expect(h.replaceMask).toHaveBeenCalledTimes(1);
    const moved = h.replaceMask.mock.calls[0]![0] as PlacedSurface;
    expect(moved.rect).toEqual({ height: 30, width: 30, x: 60, y: 30 });
  });

  it('scales a layer-local move into document space for the mask', () => {
    // On a 2× layer, a 20px local move is a 40px document move.
    const h = createHarness({ layer: paintLayer('a', { scaleX: 2, scaleY: 2 }) });
    h.controller.lift('a');
    move(h.controller, 20, 0);
    h.controller.commit();

    const moved = h.replaceMask.mock.calls[0]![0] as PlacedSurface;
    expect(moved.rect.x).toBe(60);
  });

  it('treats an unmoved float as a cancel — nothing changed, so nothing is undoable', () => {
    const h = createHarness();
    h.controller.lift('a');
    h.controller.commit();

    expect(h.controller.has()).toBe(false);
    expect(h.history.canUndo()).toBe(false);
    expect(h.replaceMask).not.toHaveBeenCalled();
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

  it('is a no-op with no float', () => {
    const h = createHarness();
    h.controller.commit();
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

  it('restores the pixels without marking dirty (the layer never changed)', () => {
    const h = createHarness();
    h.controller.lift('a');
    h.controller.cancel();

    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.calls.notifyPainted).toHaveBeenCalledTimes(2);
  });

  it('leaves the selection where it was', () => {
    const h = createHarness();
    h.controller.lift('a');
    move(h.controller, 40, 0);
    h.controller.cancel();
    expect(h.replaceMask).not.toHaveBeenCalled();
  });

  it('is a no-op with no float', () => {
    const h = createHarness();
    h.controller.cancel();
    expect(h.calls.notifyPainted).not.toHaveBeenCalled();
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
    // The restore pass ran.
    expect(harness.calls.notifyPainted).toHaveBeenCalledTimes(2);
  });

  it('refuses further lifts once disposed', () => {
    harness.controller.dispose();
    expect(harness.controller.lift('a')).toBe(false);
  });

  it('is idempotent', () => {
    harness.controller.dispose();
    harness.controller.dispose();
    expect(harness.controller.has()).toBe(false);
  });
});
