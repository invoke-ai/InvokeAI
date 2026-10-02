import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { SelectionPixelController } from './selectionPixelController';

const paintDocument = (source: Record<string, unknown> = { type: 'paint' }): CanvasDocumentContractV3 =>
  ({
    stacks: stacksFrom([
      {
        id: 'paint',
        isEnabled: true,
        isLocked: false,
        source,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
      } as unknown as CanvasLayerContract,
    ]),
    selectedLayerId: 'paint',
  }) as CanvasDocumentContractV3;

const createHarness = (
  options: {
    document?: CanvasDocumentContractV3;
    cacheRect?: Rect | null;
    selectionRect?: Rect;
    byteBudget?: number;
    mirrored?: boolean;
    cacheReady?: boolean;
  } = {}
) => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const cacheRect = options.cacheRect === undefined ? { height: 2, width: 2, x: 0, y: 0 } : options.cacheRect;
  if (cacheRect) {
    const { ctx } = layers.getOrCreateRect('paint', cacheRect).surface;
    // Stub readbacks are constant; number them so a fill's before and after differ.
    const read = ctx.getImageData.bind(ctx);
    let reads = 0;
    Object.defineProperty(ctx, 'getImageData', {
      value: (...args: Parameters<typeof read>) => {
        const pixels = read(...args);
        pixels.data[0] = reads++;
        return pixels;
      },
    });
  }
  const selectionRect = options.selectionRect ?? { height: 2, width: 2, x: 0, y: 0 };
  const mask = backend.createSurface(selectionRect.width, selectionRect.height);
  const selection = {
    bounds: () => selectionRect,
    mask: () => ({ rect: selectionRect, surface: mask }),
  } as unknown as SelectionState;
  const document = options.document ?? paintDocument();
  // A mirror that never follows the reducer fails every publication's postconditions.
  const reducerDocument = options.mirrored === false ? { ...document } : document;
  const history = createHistory({ byteBudget: options.byteBudget });
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: () => true,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => document,
    getReducerDocument: () => reducerDocument,
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
  const calls = {
    deleteDerived: vi.fn(),
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    reportRefusal: vi.fn(),
    requestRasterization: vi.fn(),
  };
  const controller = new SelectionPixelController({
    applyImagePatch: vi.fn(() => Promise.resolve()),
    backend,
    beginPixelEdit: () => null,
    canEdit: () => true,
    ctx,
    deleteDerived: calls.deleteDerived,
    getDocument: () => document,
    getFillColor: () => '#f00',
    invalidateLayer: vi.fn(),
    isGestureActive: () => false,
    isRasterCacheReady: () => options.cacheReady ?? true,
    layers,
    markDirty: calls.markDirty,
    notifyPainted: calls.notifyPainted,
    reportRefusal: calls.reportRefusal,
    requestRasterization: calls.requestRasterization,
    selection,
  });
  return { calls, controller, history, layers };
};

describe('SelectionPixelController', () => {
  it('records a raster fill as one step and persists it', () => {
    const h = createHarness();

    h.controller.run('fill');

    expect(h.calls.markDirty).toHaveBeenCalledWith('paint');
    expect(h.calls.notifyPainted).toHaveBeenCalledWith('paint');
    expect(h.history.entries().past).toEqual(['Fill selection']);
    // Before and after of the touched 2×2 region only.
    expect(h.history.byteSize()).toBe(2 * 2 * 4 * 2);
  });

  it('requests durable paint pixels instead of growing a transparent fill cache', () => {
    const h = createHarness({
      cacheReady: false,
      cacheRect: null,
      document: paintDocument({ bitmap: { height: 2, imageName: 'durable', width: 2 }, type: 'paint' }),
    });

    h.controller.run('fill');

    expect(h.calls.requestRasterization).toHaveBeenCalledWith('paint');
    expect(h.layers.peek('paint')).toBeUndefined();
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses a fill too large to undo before touching the cache', () => {
    const h = createHarness({ byteBudget: 10, selectionRect: { height: 4, width: 4, x: -1, y: -1 } });
    const version = h.layers.version('paint');

    h.controller.run('fill');

    expect(h.calls.reportRefusal).toHaveBeenCalledWith('over-budget');
    expect(h.layers.get('paint')!.rect).toEqual({ height: 2, width: 2, x: 0, y: 0 });
    expect(h.layers.version('paint')).toBe(version);
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it('restores the touched region and the original extent when the fill cannot be published', () => {
    const h = createHarness({ mirrored: false, selectionRect: { height: 4, width: 4, x: -1, y: -1 } });
    const surface = h.layers.get('paint')!.surface as StubRasterSurface;

    h.controller.run('fill');

    expect(h.layers.get('paint')!.rect).toEqual({ height: 2, width: 2, x: 0, y: 0 });
    // The before pixels went back over the whole touched (grown) region before the crop.
    const restored = surface.callLog.filter((entry) => entry.op === 'putImageData').at(-1);
    expect(restored?.args.slice(1, 3)).toEqual([0, 0]);
    expect(h.calls.deleteDerived).toHaveBeenCalledWith('paint');
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });
});
