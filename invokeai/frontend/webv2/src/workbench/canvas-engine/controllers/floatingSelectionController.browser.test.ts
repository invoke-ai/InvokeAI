import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createSelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { describe, expect, it, vi } from 'vitest';

import { FloatingSelectionController } from './floatingSelectionController';

const LAYER: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'a',
  isEnabled: true,
  isLocked: false,
  name: 'a',
  opacity: 1,
  source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const DOCUMENT: CanvasDocumentContractV3 = {
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  selectedLayerId: 'a',
  stacks: stacksFrom([LAYER]),
  version: 3,
  width: 100,
};

/** Every view reads this fixed area, so caches of different extents compare pixel for pixel. */
const VIEW: Rect = { height: 40, width: 80, x: 0, y: 0 };

const createHarness = (byteBudget = 64 * 1024 * 1024) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  const history = createHistory({ byteBudget });
  const selection = createSelectionState({
    backend,
    createPath2D: (d) => new Path2D(d),
    getDocumentSize: () => ({ height: 100, width: 100 }),
    onChange: () => undefined,
  });

  // Distinct opaque pixels everywhere, so any misplaced copy shows.
  const entry = layers.getOrCreateRect('a', { height: 40, width: 40, x: 0, y: 0 });
  const seed = entry.surface.ctx.createImageData(40, 40);
  for (let index = 0; index < 40 * 40; index += 1) {
    seed.data.set([index % 40, Math.floor(index / 40), 200, 255], index * 4);
  }
  entry.surface.ctx.putImageData(seed, 0, 0);

  const mask = backend.createSurface(10, 10);
  mask.ctx.fillStyle = '#fff';
  mask.ctx.fillRect(0, 0, 10, 10);
  selection.replaceMask({ rect: { height: 10, width: 10, x: 10, y: 10 }, surface: mask });

  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: () => true,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => DOCUMENT,
    getReducerDocument: () => DOCUMENT,
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
  const reportRefusal = vi.fn();
  const controller = new FloatingSelectionController({
    applyImagePatch: (layerId, rect, pixels) => {
      const target = layers.growToRect(layerId, rect);
      target.surface.ctx.putImageData(pixels, rect.x - target.rect.x, rect.y - target.rect.y);
      return Promise.resolve();
    },
    backend,
    ctx,
    getDocument: () => DOCUMENT,
    invalidateLayer: () => undefined,
    layers,
    markDirty: () => undefined,
    notifyPainted: () => undefined,
    onChange: () => undefined,
    reportRefusal,
    selection,
    suspendPersistence: () => () => undefined,
  });

  /** The layer's pixels over {@link VIEW}, transparent where the cache does not reach. */
  const pixels = (): number[] => {
    const view = backend.createSurface(VIEW.width, VIEW.height);
    const cache = layers.get('a')!;
    view.ctx.drawImage(cache.surface.canvas, cache.rect.x - VIEW.x, cache.rect.y - VIEW.y);
    return [...view.ctx.getImageData(0, 0, VIEW.width, VIEW.height).data];
  };
  const pixelAt = (x: number, y: number): number[] =>
    pixels().slice((y * VIEW.width + x) * 4, (y * VIEW.width + x) * 4 + 4);
  const selected = () => {
    const snapshot = selection.snapshot();
    return { alpha: snapshot.alpha ? [...snapshot.alpha] : null, rect: snapshot.rect };
  };

  return { controller, history, pixelAt, pixels, reportRefusal, selected };
};

describe('FloatingSelectionController in a real raster backend', () => {
  it('restores exact pixels and selection on undo, and the landed ones on redo', async () => {
    const h = createHarness();
    const originalPixels = h.pixels();
    const originalSelection = h.selected();

    expect(h.controller.lift('a')).toBe('lifted');
    expect(h.controller.setTransform({ rotation: 0, scaleX: 1, scaleY: 1, x: 35, y: 0 })).toBe(true);
    h.controller.commit();

    // The hole is clear and the pixels landed past the original cache edge.
    expect(h.pixelAt(12, 12)).toEqual([0, 0, 0, 0]);
    expect(h.pixelAt(47, 12)).toEqual([12, 12, 200, 255]);
    expect(h.selected().rect).toEqual({ height: 10, width: 10, x: 45, y: 10 });
    const landedPixels = h.pixels();
    const landedSelection = h.selected();

    expect(await h.history.undo()).toEqual({ status: 'applied' });
    expect(h.pixels()).toEqual(originalPixels);
    expect(h.selected()).toEqual(originalSelection);

    expect(await h.history.redo()).toEqual({ status: 'applied' });
    expect(h.pixels()).toEqual(landedPixels);
    expect(h.selected()).toEqual(landedSelection);
  });

  it('restores exact pixels on cancel, recording nothing', () => {
    const h = createHarness();
    const originalPixels = h.pixels();

    h.controller.lift('a');
    h.controller.setTransform({ rotation: 0, scaleX: 1, scaleY: 1, x: 20, y: 5 });
    h.controller.cancel();

    expect(h.pixels()).toEqual(originalPixels);
    expect(h.history.canUndo()).toBe(false);
  });

  it('commits an enlarged, rotated float with every other byte of the budget taken', () => {
    const budget = 1024 * 1024;
    const h = createHarness(budget);
    h.controller.lift('a');
    expect(h.controller.setTransform({ rotation: Math.PI / 6, scaleX: 1.5, scaleY: 2, x: 12, y: 3 })).toBe(true);

    // Take the largest edit that still fits beside the float's admission.
    let low = 0;
    let high = budget;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      const probe = h.history.admit(mid);
      probe?.release();
      if (probe) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    const rest = h.history.admit(low)!;

    h.controller.commit();

    expect(h.history.entries().past).toEqual(['Move selection']);
    expect(h.reportRefusal).not.toHaveBeenCalled();
    rest.release();
  });
});
