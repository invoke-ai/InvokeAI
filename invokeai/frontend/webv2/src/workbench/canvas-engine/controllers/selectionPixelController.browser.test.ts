import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it, vi } from 'vitest';

import { SelectionPixelController } from './selectionPixelController';

/** Selection Fill/Erase on `layer`, whose 20×20 cache sits at its local origin, over a document-space selection. */
const setup = (layer: CanvasLayerContract, selectionRect: Rect, prefill?: string | { color: string; rect: Rect }) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  const entry = layers.getOrCreateRect(layer.id, { height: 20, width: 20, x: 0, y: 0 });
  entry.stale = false;
  if (prefill) {
    const area = typeof prefill === 'string' ? { height: 20, width: 20, x: 0, y: 0 } : prefill.rect;
    entry.surface.ctx.fillStyle = typeof prefill === 'string' ? prefill : prefill.color;
    entry.surface.ctx.fillRect(area.x, area.y, area.width, area.height);
  }
  const mask = backend.createSurface(selectionRect.width, selectionRect.height);
  mask.ctx.fillStyle = '#000';
  mask.ctx.fillRect(0, 0, selectionRect.width, selectionRect.height);
  const selection = {
    bounds: () => selectionRect,
    mask: () => ({ rect: selectionRect, surface: mask }),
  } as unknown as SelectionState;
  const document = documentFrom([layer], layer.id);
  const history = createHistory();
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: () => true,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
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
  const controller = new SelectionPixelController({
    // History writes the patch back at its layer-local rect, as the engine's bridge does.
    applyImagePatch: (layerId, rect, pixels) => {
      const cache = layers.get(layerId)!;
      cache.surface.ctx.putImageData(pixels, rect.x - cache.rect.x, rect.y - cache.rect.y);
      return Promise.resolve();
    },
    backend,
    beginPixelEdit: () => null,
    canEdit: () => true,
    ctx,
    deleteDerived: vi.fn(),
    getDocument: () => document,
    getFillColor: () => '#ff0000',
    getTransformSessionLayerId: () => null,
    invalidateLayer: vi.fn(),
    isGestureActive: () => false,
    isRasterCacheReady: () => true,
    layers,
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    reportRefusal: vi.fn(),
    requestRasterization: vi.fn(),
    selection,
  });
  /** RGBA at a layer-local pixel of the cache; transparent outside it. */
  const pixelAt = (x: number, y: number): number[] => {
    const cache = layers.get(layer.id)!;
    const { rect } = cache;
    if (x < rect.x || y < rect.y || x >= rect.x + rect.width || y >= rect.y + rect.height) {
      return [0, 0, 0, 0];
    }
    return [...cache.surface.ctx.getImageData(x - rect.x, y - rect.y, 1, 1).data];
  };
  const cacheRect = () => ({ ...layers.get(layer.id)!.rect });
  return { cacheRect, controller, history, pixelAt };
};

describe('selection Fill and Erase in Chromium', () => {
  it('fills a moved paint layer where the selection covers it, and undo clears it again', async () => {
    const layer = layerContract('paint', 'raster', { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 5 } });
    // Document 12..16 × 7..11 is local 2..6 × 2..6.
    const { controller, history, pixelAt } = setup(layer, { height: 4, width: 4, x: 12, y: 7 });

    controller.run('fill');

    expect(pixelAt(3, 3)).toEqual([255, 0, 0, 255]);
    expect(pixelAt(1, 3)).toEqual([0, 0, 0, 0]);
    expect(pixelAt(13, 8)).toEqual([0, 0, 0, 0]);
    await history.undo();
    expect(pixelAt(3, 3)).toEqual([0, 0, 0, 0]);
    await history.redo();
    expect(pixelAt(3, 3)).toEqual([255, 0, 0, 255]);
  });

  it('fills a rotated, flipped paint layer beyond its cached pixels, growing the cache to take them', async () => {
    // Flipped, then a quarter turn, at (40, 0): local (x, y) lands at document (40 - y, -x).
    const layer = layerContract('paint', 'raster', {
      transform: { rotation: Math.PI / 2, scaleX: -1, scaleY: 1, x: 40, y: 0 },
    });
    // Document 10..14 × -30..-26 is local 26..30 × 26..30, outside the 20×20 cache.
    const { cacheRect, controller, history, pixelAt } = setup(layer, { height: 4, width: 4, x: 10, y: -30 });

    controller.run('fill');

    // The rotation's rounding may add a pixel; the cache must at least reach local (30, 30).
    const grown = cacheRect();
    expect([grown.x, grown.y, grown.x + grown.width >= 30, grown.y + grown.height >= 30]).toEqual([0, 0, true, true]);
    expect(pixelAt(28, 28)).toEqual([255, 0, 0, 255]);
    expect(pixelAt(25, 28)[3]).toBe(0);
    expect(pixelAt(28, 25)[3]).toBe(0);
    await history.undo();
    expect(pixelAt(28, 28)[3]).toBe(0);
    await history.redo();
    expect(pixelAt(28, 28)).toEqual([255, 0, 0, 255]);
  });

  it('fills only the existing pixels of a transparency-locked paint layer', () => {
    const layer = layerContract('paint', 'raster', { isTransparencyLocked: true } as Partial<CanvasLayerContract>);
    const { cacheRect, controller, pixelAt } = setup(
      layer,
      { height: 30, width: 30, x: -5, y: -5 },
      { color: '#ffffff', rect: { height: 20, width: 10, x: 0, y: 0 } }
    );

    controller.run('fill');

    expect(pixelAt(5, 5)).toEqual([255, 0, 0, 255]);
    expect(pixelAt(15, 5)[3]).toBe(0);
    expect(cacheRect()).toEqual({ height: 20, width: 20, x: 0, y: 0 });
  });

  it('fills a mask, and undo and redo take its coverage away and back', async () => {
    const { controller, history, pixelAt } = setup(layerContract('mask', 'inpaint_mask'), {
      height: 4,
      width: 4,
      x: 2,
      y: 2,
    });

    controller.run('fill');
    expect(pixelAt(3, 3)).toEqual([255, 255, 255, 255]);
    await history.undo();
    expect(pixelAt(3, 3)[3]).toBe(0);
    await history.redo();
    expect(pixelAt(3, 3)).toEqual([255, 255, 255, 255]);
  });

  it('fills a scaled mask with opaque white coverage through its transform', () => {
    const layer = layerContract('mask', 'inpaint_mask', {
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 0, y: 0 },
    });
    // Document 4..8 × 0..4 is local 2..4 × 0..2.
    const { controller, pixelAt } = setup(layer, { height: 4, width: 4, x: 4, y: 0 });

    controller.run('fill');

    expect(pixelAt(2, 0)).toEqual([255, 255, 255, 255]);
    expect(pixelAt(3, 1)).toEqual([255, 255, 255, 255]);
    expect(pixelAt(1, 0)[3]).toBe(0);
    expect(pixelAt(4, 0)[3]).toBe(0);
    expect(pixelAt(5, 1)[3]).toBe(0);
  });

  it('erases a rotated mask under the selection only', () => {
    // A quarter turn about the origin: local (x, y) lands at document (-y, x).
    const layer = layerContract('mask', 'inpaint_mask', {
      transform: { rotation: Math.PI / 2, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    // Document -6..-2 × 2..6 is local 2..6 × 2..6.
    const { controller, pixelAt } = setup(layer, { height: 4, width: 4, x: -6, y: 2 }, '#ffffff');

    controller.run('erase');

    expect(pixelAt(4, 4)[3]).toBe(0);
    expect(pixelAt(1, 4)[3]).toBe(255);
    expect(pixelAt(4, 7)[3]).toBe(255);
  });
});
