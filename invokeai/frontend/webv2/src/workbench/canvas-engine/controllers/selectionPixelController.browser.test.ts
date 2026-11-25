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
});
