import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it, vi } from 'vitest';

import { CropLayerController } from './cropLayerController';
import { createLayerOperationHarness } from './layerOperationHarness.testing';

const layer: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'paint',
  isEnabled: true,
  isLocked: false,
  name: 'Paint',
  opacity: 1,
  source: { bitmap: null, offset: { x: -10, y: -10 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
};

const CACHE_RECT: Rect = { height: 50, width: 50, x: -10, y: -10 };
const BBOX: Rect = { height: 20, width: 20, x: 0, y: 0 };

const createHarness = (historyBytes?: number) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: BBOX,
    height: 100,
    selectedLayerId: 'paint',
    stacks: stacksFrom([layer]),
    version: 3,
    width: 100,
  };
  const h = createLayerOperationHarness(document, { historyBytes });
  h.layers.getOrCreateRect('paint', CACHE_RECT);
  const discardPersisted = vi.fn();
  const controller = new CropLayerController({
    backend: h.backend,
    captureCache: (target) => {
      const entry = h.layers.get(target.id)!;
      const pixels = h.backend.createSurface(entry.rect.width, entry.rect.height);
      pixels.ctx.drawImage(entry.surface.canvas, 0, 0);
      return { pixels, rect: { ...entry.rect } };
    },
    ctx: h.ctx,
    discardPersisted,
    exportBaked: (layerId) => {
      const entry = h.layers.get(layerId)!;
      const current = getDocumentLayer(h.document(), layerId)!;
      return Promise.resolve({
        guard: { layer: current } as never,
        rect: { ...entry.rect },
        release: () => undefined,
        status: 'ok' as const,
        surface: entry.surface,
      });
    },
    isGuardCurrent: () => true,
    isSupportedSource: () => true,
  });
  return { ...h, controller, discardPersisted, original: getDocumentLayer(h.document(), 'paint')! };
};

describe('CropLayerController', () => {
  it('crops the layer to the bbox and replays the contract and pixels together', async () => {
    const h = createHarness();

    await expect(h.controller.crop('paint')).resolves.toEqual({ status: 'cropped' });
    expect(getDocumentLayer(h.document(), 'paint')).toMatchObject({ source: { offset: { x: 0, y: 0 } } });
    expect(h.layers.get('paint')?.rect).toEqual(BBOX);
    expect(h.discardPersisted).toHaveBeenCalledWith('paint');

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'paint')).toEqual(h.original);
    expect(h.layers.get('paint')?.rect).toEqual(CACHE_RECT);

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(h.layers.get('paint')?.rect).toEqual(BBOX);
  });

  it('refuses a crop it could never undo before anything changes', async () => {
    const h = createHarness(1024);
    const before = h.document();

    await expect(h.controller.crop('paint')).resolves.toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
    expect(h.layers.get('paint')?.rect).toEqual(CACHE_RECT);
  });

  it('leaves the step on the undo stack when its replay is refused', async () => {
    const h = createHarness();
    await h.controller.crop('paint');
    h.state.refuse = () => true;

    expect((await h.history.undo()).status).toBe('failed');
    expect(h.layers.get('paint')?.rect).toEqual(BBOX);
    expect(h.history.canUndo()).toBe(true);
  });
});
