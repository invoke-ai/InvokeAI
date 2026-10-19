import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { RasterizeLayerController } from './rasterizeLayerController';

const shapeLayer: CanvasLayerContract = {
  blendMode: 'normal',
  id: 'shape',
  isEnabled: true,
  isLocked: false,
  name: 'Shape',
  opacity: 1,
  source: { fill: '#000', height: 40, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 60 },
  transform: { rotation: 0, scaleX: 2, scaleY: 1, x: 10, y: 20 },
  type: 'raster',
};

const createHarness = (historyBytes?: number) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    selectedLayerId: 'shape',
    stacks: stacksFrom([shapeLayer]),
    version: 3,
    width: 100,
  };
  const h = createLayerOperationHarness(document, { historyBytes });
  const controller = new RasterizeLayerController({
    backend: h.backend,
    ctx: h.ctx,
    rasterizeDeps: () => ({
      backend: h.backend,
      documentSize: { height: 100, width: 100 },
      resolver: () => Promise.reject(new Error('shapes need no images')),
      store: h.layers,
    }),
  });
  return { ...h, controller };
};

const BAKED_RECT = { height: 40, width: 120, x: 10, y: 20 };

describe('RasterizeLayerController', () => {
  it('bakes source and transform into paint pixels as one undo step', () => {
    const h = createHarness();

    expect(h.controller.rasterize('shape')).toBe('rasterized');
    expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({
      source: { bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    expect(h.layers.get('shape')?.rect).toEqual(BAKED_RECT);
    expect(h.history.entries().past).toEqual(['Rasterize layer']);
  });

  it('undo restores the parametric layer; redo reinstalls the same baked pixels without rerendering', async () => {
    const h = createHarness();
    h.controller.rasterize('shape');
    const baked = h.installed[0]!;

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'shape')).toEqual(shapeLayer);

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({ source: { type: 'paint' } });
    expect(h.installed).toHaveLength(2);
    expect(h.installed[1]!.rect).toEqual(baked.rect);
  });

  it('refuses a conversion it could never undo, leaving the layer parametric', () => {
    const h = createHarness(1024);
    const before = h.document();

    expect(h.controller.rasterize('shape')).toBe('over-budget');
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
  });

  it('reports a refused conversion as failed and installs nothing', () => {
    const h = createHarness();
    h.state.refuse = () => true;

    expect(h.controller.rasterize('shape')).toBe('failed');
    expect(h.installed).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses missing, locked and already-painted layers', () => {
    const h = createHarness();
    expect(h.controller.rasterize('nope')).toBe('missing');
    h.controller.rasterize('shape');
    expect(h.controller.rasterize('shape')).toBe('unsupported');
  });
});
