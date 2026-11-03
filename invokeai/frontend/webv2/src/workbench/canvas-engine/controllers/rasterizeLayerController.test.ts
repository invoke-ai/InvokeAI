import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SpatialRunner } from '@workbench/canvas-engine/render/exactAdjustments';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it, vi } from 'vitest';

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

const createHarness = (historyBytes?: number, layer: CanvasLayerContract = shapeLayer, runSpatial?: SpatialRunner) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    selectedLayerId: 'shape',
    stacks: stacksFrom([layer]),
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
    runSpatial,
  });
  return { ...h, controller };
};

const BAKED_RECT = { height: 40, width: 120, x: 10, y: 20 };

describe('RasterizeLayerController', () => {
  it('bakes source and transform into paint pixels as one undo step', async () => {
    const h = createHarness();

    expect(await h.controller.rasterize('shape')).toBe('rasterized');
    expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({
      source: { bitmap: null, offset: { x: 10, y: 20 }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    });
    expect(h.layers.get('shape')?.rect).toEqual(BAKED_RECT);
    expect(h.history.entries().past).toEqual(['Rasterize layer']);
  });

  it('undo restores the parametric layer; redo reinstalls the same baked pixels without rerendering', async () => {
    const h = createHarness();
    await h.controller.rasterize('shape');
    const baked = h.installed[0]!;

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'shape')).toEqual(shapeLayer);

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({ source: { type: 'paint' } });
    expect(h.installed).toHaveLength(2);
    expect(h.installed[1]!.rect).toEqual(baked.rect);
  });

  it('refuses a conversion it could never undo, leaving the layer parametric', async () => {
    const h = createHarness(1024);
    const before = h.document();

    expect(await h.controller.rasterize('shape')).toBe('over-budget');
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
  });

  it('reports a refused conversion as failed and installs nothing', async () => {
    const h = createHarness();
    h.state.refuse = () => true;

    expect(await h.controller.rasterize('shape')).toBe('failed');
    expect(h.installed).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('keeps a color-only stack on the paint contract', async () => {
    const adjustments = [{ id: 'invert', isEnabled: true, type: 'invert' }] as const;
    const h = createHarness(undefined, { ...shapeLayer, adjustments });

    expect(await h.controller.rasterize('shape')).toBe('rasterized');
    expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({ adjustments, source: { type: 'paint' } });
  });

  it('reserves the readback that bakes a noisy stack', async () => {
    const noise = {
      amount: 0.2,
      distribution: 'uniform',
      grain: 'monochrome',
      id: 'grain',
      isEnabled: true,
      response: 'flat',
      seed: 1,
      size: 1,
      type: 'noise',
    } as const;
    const h = createHarness(undefined, { ...shapeLayer, adjustments: [noise] });
    // The 60×40 source render, plus the 120×40 baked pixels and their prepared replacement.
    h.state.rasterBytes = (60 * 40 + 120 * 40 * 2) * 4;

    expect(await h.controller.rasterize('shape')).toBe('over-budget');
    expect(h.dispatched).toEqual([]);
    h.state.rasterBytes += 60 * 40 * 4;
    expect(await h.controller.rasterize('shape')).toBe('rasterized');
    expect(getDocumentLayer(h.document(), 'shape')).not.toHaveProperty('adjustments');
  });

  it('refuses missing, locked and already-painted layers', async () => {
    const h = createHarness();
    expect(await h.controller.rasterize('nope')).toBe('missing');
    await h.controller.rasterize('shape');
    expect(await h.controller.rasterize('shape')).toBe('unsupported');
  });

  describe('with a spatial stack', () => {
    const spatial = {
      ...shapeLayer,
      adjustments: [
        { amount: 50, id: 'texture', isEnabled: true, size: 6, type: 'texture' },
        { id: 'invert', isEnabled: true, type: 'invert' },
      ],
    } as const satisfies CanvasLayerContract;

    it('bakes the whole stack on the layer-local render in the spatial service and drops it', async () => {
      const runSpatial = vi.fn<SpatialRunner>((request) => Promise.resolve({ pixels: request.pixels, status: 'ok' }));
      const h = createHarness(undefined, spatial, runSpatial);

      expect(await h.controller.rasterize('shape')).toBe('rasterized');

      // The 60×40 source render, before the transform scales it.
      expect(runSpatial).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ height: 40, priority: 'now', width: 60 })
      );
      expect(getDocumentLayer(h.document(), 'shape')).not.toHaveProperty('adjustments');
    });

    it.each(['too-large', 'over-budget', 'error'] as const)(
      'refuses %s with the layer name, leaving the layer parametric',
      async (reason) => {
        const h = createHarness(undefined, spatial, () => Promise.resolve({ status: reason }));

        expect(await h.controller.rasterize('shape')).toEqual({
          ownerId: 'shape',
          ownerName: 'Shape',
          reason,
          status: 'adjustment-refused',
        });
        expect(h.dispatched).toEqual([]);
        expect(h.reservedBytes()).toBe(0);
      }
    );

    it('refuses without a spatial runner rather than bake without the spatial entries', async () => {
      const h = createHarness(undefined, spatial);

      expect(await h.controller.rasterize('shape')).toMatchObject({ reason: 'error', status: 'adjustment-refused' });
      expect(h.dispatched).toEqual([]);
    });

    it('lands nothing when an edit landed while the job ran', async () => {
      const h = createHarness(undefined, spatial, (request) => {
        h.ctx.dispatch({ id: null, type: 'setCanvasSelectedLayer' });
        return Promise.resolve({ pixels: request.pixels, status: 'ok' });
      });

      expect(await h.controller.rasterize('shape')).toBe('not-ready');
      expect(getDocumentLayer(h.document(), 'shape')).toMatchObject({ source: { type: 'shape' } });
      expect(h.history.canUndo()).toBe(false);
    });
  });
});
