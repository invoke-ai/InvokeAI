import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';

import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { MaskResultController } from './maskResultController';

const createHarness = () => {
  const h = createLayerOperationHarness(documentFrom([layerContract('source', 'raster')], 'source'));
  const controller = new MaskResultController({ ctx: h.ctx });
  const commit = () =>
    controller.commit({
      guard: { layerId: 'source' } as LayerExportGuard,
      image: { height: 8, imageName: 'mask.png', width: 8 },
      rect: { height: 8, width: 8, x: 0, y: 0 },
      target: 'inpaint_mask',
    });
  const leafIds = () => getDocumentLeaves(h.document()).map((layer) => layer.id);
  return { ...h, commit, leafIds };
};

describe('MaskResultController (a guarded result)', () => {
  it('adds and selects the mask layer as one step that undo and redo replay', async () => {
    const h = createHarness();

    await expect(h.commit()).resolves.toEqual({ layerId: 'new-1', status: 'committed' });
    expect(h.document().selectedLayerId).toBe('new-1');
    expect(h.history.entries().past).toEqual(['Create inpaint mask from object']);
    await h.history.undo();
    expect(h.leafIds()).toEqual(['source']);
    expect(h.document().selectedLayerId).toBe('source');
    await h.history.redo();
    expect(h.leafIds()).toContain('new-1');
  });

  it('reports a reducer refusal as stale, recording nothing and holding nothing', async () => {
    const h = createHarness();
    const before = h.document();
    h.state.refuse = () => true;

    await expect(h.commit()).resolves.toEqual({ status: 'stale' });
    expect(h.document()).toBe(before);
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });

  it('rolls back an addition whose postconditions fail and reports it as stale', async () => {
    const h = createHarness();
    // An interleaved edit to the new layer means the document no longer holds the contract this result added.
    h.state.interleave = (mutation) =>
      mutation.type === 'addCanvasLayer'
        ? { id: mutation.layer.id, patch: { name: 'Interleaved' }, type: 'updateCanvasLayer' }
        : null;

    await expect(h.commit()).resolves.toEqual({ status: 'stale' });
    expect(h.leafIds()).toEqual(['source']);
    expect(h.document().selectedLayerId).toBe('source');
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });
});
