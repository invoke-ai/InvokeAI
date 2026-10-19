import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { describe, expect, it } from 'vitest';

import { createLayerOperationHarness } from './layerOperationHarness.testing';
import { MergeLayerController } from './mergeLayerController';

const paintLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const UPPER_RECT: Rect = { height: 10, width: 10, x: 0, y: 0 };
const LOWER_RECT: Rect = { height: 10, width: 10, x: 5, y: 5 };
const MERGED_RECT: Rect = { height: 15, width: 15, x: 0, y: 0 };

const createHarness = (historyBytes?: number) => {
  const upper = paintLayer('upper');
  const lower = paintLayer('lower');
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: { height: 100, width: 100, x: 0, y: 0 },
    height: 100,
    selectedLayerId: 'upper',
    stacks: stacksFrom([upper, lower]),
    version: 3,
    width: 100,
  };
  const h = createLayerOperationHarness(document, { historyBytes });
  h.layers.getOrCreateRect('upper', UPPER_RECT);
  h.layers.getOrCreateRect('lower', LOWER_RECT);
  const controller = new MergeLayerController({
    backend: h.backend,
    ctx: h.ctx,
    exportBaked: () => Promise.reject(new Error('merge down reads live caches')),
    hasExportableContent: (layerId) => (h.layers.get(layerId)?.rect.width ?? 0) > 0,
    isCacheReady: () => true,
    layers: h.layers,
    needsPixelPersistence: () => true,
    publishSelectedLayerIds: () => undefined,
  });
  // The reducer's own contracts, which merge and undo must restore by identity.
  const originals = {
    lower: getDocumentLayer(h.document(), 'lower')!,
    upper: getDocumentLayer(h.document(), 'upper')!,
  };
  return { ...h, controller, originals };
};

const leafIds = (document: CanvasDocumentContractV3 | null): string[] =>
  getDocumentLeaves(document).map((layer) => layer.id);

describe('MergeLayerController: merge down', () => {
  it('bakes the upper layer into the lower one as one undo step', () => {
    const h = createHarness();

    expect(h.controller.mergeDown('upper')).toBe('merged');
    expect(leafIds(h.document())).toEqual(['lower']);
    expect(getDocumentLayer(h.document(), 'lower')).toMatchObject({
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
    });
    expect(h.layers.get('lower')?.rect).toEqual(MERGED_RECT);
    expect(h.history.entries().past).toEqual(['Merge down']);
  });

  it('undo restores both layers, their pixels and the selection; redo merges again', async () => {
    const h = createHarness();
    h.controller.mergeDown('upper');

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(leafIds(h.document())).toEqual(['upper', 'lower']);
    expect(getDocumentLayer(h.document(), 'upper')).toBe(h.originals.upper);
    expect(getDocumentLayer(h.document(), 'lower')).toBe(h.originals.lower);
    expect(h.document().selectedLayerId).toBe('upper');
    expect(h.layers.get('upper')?.rect).toEqual(UPPER_RECT);
    expect(h.layers.get('lower')?.rect).toEqual(LOWER_RECT);

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(leafIds(h.document())).toEqual(['lower']);
    expect(h.layers.get('lower')?.rect).toEqual(MERGED_RECT);
  });

  it('puts the merge back and keeps the step when the upper layer cannot be reinserted', async () => {
    const h = createHarness();
    h.controller.mergeDown('upper');
    const merged = h.document();
    h.state.refuse = (mutation) => mutation.type === 'applyCanvasLayerStackMutation';

    expect((await h.history.undo()).status).toBe('failed');
    expect(leafIds(h.document())).toEqual(['lower']);
    expect(getDocumentLayer(h.document(), 'lower')).toBe(getDocumentLayer(merged, 'lower'));
    expect(h.layers.get('lower')?.rect).toEqual(MERGED_RECT);
    expect(h.history.canUndo()).toBe(true);
    expect(h.history.canRedo()).toBe(false);
  });

  it('refuses a merge it could never undo before anything changes', () => {
    const h = createHarness(1024);
    const before = h.document();

    expect(h.controller.mergeDown('upper')).toBe('over-budget');
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
    expect(h.layers.get('upper')?.rect).toEqual(UPPER_RECT);
  });

  it('refuses when its pixels cannot be reserved', () => {
    const h = createHarness();
    h.state.rasterBytes = 1024;

    expect(h.controller.mergeDown('upper')).toBe('over-budget');
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses the bottom-most layer and a busy canvas', () => {
    const h = createHarness();
    expect(h.controller.mergeDown('lower')).toBe('unsupported');
    expect(h.controller.mergeDown('missing')).toBe('missing');
    h.state.gestureActive = true;
    expect(h.controller.mergeDown('upper')).toBe('busy');
    expect(h.dispatched).toEqual([]);
  });
});
