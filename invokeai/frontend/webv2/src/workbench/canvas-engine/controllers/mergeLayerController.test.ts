import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentIndex, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
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

describe('MergeLayerController: merge masks', () => {
  const RECTS: Record<string, Rect> = {
    bottom: { height: 10, width: 10, x: 30, y: 30 },
    middle: { height: 10, width: 10, x: 10, y: 10 },
    top: { height: 10, width: 10, x: 0, y: 0 },
  };

  const createMaskHarness = (options: { historyBytes?: number; onExport?: () => void } = {}) => {
    const top = layerContract('top', 'inpaint_mask', {
      denoise: { isEnabled: true, limit: 0.4 },
      mask: { bitmap: null, fill: { color: '#123456', style: 'grid' } },
      noise: { isEnabled: true, level: 0.2 },
      opacity: 0.5,
    } as Partial<CanvasLayerContract>);
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      selectedLayerId: 'middle',
      stacks: stacksFrom([
        groupContract('first', [layerContract('other', 'inpaint_mask'), top]),
        groupContract('second', [layerContract('middle', 'inpaint_mask')]),
        layerContract('bottom', 'inpaint_mask'),
      ]),
      version: 3,
      width: 100,
    };
    const h = createLayerOperationHarness(document, { historyBytes: options.historyBytes });
    for (const [id, rect] of Object.entries({ ...RECTS, other: { height: 4, width: 4, x: 0, y: 0 } })) {
      h.layers.getOrCreateRect(id, rect).stale = false;
    }
    const released: string[] = [];
    const selections: [string | null, readonly string[]][] = [];
    const controller = new MergeLayerController({
      backend: h.backend,
      ctx: h.ctx,
      exportBaked: (layerId) => {
        options.onExport?.();
        const entry = h.layers.get(layerId)!;
        return Promise.resolve({
          guard: { layer: getDocumentLayer(h.document(), layerId) } as never,
          rect: { ...entry.rect },
          release: () => released.push(layerId),
          status: 'ok' as const,
          surface: entry.surface,
        });
      },
      hasExportableContent: (layerId) => (h.layers.get(layerId)?.rect.width ?? 0) > 0,
      isCacheReady: () => true,
      layers: h.layers,
      needsPixelPersistence: () => true,
      publishSelectedLayerIds: (primary, ids) => selections.push([primary, ids]),
    });
    return { ...h, controller, original: h.document(), released, selections };
  };

  const structure = (document: CanvasDocumentContractV3) =>
    [...getDocumentIndex(document).byId.values()].map((entry) => [entry.node.id, entry.parentId, entry.siblingIndex]);

  it('unions masks from different groups into one mask above the topmost, configured like it, as one step', async () => {
    const h = createMaskHarness();

    const result = await h.controller.mergeInpaintMasks(['bottom', 'top', 'middle']);
    expect(result).toEqual({ layerId: 'new-1', status: 'merged' });

    const merged = getDocumentLayer(h.document(), 'new-1');
    expect(merged).toEqual({
      blendMode: 'normal',
      denoise: { isEnabled: true, limit: 0.4 },
      id: 'new-1',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: null, fill: { color: '#123456', style: 'grid' }, offset: { x: 0, y: 0 } },
      name: 'top merged',
      noise: { isEnabled: true, level: 0.2 },
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'inpaint_mask',
    });
    // The union of the three sources' extents, in place of the topmost source; the others are gone.
    expect(h.layers.get('new-1')?.rect).toEqual({ height: 40, width: 40, x: 0, y: 0 });
    expect(getDocumentIndex(h.document()).byId.get('new-1')).toMatchObject({ parentId: 'first', siblingIndex: 1 });
    expect(leafIds(h.document())).toEqual(['other', 'new-1']);
    expect(h.document().selectedLayerId).toBe('new-1');
    expect(h.history.entries().past).toEqual(['Merge masks']);
    expect(h.released.sort()).toEqual(['bottom', 'middle', 'top']);
    expect(h.reservedBytes()).toBe(0);

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(structure(h.document())).toEqual(structure(h.original));
    for (const id of ['top', 'middle', 'bottom']) {
      expect(getDocumentLayer(h.document(), id)).toBe(getDocumentLayer(h.original, id));
      expect(h.layers.get(id)?.rect).toEqual(RECTS[id]);
    }
    expect(h.document().selectedLayerId).toBe('middle');

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(leafIds(h.document())).toEqual(['other', 'new-1']);
  });

  it.each([
    ['one mask', ['top'], 'ineligible'],
    ['a regional or raster leaf', ['top', 'first'], 'ineligible'],
    ['a missing id', ['top', 'gone'], 'missing'],
  ] as const)('refuses %s without reading pixels', async (_, ids, status) => {
    const h = createMaskHarness({ onExport: () => expect.fail('read pixels') });

    await expect(h.controller.mergeInpaintMasks(ids)).resolves.toEqual({ status });
    expect(h.dispatched).toEqual([]);
  });

  it('refuses a merge it could never undo with nothing changed', async () => {
    const h = createMaskHarness({ historyBytes: 1024 });

    await expect(h.controller.mergeInpaintMasks(['top', 'middle'])).resolves.toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(h.original);
    expect(h.dispatched).toEqual([]);
    expect(h.reservedBytes()).toBe(0);
  });

  it('refuses when a source changes while its pixels are read', async () => {
    let edited = false;
    const h = createMaskHarness({
      onExport: () => {
        if (!edited) {
          edited = true;
          h.ctx.applyStep({
            accepted: (document) => getDocumentLayer(document, 'middle')?.name === 'Renamed',
            mutation: { id: 'middle', patch: { name: 'Renamed' }, type: 'updateCanvasLayer' },
          });
        }
      },
    });

    await expect(h.controller.mergeInpaintMasks(['top', 'middle'])).resolves.toEqual({ status: 'not-ready' });
    expect(h.dispatched.map((mutation) => mutation.type)).toEqual(['updateCanvasLayer']);
    expect(h.history.canUndo()).toBe(false);
  });
});
