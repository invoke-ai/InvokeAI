import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
} from '@workbench/canvas-engine/contracts';
import type { AdjustmentRefusal, SpatialRunner } from '@workbench/canvas-engine/render/exactAdjustments';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentIndex, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

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

const createHarness = (
  historyBytes?: number,
  {
    lowerAdjusted = false,
    lowerAdjustments,
    displayed = null,
    runSpatial,
  }: {
    lowerAdjusted?: boolean;
    lowerAdjustments?: CanvasAdjustmentsContract;
    displayed?: RasterSurface | null;
    runSpatial?: SpatialRunner;
  } = {}
) => {
  const upper = paintLayer('upper');
  const lower: CanvasLayerContract = {
    ...paintLayer('lower'),
    ...(lowerAdjusted ? { adjustments: [{ id: 'invert', isEnabled: true, type: 'invert' }] } : {}),
    ...(lowerAdjustments ? { adjustments: lowerAdjustments } : {}),
  };
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
    peekAdjustedSurface: () => displayed,
    publishSelectedLayerIds: () => undefined,
    runSpatial,
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
  it('bakes the upper layer into the lower one as one undo step', async () => {
    const h = createHarness();

    expect(await h.controller.mergeDown('upper')).toBe('merged');
    expect(leafIds(h.document())).toEqual(['lower']);
    expect(getDocumentLayer(h.document(), 'lower')).toMatchObject({
      source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
    });
    expect(h.layers.get('lower')?.rect).toEqual(MERGED_RECT);
    expect(h.history.entries().past).toEqual(['Merge down']);
  });

  it('undo restores both layers, their pixels and the selection; redo merges again', async () => {
    const h = createHarness();
    await h.controller.mergeDown('upper');

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
    await h.controller.mergeDown('upper');
    const merged = h.document();
    h.state.refuse = (mutation) => mutation.type === 'applyCanvasLayerStackMutation';

    expect((await h.history.undo()).status).toBe('failed');
    expect(leafIds(h.document())).toEqual(['lower']);
    expect(getDocumentLayer(h.document(), 'lower')).toBe(getDocumentLayer(merged, 'lower'));
    expect(h.layers.get('lower')?.rect).toEqual(MERGED_RECT);
    expect(h.history.canUndo()).toBe(true);
    expect(h.history.canRedo()).toBe(false);
  });

  it('refuses a merge it could never undo before anything changes', async () => {
    const h = createHarness(1024);
    const before = h.document();

    expect(await h.controller.mergeDown('upper')).toBe('over-budget');
    expect(h.document()).toBe(before);
    expect(h.dispatched).toEqual([]);
    expect(h.layers.get('upper')?.rect).toEqual(UPPER_RECT);
  });

  it('refuses when its pixels cannot be reserved', async () => {
    const h = createHarness();
    h.state.rasterBytes = 1024;

    expect(await h.controller.mergeDown('upper')).toBe('over-budget');
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('reserves a layer-local adjusted copy only when the display has no current one', async () => {
    // Raw undo snapshots plus the merged pixels and their prepared replacement; a copy adds a surface and ImageData.
    const baseBytes = (100 + 100) * 4 + 225 * 4 * 2;
    const copying = createHarness(undefined, { lowerAdjusted: true });
    copying.state.rasterBytes = baseBytes;
    expect(await copying.controller.mergeDown('upper')).toBe('over-budget');
    copying.state.rasterBytes = baseBytes + 100 * 4 * 2;
    expect(await copying.controller.mergeDown('upper')).toBe('merged');

    const displayed = createHarness(undefined, {
      displayed: createTestStubRasterBackend().createSurface(10, 10),
      lowerAdjusted: true,
    });
    displayed.state.rasterBytes = baseBytes;
    expect(await displayed.controller.mergeDown('upper')).toBe('merged');
  });

  it('refuses the bottom-most layer and a busy canvas', async () => {
    const h = createHarness();
    expect(await h.controller.mergeDown('lower')).toBe('unsupported');
    expect(await h.controller.mergeDown('missing')).toBe('missing');
    h.state.gestureActive = true;
    expect(await h.controller.mergeDown('upper')).toBe('busy');
    expect(h.dispatched).toEqual([]);
  });

  describe('with a spatial stack', () => {
    const CLARITY: CanvasAdjustmentsContract = [
      { amount: 60, id: 'clarity', isEnabled: true, size: 20, type: 'clarity' },
    ];
    const passThrough = () =>
      vi.fn<SpatialRunner>((request) => Promise.resolve({ pixels: request.pixels, status: 'ok' }));

    it('evaluates the layer-local stack in the spatial service before merging, and drops it', async () => {
      const runSpatial = passThrough();
      const h = createHarness(undefined, { lowerAdjustments: CLARITY, runSpatial });

      expect(await h.controller.mergeDown('upper')).toBe('merged');

      expect(runSpatial).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ height: LOWER_RECT.height, priority: 'now', width: LOWER_RECT.width })
      );
      expect(getDocumentLayer(h.document(), 'lower')).not.toHaveProperty('adjustments');
    });

    it("reuses the display's exact copy without a job", async () => {
      const runSpatial = passThrough();
      const h = createHarness(undefined, {
        displayed: createTestStubRasterBackend().createSurface(10, 10),
        lowerAdjustments: CLARITY,
        runSpatial,
      });

      expect(await h.controller.mergeDown('upper')).toBe('merged');
      expect(runSpatial).not.toHaveBeenCalled();
    });

    it.each(['too-large', 'over-budget', 'error'] as const)(
      'refuses %s with the layer name before anything changes',
      async (reason) => {
        const h = createHarness(undefined, {
          lowerAdjustments: CLARITY,
          runSpatial: () => Promise.resolve({ status: reason }),
        });

        expect(await h.controller.mergeDown('upper')).toEqual({
          ownerId: 'lower',
          ownerName: 'lower',
          reason,
          status: 'adjustment-refused',
        });
        expect(h.dispatched).toEqual([]);
        expect(h.history.canUndo()).toBe(false);
        expect(h.reservedBytes()).toBe(0);
      }
    );

    it('refuses without a spatial runner rather than merge without the spatial entries', async () => {
      const h = createHarness(undefined, { lowerAdjustments: CLARITY });

      expect(await h.controller.mergeDown('upper')).toMatchObject({ reason: 'error', status: 'adjustment-refused' });
      expect(h.dispatched).toEqual([]);
    });

    it('lands nothing when a layer changed while the job ran', async () => {
      const h = createHarness(undefined, {
        lowerAdjustments: CLARITY,
        runSpatial: (request) => {
          h.layers.publishPixels('upper');
          return Promise.resolve({ pixels: request.pixels, status: 'ok' });
        },
      });

      expect(await h.controller.mergeDown('upper')).toBe('not-ready');
      expect(h.dispatched).toEqual([]);
      expect(h.history.canUndo()).toBe(false);
    });
  });
});

describe('MergeLayerController: merge visible with a spatial group stack', () => {
  const scene = (runSpatial?: SpatialRunner, exportRefusal?: AdjustmentRefusal) => {
    const first = { ...paintLayer('first'), name: 'First' };
    const second = { ...paintLayer('second'), name: 'Second' };
    const document: CanvasDocumentContractV3 = {
      background: 'transparent',
      bbox: { height: 100, width: 100, x: 0, y: 0 },
      height: 100,
      selectedLayerId: null,
      stacks: stacksFrom([
        groupContract('group', [first, second], {
          adjustments: [{ amount: 30, id: 'dehaze', isEnabled: true, type: 'dehaze' }],
          name: 'Haze',
        }),
      ]),
      version: 3,
      width: 100,
    };
    const h = createLayerOperationHarness(document);
    const rects: Record<string, Rect> = { first: UPPER_RECT, second: LOWER_RECT };
    const controller = new MergeLayerController({
      backend: h.backend,
      ctx: h.ctx,
      exportBaked: (layerId) =>
        Promise.resolve(
          exportRefusal?.ownerId === layerId
            ? exportRefusal
            : {
                guard: {
                  cacheVersion: 1,
                  documentGeneration: 1,
                  layer: getDocumentLayer(h.document(), layerId)!,
                  layerId,
                  projectId: 'p',
                },
                rect: rects[layerId]!,
                release: () => undefined,
                status: 'ok',
                surface: h.backend.createSurface(rects[layerId]!.width, rects[layerId]!.height),
              }
        ),
      hasExportableContent: () => true,
      isCacheReady: () => true,
      layers: h.layers,
      needsPixelPersistence: () => false,
      peekAdjustedSurface: () => null,
      publishSelectedLayerIds: () => undefined,
      runSpatial,
    });
    return { ...h, controller };
  };

  it('preserves a selected layer export refusal with its owner and reason, merging nothing', async () => {
    const refusal: AdjustmentRefusal = {
      ownerId: 'second',
      ownerName: 'Second',
      reason: 'too-large',
      status: 'adjustment-refused',
    };
    const h = scene(undefined, refusal);
    const original = h.document();

    expect(await h.controller.mergeSelected(['first', 'second'])).toEqual(refusal);
    expect(h.document()).toBe(original);
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
    expect(h.reservedBytes()).toBe(0);
  });

  it("evaluates the group's spatial stack on the merged members", async () => {
    const runSpatial = vi.fn<SpatialRunner>((request) => Promise.resolve({ pixels: request.pixels, status: 'ok' }));
    const h = scene(runSpatial);

    expect(await h.controller.mergeVisible()).toBe('merged');
    // The buffer holds every member: the union of their baked rects.
    expect(runSpatial).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        height: MERGED_RECT.height,
        steps: [{ amount: 30, kind: 'dehaze' }],
        width: MERGED_RECT.width,
      })
    );
  });

  it('refuses with the group name when its stack cannot run', async () => {
    const h = scene(() => Promise.resolve({ status: 'too-large' }));

    expect(await h.controller.mergeVisible()).toEqual({
      ownerId: 'group',
      ownerName: 'Haze',
      reason: 'too-large',
      status: 'adjustment-refused',
    });
    expect(h.history.canUndo()).toBe(false);
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
      peekAdjustedSurface: () => null,
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
