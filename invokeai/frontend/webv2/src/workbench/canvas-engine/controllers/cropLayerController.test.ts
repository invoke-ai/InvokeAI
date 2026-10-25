import type {
  CanvasAdjustmentsContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { AdjustmentRefusal } from '@workbench/canvas-engine/render/exactAdjustments';
import type { Rect } from '@workbench/canvas-engine/types';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentIndex, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
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

const createHarness = (
  historyBytes?: number,
  options: { readonly nodes?: CanvasNodeContract[]; readonly unreadable?: boolean } = {}
) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: BBOX,
    height: 100,
    selectedLayerId: 'paint',
    stacks: stacksFrom(options.nodes ?? [layer]),
    version: 3,
    width: 100,
  };
  const h = createLayerOperationHarness(document, { historyBytes });
  h.layers.getOrCreateRect('paint', CACHE_RECT);
  const discardPersisted = vi.fn();
  const exported: string[] = [];
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
      exported.push(layerId);
      if (options.unreadable) {
        return Promise.resolve({ status: 'unreadable' as const });
      }
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
    flushPendingPixels: () => Promise.resolve(),
    isGuardCurrent: () => true,
    isSupportedSource: () => true,
  });
  return { ...h, controller, discardPersisted, exported, original: getDocumentLayer(h.document(), 'paint')! };
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

  it('refuses a layer under a locked group, as crop all skips it, without reading it', async () => {
    const h = createHarness(undefined, { nodes: [groupContract('G', [layer], { isLocked: true })] });

    await expect(h.controller.crop('paint')).resolves.toEqual({ status: 'locked' });
    expect(h.exported).toEqual([]);
    expect(h.dispatched).toEqual([]);
  });

  it('says a layer that could not be read is unreadable, not still loading', async () => {
    const h = createHarness(undefined, { unreadable: true });

    await expect(h.controller.crop('paint')).resolves.toEqual({ status: 'unreadable' });
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses to empty a layer that lies wholly outside the frame', async () => {
    const h = createHarness();
    h.layers.delete('paint');
    h.layers.getOrCreateRect('paint', { height: 10, width: 10, x: 60, y: 60 });

    await expect(h.controller.crop('paint')).resolves.toEqual({ status: 'outside-frame' });
    expect(h.dispatched).toEqual([]);
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

const FRAME: Rect = { height: 20, width: 20, x: 0, y: 0 };

/** Each leaf's cache rect, which the fake export also reports as its baked pixels. */
const CACHE_RECTS: Record<string, Rect> = {
  control: { height: 10, width: 30, x: 5, y: 5 },
  inside: { height: 5, width: 5, x: 2, y: 2 },
  locked: { height: 50, width: 50, x: -10, y: -10 },
  lockedByGroup: { height: 50, width: 50, x: -10, y: -10 },
  lockedInside: { height: 5, width: 5, x: 2, y: 2 },
  mask: { height: 10, width: 10, x: 60, y: 60 },
  raster: { height: 50, width: 50, x: -10, y: -10 },
  region: { height: 10, width: 10, x: -5, y: 0 },
};

const bytes = (rect: Rect): number => rect.width * rect.height * 4;

/** A locked paint leaf whose persisted pixels cover its cache rect. */
const lockedPaint = (id: string, locked = true) =>
  layerContract(id, 'raster', {
    isLocked: locked,
    source: {
      bitmap: { height: CACHE_RECTS[id]!.height, imageName: `${id}.png`, width: CACHE_RECTS[id]!.width },
      offset: { x: CACHE_RECTS[id]!.x, y: CACHE_RECTS[id]!.y },
      type: 'paint',
    },
  } as Partial<CanvasLayerContract>);

const CLARITY: CanvasAdjustmentsContract = [{ amount: 90, id: 'clarity', isEnabled: true, size: 8, type: 'clarity' }];

const createCropAllHarness = (
  options: {
    /** Raster leaves carrying a Clarity stack. */
    adjusted?: readonly string[];
    historyBytes?: number;
    onExport?: (layerId: string) => void;
    /** Refuses the owner's bake; its raw export still succeeds. */
    refusal?: AdjustmentRefusal;
    /** Leaves added on top whose pixels cannot be read. */
    unreadable?: readonly CanvasLayerContract[];
    /** Settles pending paint uploads; resolves at once by default. */
    flushPendingPixels?: () => Promise<void>;
  } = {}
) => {
  const raster = (id: string) =>
    layerContract(
      id,
      'raster',
      options.adjusted?.includes(id) ? ({ adjustments: CLARITY } as Partial<CanvasLayerContract>) : {}
    );
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: FRAME,
    height: 100,
    selectedLayerId: 'raster',
    stacks: stacksFrom([
      ...(options.unreadable ?? []),
      raster('inside'),
      raster('raster'),
      lockedPaint('locked'),
      lockedPaint('lockedInside'),
      groupContract('lockedGroup', [lockedPaint('lockedByGroup', false)], { isLocked: true }),
      layerContract('control', 'control', { isEnabled: false }),
      groupContract('regions', [layerContract('region', 'regional_guidance')]),
      layerContract('mask', 'inpaint_mask', { isHidden: true } as Partial<CanvasLayerContract>),
    ]),
    version: 3,
    width: 100,
  };
  const h = createLayerOperationHarness(document, { historyBytes: options.historyBytes });
  for (const [id, rect] of Object.entries(CACHE_RECTS)) {
    h.layers.getOrCreateRect(id, rect);
  }
  // Raw reads of adjusted leaves are recorded as `<id>:raw`.
  const exported: string[] = [];
  const released: string[] = [];
  // Reserved raster bytes when each export is read; every export holds its own reservation until released.
  const reservedAtExport: number[] = [];
  const guard = { current: true };
  const discardPersisted = vi.fn();
  const controller = new CropLayerController({
    backend: h.backend,
    captureCache: (target, _document, admit = () => true) => {
      const entry = h.layers.get(target.id)!;
      if (!admit(entry.rect)) {
        return 'over-budget';
      }
      const pixels = h.backend.createSurface(entry.rect.width, entry.rect.height);
      pixels.ctx.drawImage(entry.surface.canvas, 0, 0);
      return { pixels, rect: { ...entry.rect } };
    },
    ctx: h.ctx,
    discardPersisted,
    exportBaked: (layerId, exportOptions) => {
      const read = exportOptions?.applyAdjustments === false ? `${layerId}:raw` : layerId;
      exported.push(read);
      options.onExport?.(layerId);
      if (options.unreadable?.some((leaf) => leaf.id === layerId)) {
        return Promise.resolve({ status: 'unreadable' as const });
      }
      if (options.refusal?.ownerId === layerId && read === layerId) {
        return Promise.resolve(options.refusal);
      }
      const entry = h.layers.get(layerId)!;
      const reservation = h.ctx.reserveRaster(bytes(entry.rect));
      reservedAtExport.push(h.reservedBytes());
      return Promise.resolve({
        guard: { layer: getDocumentLayer(h.document(), layerId) } as never,
        rect: { ...entry.rect },
        release: () => {
          reservation?.release();
          released.push(read);
        },
        status: 'ok' as const,
        surface: entry.surface,
      });
    },
    flushPendingPixels: options.flushPendingPixels ?? (() => Promise.resolve()),
    isGuardCurrent: () => guard.current,
    isSupportedSource: () => true,
  });
  return { ...h, controller, discardPersisted, exported, guard, original: h.document(), released, reservedAtExport };
};

describe('CropLayerController: crop all layers', () => {
  it.each(['too-large', 'over-budget', 'error'] as const)(
    'refuses the whole crop on a %s adjustment of a leaf it must crop, releasing earlier copies',
    async (reason) => {
      const refusal: AdjustmentRefusal = {
        ownerId: 'raster',
        ownerName: 'Raster',
        reason,
        status: 'adjustment-refused',
      };
      const h = createCropAllHarness({ adjusted: ['raster'], refusal });

      await expect(h.controller.cropAll()).resolves.toEqual(refusal);
      // The raw read shows the leaf reaches past the frame, so its stack must run.
      expect(h.exported).toEqual(['mask', 'region', 'control', 'inside', 'raster:raw', 'raster']);
      expect(h.released).toEqual(['mask', 'region', 'control', 'inside', 'raster:raw']);
      expect(h.document()).toBe(h.original);
      expect(h.dispatched).toEqual([]);
      expect(h.history.canUndo()).toBe(false);
      expect(h.reservedBytes()).toBe(0);
    }
  );

  it('never runs the stack of an adjusted leaf already inside the frame', async () => {
    const refusal: AdjustmentRefusal = {
      ownerId: 'inside',
      ownerName: 'Inside',
      reason: 'too-large',
      status: 'adjustment-refused',
    };
    const h = createCropAllHarness({ adjusted: ['inside'], refusal });

    await expect(h.controller.cropAll()).resolves.toEqual({
      cropped: 4,
      skippedLocked: 2,
      skippedUnreadable: 0,
      status: 'cropped',
    });
    expect(h.exported).toEqual(['mask', 'region', 'control', 'inside:raw', 'raster']);
    expect(getDocumentLayer(h.document(), 'inside')).toBe(getDocumentLayer(h.original, 'inside'));
    expect(h.layers.get('raster')?.rect).toEqual(FRAME);
    expect(h.reservedBytes()).toBe(0);
  });

  it('bakes an adjusted leaf it must crop, after releasing its raw read', async () => {
    const h = createCropAllHarness({ adjusted: ['raster'] });

    await expect(h.controller.cropAll()).resolves.toEqual({
      cropped: 4,
      skippedLocked: 2,
      skippedUnreadable: 0,
      status: 'cropped',
    });
    expect(h.exported).toEqual(['mask', 'region', 'control', 'inside', 'raster:raw', 'raster']);
    // The raw read is released before the bake reserves, so the peak stays one export.
    const croppedRegion = bytes({ height: 10, width: 5, x: 0, y: 0 });
    const croppedControl = bytes({ height: 10, width: 15, x: 0, y: 0 });
    expect(h.reservedAtExport.at(-1)).toBe(croppedRegion + croppedControl + bytes(CACHE_RECTS.raster!));
    expect(getDocumentLayer(h.document(), 'raster')).not.toHaveProperty('adjustments');
    expect(h.layers.get('raster')?.rect).toEqual(FRAME);
    expect(h.reservedBytes()).toBe(0);
  });

  it('crops every unlocked leaf that reaches past the frame in one entry, and replays it as one', async () => {
    const h = createCropAllHarness();

    await expect(h.controller.cropAll()).resolves.toEqual({
      cropped: 4,
      skippedLocked: 2,
      skippedUnreadable: 0,
      status: 'cropped',
    });

    // Hidden, disabled and grouped leaves crop too; locked ones are never read, and only those reaching past the
    // frame count as skipped.
    expect(h.exported).toEqual(['mask', 'region', 'control', 'inside', 'raster']);
    // Each export is released before the next is read, keeping only the cropped pixels: the peak is one export.
    const croppedRegion = bytes({ height: 10, width: 5, x: 0, y: 0 });
    const croppedControl = bytes({ height: 10, width: 15, x: 0, y: 0 });
    expect(h.reservedAtExport).toEqual([
      bytes(CACHE_RECTS.mask!),
      bytes(CACHE_RECTS.region!),
      croppedRegion + bytes(CACHE_RECTS.control!),
      croppedRegion + croppedControl + bytes(CACHE_RECTS.inside!),
      croppedRegion + croppedControl + bytes(CACHE_RECTS.raster!),
    ]);
    expect(h.released.sort()).toEqual(h.exported.sort());
    expect(h.dispatched.map((mutation) => mutation.type)).toEqual(['replaceCanvasLayers']);
    expect(h.history.entries().past).toEqual(['Crop all layers to frame']);
    // Before and after pixels of every cropped leaf, admitted once for the entry.
    const after =
      bytes(FRAME) + bytes({ height: 10, width: 15, x: 0, y: 0 }) + bytes({ height: 10, width: 5, x: 0, y: 0 });
    const before = ['raster', 'control', 'mask', 'region'].reduce((sum, id) => sum + bytes(CACHE_RECTS[id]!), 0);
    expect(h.history.byteSize()).toBe(before + after + HISTORY_ENTRY_OVERHEAD_BYTES);
    expect(h.reservedBytes()).toBe(0);

    const cropped = h.document();
    expect(h.layers.get('raster')?.rect).toEqual(FRAME);
    expect(h.layers.get('control')?.rect).toEqual({ height: 10, width: 15, x: 5, y: 5 });
    expect(h.layers.get('region')?.rect).toEqual({ height: 10, width: 5, x: 0, y: 0 });
    // The mask outside the frame stays, emptied.
    expect(h.layers.get('mask')?.rect).toMatchObject({ height: 0, width: 0 });
    expect(getDocumentLayer(cropped, 'mask')).toMatchObject({ isHidden: true, mask: { bitmap: null } });
    expect(getDocumentLayer(cropped, 'control')).toMatchObject({
      isEnabled: false,
      source: { offset: { x: 5, y: 5 } },
    });
    for (const id of ['inside', 'locked', 'lockedInside', 'lockedByGroup']) {
      expect(getDocumentLayer(cropped, id)).toBe(getDocumentLayer(h.original, id));
      expect(h.layers.get(id)?.rect).toEqual(CACHE_RECTS[id]);
    }
    // Structure is untouched: same ids, parents and order.
    const structure = (document: CanvasDocumentContractV3) =>
      [...getDocumentIndex(document).byId.values()].map((entry) => [entry.node.id, entry.parentId, entry.siblingIndex]);
    expect(structure(cropped)).toEqual(structure(h.original));
    expect(h.discardPersisted.mock.calls.map(([id]) => id).sort()).toEqual(['control', 'mask', 'raster', 'region']);

    await expect(h.history.undo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLeaves(h.document())).toEqual(getDocumentLeaves(h.original));
    for (const id of ['raster', 'control', 'mask', 'region']) {
      expect(h.layers.get(id)?.rect).toEqual(CACHE_RECTS[id]);
    }

    await expect(h.history.redo()).resolves.toEqual({ status: 'applied' });
    expect(getDocumentLeaves(h.document())).toEqual(getDocumentLeaves(cropped));
    expect(h.layers.get('raster')?.rect).toEqual(FRAME);
  });

  it('skips leaves whose pixels cannot be read and crops the rest, counting those that reach past the frame', async () => {
    /** An image layer whose decode failed; its contract still says where its content lies. */
    const unreadableImage = (id: string, x: number) =>
      layerContract(id, 'raster', {
        source: { image: { height: 10, imageName: `${id}.png`, width: 10 }, type: 'image' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x, y: 0 },
      } as Partial<CanvasLayerContract>);
    const h = createCropAllHarness({ unreadable: [unreadableImage('gone', 15), unreadableImage('goneInside', 5)] });

    await expect(h.controller.cropAll()).resolves.toEqual({
      cropped: 4,
      skippedLocked: 2,
      skippedUnreadable: 1,
      status: 'cropped',
    });
    expect(h.exported).toEqual(['mask', 'region', 'control', 'gone', 'goneInside', 'inside', 'raster']);
    expect(getDocumentLayer(h.document(), 'gone')).toBe(getDocumentLayer(h.original, 'gone'));
    expect(getDocumentLayer(h.document(), 'raster')).not.toBe(getDocumentLayer(h.original, 'raster'));
    expect(h.history.entries().past).toEqual(['Crop all layers to frame']);
  });

  it('settles pending paint uploads before it reads any leaf', async () => {
    let settle!: () => void;
    const h = createCropAllHarness({
      flushPendingPixels: () =>
        new Promise<void>((resolve) => {
          settle = resolve;
        }),
    });

    const crop = h.controller.cropAll();
    await Promise.resolve();
    expect(h.exported).toEqual([]);
    settle();

    await expect(crop).resolves.toMatchObject({ cropped: 4, status: 'cropped' });
  });

  it('reports nothing when every unlocked leaf already fits the frame', async () => {
    const h = createCropAllHarness();
    for (const id of ['raster', 'control', 'mask', 'region']) {
      h.layers.delete(id);
      h.layers.getOrCreateRect(id, CACHE_RECTS.inside!);
    }

    await expect(h.controller.cropAll()).resolves.toEqual({
      skippedLocked: 2,
      skippedUnreadable: 0,
      status: 'nothing',
    });
    expect(h.dispatched).toEqual([]);
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses a set it could never undo with nothing changed and every lease released', async () => {
    const h = createCropAllHarness({ historyBytes: 8 * 1024 });

    await expect(h.controller.cropAll()).resolves.toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(h.original);
    expect(h.dispatched).toEqual([]);
    expect(h.layers.get('raster')?.rect).toEqual(CACHE_RECTS.raster);
    expect(h.history.byteSize()).toBe(0);
    expect(h.reservedBytes()).toBe(0);
    expect(h.released.sort()).toEqual(h.exported.sort());
  });

  it('releases every reservation when copying the cropped pixels throws', async () => {
    const h = createCropAllHarness();
    const createSurface = h.backend.createSurface.bind(h.backend);
    // The emptied mask's zero-sized copy succeeds; the first real crop, after its reservation, throws.
    vi.spyOn(h.backend, 'createSurface').mockImplementation((width: number, height: number) => {
      if (width > 0) {
        throw new Error('surface allocation failed');
      }
      return createSurface(width, height);
    });

    await expect(h.controller.cropAll()).resolves.toEqual({ status: 'failed' });
    expect(h.reservedBytes()).toBe(0);
    expect(h.document()).toBe(h.original);
    expect(h.released).toEqual(h.exported);
  });

  it('refuses as over-budget when the cropped pixels cannot be reserved', async () => {
    const h = createCropAllHarness();
    h.state.rasterBytes = 1024;

    await expect(h.controller.cropAll()).resolves.toEqual({ status: 'over-budget' });
    expect(h.document()).toBe(h.original);
    expect(h.history.canUndo()).toBe(false);
  });

  it('is stale when the document changes while the pixels are read, and changes nothing', async () => {
    let edited = false;
    const h = createCropAllHarness({
      onExport: (layerId) => {
        if (layerId === 'control' && !edited) {
          edited = true;
          h.ctx.applyStep({
            accepted: (document) => document?.bbox.width === 30,
            mutation: { bbox: { ...FRAME, width: 30 }, type: 'setCanvasBbox' },
          });
        }
      },
    });
    await expect(h.controller.cropAll()).resolves.toEqual({ status: 'stale' });
    expect(h.dispatched.map((mutation) => mutation.type)).toEqual(['setCanvasBbox']);
    expect(getDocumentLeaves(h.document())).toEqual(getDocumentLeaves(h.original));
    expect(h.history.canUndo()).toBe(false);
    expect(h.released.sort()).toEqual(h.exported.sort());
  });

  it('is stale when a leaf pixel guard went out of date', async () => {
    const h = createCropAllHarness();
    h.guard.current = false;

    await expect(h.controller.cropAll()).resolves.toEqual({ status: 'stale' });
    expect(h.dispatched).toEqual([]);
  });

  it('leaves the entry on the undo stack when its replay is refused', async () => {
    const h = createCropAllHarness();
    await h.controller.cropAll();
    h.state.refuse = () => true;

    expect((await h.history.undo()).status).toBe('failed');
    expect(h.layers.get('raster')?.rect).toEqual(FRAME);
    expect(h.history.canUndo()).toBe(true);
  });
});
