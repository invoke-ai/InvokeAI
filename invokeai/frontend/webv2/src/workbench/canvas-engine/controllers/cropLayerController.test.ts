import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
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

const createCropAllHarness = (options: { historyBytes?: number; onExport?: (layerId: string) => void } = {}) => {
  const document: CanvasDocumentContractV3 = {
    background: 'transparent',
    bbox: FRAME,
    height: 100,
    selectedLayerId: 'raster',
    stacks: stacksFrom([
      layerContract('inside'),
      layerContract('raster'),
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
    exportBaked: (layerId) => {
      exported.push(layerId);
      options.onExport?.(layerId);
      const entry = h.layers.get(layerId)!;
      const reservation = h.ctx.reserveRaster(bytes(entry.rect));
      reservedAtExport.push(h.reservedBytes());
      return Promise.resolve({
        guard: { layer: getDocumentLayer(h.document(), layerId) } as never,
        rect: { ...entry.rect },
        release: () => {
          reservation?.release();
          released.push(layerId);
        },
        status: 'ok' as const,
        surface: entry.surface,
      });
    },
    isGuardCurrent: () => guard.current,
    isSupportedSource: () => true,
  });
  return { ...h, controller, discardPersisted, exported, guard, original: h.document(), released, reservedAtExport };
};

describe('CropLayerController: crop all layers', () => {
  it('crops every unlocked leaf that reaches past the frame in one entry, and replays it as one', async () => {
    const h = createCropAllHarness();

    await expect(h.controller.cropAll()).resolves.toEqual({ cropped: 4, skippedLocked: 2, status: 'cropped' });

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

  it('reports nothing when every unlocked leaf already fits the frame', async () => {
    const h = createCropAllHarness();
    for (const id of ['raster', 'control', 'mask', 'region']) {
      h.layers.delete(id);
      h.layers.getOrCreateRect(id, CACHE_RECTS.inside!);
    }

    await expect(h.controller.cropAll()).resolves.toEqual({ skippedLocked: 2, status: 'nothing' });
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
