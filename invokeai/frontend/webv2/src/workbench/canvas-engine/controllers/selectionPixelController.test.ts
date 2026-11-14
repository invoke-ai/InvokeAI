import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PixelEditTransaction } from '@workbench/canvas-engine/tools/tool';
import type { Rect } from '@workbench/canvas-engine/types';

import { createCanvasMutationContext } from '@workbench/canvas-engine/controllers/mutationContext';
import {
  documentFrom,
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createHistory } from '@workbench/canvas-engine/history/history';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { SelectionPixelController } from './selectionPixelController';

const paintDocument = (
  source: Record<string, unknown> = { type: 'paint' },
  layer: Record<string, unknown> = {}
): CanvasDocumentContractV3 =>
  ({
    stacks: stacksFrom([
      {
        id: 'paint',
        isEnabled: true,
        isLocked: false,
        name: 'paint',
        source,
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
        type: 'raster',
        ...layer,
      } as unknown as CanvasLayerContract,
    ]),
    selectedLayerId: 'paint',
  }) as CanvasDocumentContractV3;

const createHarness = (
  options: {
    document?: CanvasDocumentContractV3;
    cacheRect?: Rect | null;
    selectionRect?: Rect;
    byteBudget?: number;
    mirrored?: boolean;
    cacheReady?: boolean;
    beginPixelEdit?: (layerId: string) => PixelEditTransaction | null;
    transformSessionLayerId?: string;
  } = {}
) => {
  const stub = createTestStubRasterBackend();
  const surfaces: StubRasterSurface[] = [];
  const backend = {
    ...stub,
    createSurface: (width: number, height: number) => {
      const surface = stub.createSurface(width, height);
      surfaces.push(surface);
      return surface;
    },
  };
  const layers = createLayerCacheStore(backend);
  const cacheRect = options.cacheRect === undefined ? { height: 2, width: 2, x: 0, y: 0 } : options.cacheRect;
  if (cacheRect) {
    const { ctx } = layers.getOrCreateRect('paint', cacheRect).surface;
    // Stub readbacks are constant; number them so a fill's before and after differ.
    const read = ctx.getImageData.bind(ctx);
    let reads = 0;
    Object.defineProperty(ctx, 'getImageData', {
      value: (...args: Parameters<typeof read>) => {
        const pixels = read(...args);
        pixels.data[0] = reads++;
        return pixels;
      },
    });
  }
  const selectionRect = options.selectionRect ?? { height: 2, width: 2, x: 0, y: 0 };
  const mask = backend.createSurface(selectionRect.width, selectionRect.height);
  const selection = {
    bounds: () => selectionRect,
    mask: () => ({ rect: selectionRect, surface: mask }),
  } as unknown as SelectionState;
  const document = options.document ?? paintDocument();
  // A mirror that never follows the reducer fails every publication's postconditions.
  const reducerDocument = options.mirrored === false ? { ...document } : document;
  const history = createHistory({ byteBudget: options.byteBudget });
  const ctx = createCanvasMutationContext({
    commitEdit: vi.fn(),
    createLayerId: () => 'unused',
    dispatch: () => true,
    editOwner: Symbol('owner'),
    editingLocked: { get: () => false, subscribe: () => () => undefined },
    getDocument: () => document,
    getReducerDocument: () => reducerDocument,
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
  const calls = {
    deleteDerived: vi.fn(),
    markDirty: vi.fn(),
    notifyPainted: vi.fn(),
    reportRefusal: vi.fn(),
    requestRasterization: vi.fn(),
  };
  const beginPixelEdit = vi.fn(options.beginPixelEdit ?? ((_layerId: string): PixelEditTransaction | null => null));
  const applyImagePatch = vi.fn((_layerId: string, _rect: Rect, _pixels: ImageData) => Promise.resolve());
  const controller = new SelectionPixelController({
    applyImagePatch,
    backend,
    beginPixelEdit,
    canEdit: () => true,
    ctx,
    deleteDerived: calls.deleteDerived,
    getDocument: () => document,
    getFillColor: () => '#f00',
    getTransformSessionLayerId: () => options.transformSessionLayerId ?? null,
    invalidateLayer: vi.fn(),
    isGestureActive: () => false,
    isRasterCacheReady: () => options.cacheReady ?? true,
    layers,
    markDirty: calls.markDirty,
    notifyPainted: calls.notifyPainted,
    reportRefusal: calls.reportRefusal,
    requestRasterization: calls.requestRasterization,
    selection,
  });
  /** Every color a surface was filled with, in order. */
  const fillColors = () =>
    surfaces.flatMap((surface) =>
      surface.callLog
        .filter((entry) => entry.op === 'set' && entry.args[0] === 'fillStyle')
        .map((entry) => entry.args[1])
    );
  return { applyImagePatch, beginPixelEdit, calls, controller, fillColors, history, layers };
};

describe('SelectionPixelController', () => {
  it('records a raster fill as one step and persists it', () => {
    const h = createHarness();

    h.controller.run('fill');

    expect(h.calls.markDirty).toHaveBeenCalledWith('paint');
    expect(h.calls.notifyPainted).toHaveBeenCalledWith('paint');
    expect(h.history.entries().past).toEqual(['Fill selection']);
    // Before and after of the touched 2×2 region only.
    expect(h.history.byteSize()).toBe(2 * 2 * 4 * 2);
  });

  it('requests durable paint pixels instead of growing a transparent fill cache', () => {
    const h = createHarness({
      cacheReady: false,
      cacheRect: null,
      document: paintDocument({ bitmap: { height: 2, imageName: 'durable', width: 2 }, type: 'paint' }),
    });

    h.controller.run('fill');

    expect(h.calls.requestRasterization).toHaveBeenCalledWith('paint');
    expect(h.calls.reportRefusal).toHaveBeenCalledWith({
      layerId: 'paint',
      layerName: 'paint',
      status: 'layer-loading',
    });
    expect(h.layers.peek('paint')).toBeUndefined();
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it('refuses a fill too large to undo before touching the cache', () => {
    const h = createHarness({ byteBudget: 10, selectionRect: { height: 4, width: 4, x: -1, y: -1 } });
    const version = h.layers.version('paint');

    h.controller.run('fill');

    expect(h.calls.reportRefusal).toHaveBeenCalledWith('over-budget');
    expect(h.layers.get('paint')!.rect).toEqual({ height: 2, width: 2, x: 0, y: 0 });
    expect(h.layers.version('paint')).toBe(version);
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it('restores the touched region and the original extent when the fill cannot be published', () => {
    const h = createHarness({ mirrored: false, selectionRect: { height: 4, width: 4, x: -1, y: -1 } });
    const surface = h.layers.get('paint')!.surface as StubRasterSurface;

    h.controller.run('fill');

    expect(h.layers.get('paint')!.rect).toEqual({ height: 2, width: 2, x: 0, y: 0 });
    // The before pixels went back over the whole touched (grown) region before the crop.
    const restored = surface.callLog.filter((entry) => entry.op === 'putImageData').at(-1);
    expect(restored?.args.slice(1, 3)).toEqual([0, 0]);
    expect(h.calls.deleteDerived).toHaveBeenCalledWith('paint');
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it("edits a moved layer's pixels under the selection, in its local space", async () => {
    const h = createHarness({
      document: paintDocument({ type: 'paint' }, { transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 5 } }),
      selectionRect: { height: 2, width: 2, x: 10, y: 5 },
    });

    h.controller.run('fill');

    // The selection sits over the cached pixels at local (0, 0); the cache does not grow out to document (10, 5).
    expect(h.layers.get('paint')!.rect).toEqual({ height: 2, width: 2, x: 0, y: 0 });
    await h.history.undo();
    expect(h.applyImagePatch).toHaveBeenCalledWith('paint', { height: 2, width: 2, x: 0, y: 0 }, expect.anything());
  });

  it.each([
    ['an inpaint mask', 'inpaint_mask'],
    ['a regional guidance layer', 'regional_guidance'],
  ] as const)('fills %s with opaque mask coverage as one step and persists it', (_, type) => {
    const h = createHarness({ document: documentFrom([layerContract('paint', type)], 'paint') });

    h.controller.run('fill');

    expect(h.calls.reportRefusal).not.toHaveBeenCalled();
    expect(h.fillColors()).toEqual(['#ffffff']);
    expect(h.calls.markDirty).toHaveBeenCalledWith('paint');
    expect(h.history.entries().past).toEqual(['Fill selection']);
  });

  it('erases mask coverage as one step', () => {
    const h = createHarness({ document: documentFrom([layerContract('paint', 'inpaint_mask')], 'paint') });

    h.controller.run('erase');

    expect(h.calls.markDirty).toHaveBeenCalledWith('paint');
    expect(h.history.entries().past).toEqual(['Erase selection']);
  });

  it.each([
    ['fill', 'a disabled paint layer', layerContract('paint', 'raster', { isEnabled: false }), 'layer-hidden'],
    ['erase', 'a locked mask', layerContract('paint', 'inpaint_mask', { isLocked: true }), 'layer-locked'],
    [
      'fill',
      'a mask hidden on the canvas',
      layerContract('paint', 'inpaint_mask', { isHidden: true }),
      'layer-display-hidden',
    ],
    [
      'erase',
      'a transparency-locked paint layer',
      layerContract('paint', 'raster', { isTransparencyLocked: true } as never),
      'layer-alpha-locked',
    ],
  ] as const)('refuses to %s %s as the brush and eraser do', (kind, _, layer, status) => {
    const h = createHarness({ document: documentFrom([layer], 'paint') });

    h.controller.run(kind);

    expect(h.calls.reportRefusal).toHaveBeenCalledWith({ layerId: 'paint', layerName: 'paint', status });
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it('says what an erase cannot act on, and leaves Fill, which is never offered there, silent', () => {
    const group = createHarness({ document: documentFrom([groupContract('g', [layerContract('paint')])], 'g') });
    group.controller.run('erase');
    group.controller.run('fill');
    expect(group.calls.reportRefusal.mock.calls).toEqual([
      [{ layerId: 'g', layerName: 'g', status: 'selection-erase-unsupported' }],
    ]);

    const none = createHarness({ document: documentFrom([layerContract('paint')], null) });
    none.controller.run('erase');
    expect(none.calls.reportRefusal.mock.calls).toEqual([[{ status: 'selection-erase-no-target' }]]);
    expect(group.history.canUndo() || none.history.canUndo()).toBe(false);
  });

  it('edits a raster image through a pixel-edit transaction, which bakes it in place', () => {
    const transaction: PixelEditTransaction = {
      cancel: vi.fn(),
      commit: vi.fn(() => true),
      commitPatch: vi.fn(() => true),
      grow: vi.fn(() => true),
      layerId: 'paint',
    };
    const image = layerContract('paint', 'raster', {
      source: { image: { height: 2, imageName: 'photo', width: 2 }, type: 'image' },
    } as never);
    const h = createHarness({ beginPixelEdit: () => transaction, document: documentFrom([image], 'paint') });

    h.controller.run('fill');

    expect(transaction.commitPatch).toHaveBeenCalledWith(
      'Fill selection',
      expect.objectContaining({ rect: expect.anything() })
    );
    expect(transaction.cancel).not.toHaveBeenCalled();
  });

  it('erasing outside a raster image leaves it unbaked', () => {
    const image = layerContract('paint', 'raster', {
      source: { image: { height: 2, imageName: 'photo', width: 2 }, type: 'image' },
    } as never);
    const h = createHarness({
      document: documentFrom([image], 'paint'),
      selectionRect: { height: 2, width: 2, x: 20, y: 20 },
    });

    h.controller.run('erase');

    expect(h.beginPixelEdit).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it.each(['fill', 'erase'] as const)('refuses to %s while a transform session previews another placement', (kind) => {
    const transaction: PixelEditTransaction = {
      cancel: vi.fn(),
      commit: vi.fn(() => true),
      commitPatch: vi.fn(() => true),
      grow: vi.fn(() => true),
      layerId: 'paint',
    };
    const h = createHarness({
      beginPixelEdit: () => transaction,
      document: documentFrom([layerContract('paint', 'inpaint_mask')], 'paint'),
      transformSessionLayerId: 'paint',
    });

    h.controller.run(kind);

    expect(h.calls.reportRefusal).toHaveBeenCalledWith({
      layerId: 'paint',
      layerName: 'paint',
      status: 'transform-session',
    });
    expect(h.beginPixelEdit).not.toHaveBeenCalled();
    expect(h.calls.markDirty).not.toHaveBeenCalled();
    expect(h.history.canUndo()).toBe(false);
  });

  it("erases a control layer's pixels that are painted but not saved yet", () => {
    const transaction: PixelEditTransaction = {
      cancel: vi.fn(),
      commit: vi.fn(() => true),
      commitPatch: vi.fn(() => true),
      grow: vi.fn(() => true),
      layerId: 'paint',
    };
    // The contract still has no bitmap; only the live 2×2 cache holds the strokes.
    const h = createHarness({
      beginPixelEdit: () => transaction,
      document: documentFrom([layerContract('paint', 'control')], 'paint'),
    });

    h.controller.run('erase');

    expect(h.beginPixelEdit).toHaveBeenCalledWith('paint');
    expect(transaction.commitPatch).toHaveBeenCalledWith('Erase selection', expect.anything());
  });

  it('admits an erase only where one would run, reporting the refusal as the erase would', () => {
    const group = createHarness({ document: documentFrom([groupContract('g', [layerContract('paint')])], 'g') });
    expect(group.controller.admitErase()).toBe(false);
    expect(group.calls.reportRefusal).toHaveBeenCalledWith({
      layerId: 'g',
      layerName: 'g',
      status: 'selection-erase-unsupported',
    });

    const paint = createHarness();
    expect(paint.controller.admitErase()).toBe(true);
    expect(paint.calls.reportRefusal).not.toHaveBeenCalled();
    expect(paint.history.canUndo()).toBe(false);
  });
});
