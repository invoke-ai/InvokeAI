import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { PaintBucketOptions } from '@workbench/canvas-engine/engineStores';
import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PlacedSurface, PointerInput, Rect } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import {
  documentFrom,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores, DEFAULT_PAINT_BUCKET_OPTIONS } from '@workbench/canvas-engine/engineStores';
import { applyToPoint, fromTRS, identity, invert, multiply, translate } from '@workbench/canvas-engine/math/mat2d';
import { roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { createFloodFillService } from '@workbench/canvas-engine/render/fill/floodFillService';
import { createInlineFloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService.testing';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createPaintBucketTool } from './paintBucketTool';

/** Real pixels and the real region kernels: what lands where for each target and option. */

const pointer = (x: number, y: number, buttons = 1): PointerInput => ({
  buttons,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

const settle = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

/** Fills document rects of a layer's cache (identity transform) with colors; the rest stays transparent. */
type Paint = readonly { rect: Rect; color: string }[];

const harness = (
  layers: CanvasLayerContract[],
  selectedId: string | null,
  paint: Record<string, Paint>,
  options: { selection?: PlacedSurface } = {}
) => {
  const backend = createDomRasterBackend();
  const cache = createLayerCacheStore(backend);
  for (const [layerId, fills] of Object.entries(paint)) {
    const entry = cache.growToRect(layerId, { height: 100, width: 100, x: 0, y: 0 });
    entry.stale = false;
    for (const { color, rect } of fills) {
      entry.surface.ctx.fillStyle = color;
      entry.surface.ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
  }
  const stores = createEngineStores();
  stores.brushOptions.set({ ...stores.brushOptions.get(), color: '#2050e0' });
  const strokeEdit = createRecordingStrokeEdit();
  const service = createFloodFillService({ createExecutor: createInlineFloodFillExecutor });
  let doc = { ...documentFrom(layers, selectedId), bbox: { height: 100, width: 100, x: 0, y: 0 } };
  const transformOverrides = new Map<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >();
  const dispatched: CanvasProjectMutation[] = [];
  const ctx: ToolContext = {
    backend,
    /**
     * The engine's materialization contract for raster images and control layers: the cache becomes the layer's
     * displayed pixels baked into document space, drawn at an identity override while the document keeps the layer's
     * transform until the edit commits.
     */
    beginPixelEdit: (layerId) => {
      const layer = layers.find((candidate) => candidate.id === layerId)!;
      const entry = cache.peek(layerId)!;
      const { rotation, scaleX, scaleY, x, y } = layer.transform;
      const matrix = fromTRS({ x, y }, rotation, scaleX, scaleY);
      const bounds = roundOut(transformBounds(matrix, entry.rect));
      const baked = backend.createSurface(bounds.width, bounds.height);
      const place = multiply(translate(identity(), { x: -bounds.x, y: -bounds.y }), matrix);
      baked.ctx.setTransform(place.a, place.b, place.c, place.d, place.e, place.f);
      baked.ctx.drawImage(entry.surface.canvas, entry.rect.x, entry.rect.y);
      cache.delete(layerId);
      const bakedEntry = cache.growToRect(layerId, bounds);
      bakedEntry.stale = false;
      bakedEntry.surface.ctx.drawImage(baked.canvas, 0, 0);
      transformOverrides.set(layerId, { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
      return { ...strokeEdit.edit, commitPatch: () => true, layerId };
    },
    beginStrokeEdit: () => strokeEdit.edit,
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    commitStructural: () => ({ status: 'committed' as const }),
    createLayerId: () => 'created',
    createPath2D: (d) => new Path2D(d),
    dispatch: (action) => {
      dispatched.push(action);
      if (action.type === 'addCanvasLayer') {
        doc = { ...doc, selectedLayerId: action.layer.id, stacks: stacksFrom([action.layer, ...doc.stacks.raster]) };
      }
    },
    floodFill: {
      begin: (...args) => service.begin(...args),
      end: (pressId) => service.end(pressId),
      region: (...args) => service.region(...args),
      reserveSample: () => ({ release: () => undefined }),
    },
    getDocument: () => doc,
    getSelectionMask: () => options.selection ?? null,
    invalidate: vi.fn(),
    layers: cache,
    notifyLayerPainted: vi.fn(),
    scheduleFrame: () => () => undefined,
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    transformOverrides,
    updateCursor: vi.fn(),
    viewport: { documentToScreen: (p: { x: number; y: number }) => p } as unknown as Viewport,
  };
  const tool = createPaintBucketTool();
  /** RGBA of `layerId` at the document point, through its transform. */
  const pixelAt = (layerId: string, x: number, y: number): number[] => {
    const layer = layers.find((candidate) => candidate.id === layerId)!;
    const { rotation, scaleX, scaleY, x: tx, y: ty } = layer.transform;
    const local = applyToPoint(invert(fromTRS({ x: tx, y: ty }, rotation, scaleX, scaleY))!, { x, y });
    const entry = cache.peek(layerId);
    const lx = Math.floor(local.x) - (entry?.rect.x ?? 0);
    const ly = Math.floor(local.y) - (entry?.rect.y ?? 0);
    if (!entry || lx < 0 || ly < 0 || lx >= entry.rect.width || ly >= entry.rect.height) {
      return [0, 0, 0, 0];
    }
    return [...entry.surface.ctx.getImageData(lx, ly, 1, 1).data];
  };
  /** RGBA of a cache at a document point, for layers drawn at identity (created or materialized). */
  const documentPixelAt = (layerId: string, x: number, y: number): number[] => {
    const entry = cache.peek(layerId);
    const lx = Math.floor(x) - (entry?.rect.x ?? 0);
    const ly = Math.floor(y) - (entry?.rect.y ?? 0);
    if (!entry || lx < 0 || ly < 0 || lx >= entry.rect.width || ly >= entry.rect.height) {
      return [0, 0, 0, 0];
    }
    return [...entry.surface.ctx.getImageData(lx, ly, 1, 1).data];
  };
  return {
    cache,
    dispatched,
    documentPixelAt,
    click: async (x: number, y: number, options: Partial<PaintBucketOptions> = {}) => {
      stores.paintBucketOptions.set({ ...DEFAULT_PAINT_BUCKET_OPTIONS, ...options });
      tool.onPointerDown?.(ctx, pointer(x, y));
      tool.onPointerUp?.(ctx, pointer(x, y, 0));
      await settle();
    },
    ctx,
    pixelAt,
    record: strokeEdit.record,
    stores,
    tool,
  };
};

const BLUE = [32, 80, 224, 255];
const CLEAR = [0, 0, 0, 0];

describe('paint bucket in Chromium', () => {
  it('samples every visible layer for All layers, and only the target for Layer', async () => {
    // The target lies under a layer with a red square; clicking the empty corner fills around the square only when
    // the fill sees it.
    const layers = [layerContract('above'), layerContract('target')];
    const paint = { above: [{ color: '#ff0000', rect: { height: 20, width: 20, x: 40, y: 40 } }] };

    const all = harness(layers, 'target', paint);
    await all.click(5, 5, { sample: 'all' });
    expect(all.pixelAt('target', 5, 5)).toEqual(BLUE);
    expect(all.pixelAt('target', 50, 50)).toEqual(CLEAR);

    const own = harness(layers, 'target', paint);
    await own.click(5, 5, { sample: 'layer' });
    expect(own.pixelAt('target', 5, 5)).toEqual(BLUE);
    expect(own.pixelAt('target', 50, 50)).toEqual(BLUE);
  });

  it('fills the sampled region under the cursor on a rotated layer', async () => {
    // A black outline square (document 20..70) below a rotated empty paint layer.
    const outline = [
      { color: '#000000', rect: { height: 2, width: 50, x: 20, y: 20 } },
      { color: '#000000', rect: { height: 2, width: 50, x: 20, y: 68 } },
      { color: '#000000', rect: { height: 50, width: 2, x: 20, y: 20 } },
      { color: '#000000', rect: { height: 50, width: 2, x: 68, y: 20 } },
    ];
    const rotated = layerContract('rotated', 'raster', {
      transform: { rotation: 30, scaleX: 1, scaleY: 1, x: 45, y: 10 },
    });
    const h = harness([rotated, layerContract('lines')], 'rotated', { lines: outline });
    await h.click(45, 45, { antiAlias: false });

    expect(h.record.commits).toHaveLength(1);
    for (const [x, y] of [
      [26, 26],
      [45, 45],
      [64, 64],
      [26, 64],
    ] as const) {
      expect(h.pixelAt('rotated', x, y)).toEqual(BLUE);
    }
    for (const [x, y] of [
      [10, 45],
      [80, 45],
      [45, 85],
      [45, 15],
    ] as const) {
      expect(h.pixelAt('rotated', x, y)[3]).toBe(0);
    }
  });

  it('covers a mask with opaque, hard coverage grown by the mask Expand', async () => {
    const h = harness([layerContract('mask', 'inpaint_mask'), layerContract('art')], 'mask', {
      art: [{ color: '#202020', rect: { height: 20, width: 20, x: 30, y: 30 } }],
    });
    await h.click(40, 40, { expandMask: 2, opacity: 30 });

    // The square spans 30..49; Expand +2 takes 28..51 and nothing past it, at full alpha.
    for (const x of [28, 29, 40, 50, 51]) {
      expect(h.pixelAt('mask', x, 40)[3]).toBe(255);
    }
    expect(h.pixelAt('mask', 27, 40)[3]).toBe(0);
    expect(h.pixelAt('mask', 52, 40)[3]).toBe(0);
    expect(h.pixelAt('mask', 40, 27)[3]).toBe(0);
  });

  it('commits the region of the tolerance a drag landed on', async () => {
    // Two flat grays about 27 % apart in OKLab lightness.
    const h = harness([layerContract('target'), layerContract('art')], 'target', {
      art: [
        { color: '#505050', rect: { height: 100, width: 50, x: 0, y: 0 } },
        { color: '#a0a0a0', rect: { height: 100, width: 50, x: 50, y: 0 } },
      ],
    });
    h.stores.paintBucketOptions.set({ ...DEFAULT_PAINT_BUCKET_OPTIONS, tolerance: 10 });
    h.tool.onPointerDown?.(h.ctx, pointer(20, 20));
    await settle();
    expect(h.pixelAt('target', 20, 20)).toEqual(BLUE);
    expect(h.pixelAt('target', 80, 20)).toEqual(CLEAR);

    // +30 % (120 px right) takes the second gray too, live.
    h.tool.onPointerMove?.(h.ctx, pointer(140, 20), [pointer(140, 20)]);
    await settle();
    expect(h.pixelAt('target', 80, 20)).toEqual(BLUE);

    h.tool.onPointerUp?.(h.ctx, pointer(140, 20, 0));
    await settle();
    expect(h.stores.paintBucketOptions.get().tolerance).toBe(40);
    const commit = h.record.commits[0]!;
    expect(commit.dirtyRect).toEqual({ height: 100, width: 100, x: 0, y: 0 });
    const after = commit.afterImageData.data;
    expect([...after.subarray((20 * 100 + 80) * 4, (20 * 100 + 80) * 4 + 4)]).toEqual(BLUE);
  });
  it.each([
    ['moved', { rotation: 0, scaleX: 1, scaleY: 1, x: 512, y: 0 }],
    ['rotated', { rotation: 30, scaleX: 1, scaleY: 1, x: 60, y: 10 }],
  ] as const)('fills where the picture is on a %s image layer it materializes', async (_name, transform) => {
    // The picture: a red square at local 20..60 on transparency.
    const image = layerContract('image', 'raster', {
      source: { image: { height: 100, imageName: 'picture', width: 100 }, type: 'image' },
      transform,
    } as Partial<CanvasLayerContract>);
    const h = harness([image], 'image', {
      image: [{ color: '#ff0000', rect: { height: 40, width: 40, x: 20, y: 20 } }],
    });
    const matrix = fromTRS({ x: transform.x, y: transform.y }, transform.rotation, transform.scaleX, transform.scaleY);
    const inside = applyToPoint(matrix, { x: 40, y: 40 });
    const besideIt = applyToPoint(matrix, { x: 8, y: 40 });
    await h.click(inside.x, inside.y);

    expect(h.record.commits).toHaveLength(1);
    expect(h.documentPixelAt('image', inside.x, inside.y)).toEqual(BLUE);
    // The transparent part of the picture is another region: untouched.
    expect(h.documentPixelAt('image', besideIt.x, besideIt.y)[3]).toBe(0);
  });

  it('records nothing for a region the selection shape leaves nothing of, and keeps no created layer', async () => {
    // An elliptical selection over the frame; the red square in its corner lies inside its bounds, outside its shape.
    const selection = createDomRasterBackend().createSurface(100, 100);
    selection.ctx.beginPath();
    selection.ctx.ellipse(50, 50, 50, 50, 0, 0, Math.PI * 2);
    selection.ctx.fill();
    const paint = { art: [{ color: '#ff0000', rect: { height: 12, width: 12, x: 0, y: 0 } }] };
    const placed = { rect: { height: 100, width: 100, x: 0, y: 0 }, surface: selection };

    const corner = harness([layerContract('art')], null, paint, { selection: placed });
    await corner.click(3, 3);
    expect(corner.record.commits).toHaveLength(0);
    expect(corner.dispatched.map((action) => action.type)).toEqual([
      'addCanvasLayer',
      'removeCanvasLayers',
      'setCanvasSelectedLayer',
    ]);
    expect(corner.cache.peek('created')).toBeUndefined();

    const middle = harness([layerContract('art')], null, paint, { selection: placed });
    await middle.click(50, 50);
    expect(middle.record.commits).toHaveLength(1);
    expect(middle.documentPixelAt('created', 50, 50)).toEqual(BLUE);
    expect(middle.documentPixelAt('created', 3, 3)[3]).toBe(0);
  });

  it('samples every layer for a layer it creates, even with Sample set to Layer', async () => {
    const h = harness([layerContract('art')], null, {
      art: [{ color: '#ff0000', rect: { height: 20, width: 20, x: 40, y: 40 } }],
    });
    await h.click(5, 5, { sample: 'layer' });
    expect(h.documentPixelAt('created', 5, 5)).toEqual(BLUE);
    // The new layer is empty; sampling it alone would have filled over the square.
    expect(h.documentPixelAt('created', 50, 50)[3]).toBe(0);
  });
});
