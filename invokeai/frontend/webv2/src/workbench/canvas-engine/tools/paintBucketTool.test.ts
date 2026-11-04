import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { PaintBucketOptions } from '@workbench/canvas-engine/engineStores';
import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PlacedSurface, PointerInput, Rect } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores, DEFAULT_PAINT_BUCKET_OPTIONS } from '@workbench/canvas-engine/engineStores';
import { createFloodFillService, type FloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService';
import { createInlineFloodFillExecutor } from '@workbench/canvas-engine/render/fill/floodFillService.testing';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend, type StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createPaintBucketTool } from './paintBucketTool';
import { MASK_PAINT_COLOR } from './paintConstants';

/**
 * The stub backend reads back uniform opaque pixels, so every sample is one color and a fill takes its whole domain:
 * these tests cover where a fill goes, what it records and how it ends. Region shapes need real pixels and live in
 * the browser suite.
 */

const paintLayer = (over: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id: 'paint-1',
    isEnabled: true,
    isLocked: false,
    name: 'Paint',
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
    ...over,
  }) as CanvasLayerContract;

const docWith = (
  layers: CanvasNodeContract[],
  selectedLayerId: string | null,
  over: Partial<CanvasDocumentContractV3> = {}
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 96, width: 96, x: 0, y: 0 },
  height: 512,
  selectedLayerId,
  stacks: stacksFrom(layers),
  version: 3,
  width: 512,
  ...over,
});

const pointer = (x: number, y: number, buttons = 1): PointerInput => ({
  buttons,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

/** Lets the inline worker answer everything queued. */
const settle = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

const createHarness = (
  initial: CanvasDocumentContractV3,
  options: {
    selectionMask?: PlacedSurface;
    clipRect?: Rect;
    reserveRefused?: boolean;
    bucket?: Partial<PaintBucketOptions>;
    /** Replaces the inline worker, e.g. with one that fails. */
    executor?: () => FloodFillExecutor;
    commitThrows?: boolean;
  } = {}
) => {
  let doc = initial;
  const recording = createRecordingStrokeEdit();
  const strokeEdit = options.commitThrows
    ? {
        ...recording,
        edit: {
          ...recording.edit,
          commit: () => {
            throw new Error('history refused the step');
          },
        },
      }
    : recording;
  const refusals: CanvasEditRefusal[] = [];
  const dispatched: CanvasProjectMutation[] = [];
  const stores = createEngineStores();
  stores.paintBucketOptions.set({ ...DEFAULT_PAINT_BUCKET_OPTIONS, ...options.bucket });
  stores.brushOptions.set({ ...stores.brushOptions.get(), color: '#3366ff' });
  const stubBackend = createTestStubRasterBackend();
  const surfaces: StubRasterSurface[] = [];
  const backend = {
    ...stubBackend,
    createSurface: (width: number, height: number) => {
      const surface = stubBackend.createSurface(width, height);
      surfaces.push(surface);
      return surface;
    },
  };
  const layers = createLayerCacheStore(backend);
  const service = createFloodFillService({ createExecutor: options.executor ?? createInlineFloodFillExecutor });
  const reserved = { bytes: 0, held: 0 };
  const openPresses = new Set<number>();
  const requestRasterization = vi.fn();
  const ctx: ToolContext = {
    backend,
    beginPixelEdit: (layerId) => ({ ...strokeEdit.edit, commitPatch: () => true, layerId }),
    beginStrokeEdit: () => strokeEdit.edit,
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    commitStructural: () => ({ status: 'committed' as const }),
    createLayerId: () => 'created',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => {
      dispatched.push(action);
      if (action.type === 'addCanvasLayer') {
        doc = { ...doc, selectedLayerId: action.layer.id, stacks: stacksFrom([...doc.stacks.raster, action.layer]) };
      }
    },
    floodFill: {
      begin: (pressId, sample, seed, beginOptions, lease) => {
        openPresses.add(pressId);
        return service.begin(pressId, sample, seed, beginOptions, lease);
      },
      end: (pressId) => {
        openPresses.delete(pressId);
        service.end(pressId);
      },
      region: (pressId, params) => service.region(pressId, params),
      reserveSample: (bytes) => {
        if (options.reserveRefused) {
          return null;
        }
        reserved.bytes += bytes;
        reserved.held += 1;
        return { release: () => (reserved.held -= 1) };
      },
    },
    getDocument: () => doc,
    getSelectionMask: () => options.selectionMask ?? null,
    getStrokeClipRect: () => options.clipRect ?? null,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    reportEditRefusal: (refusal) => refusals.push(refusal),
    requestLayerRasterization: requestRasterization,
    scheduleFrame: () => () => undefined,
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: { documentToScreen: (p: { x: number; y: number }) => p } as unknown as Viewport,
  };
  const tool = createPaintBucketTool();
  return {
    /** Surfaces created so far, and `createImageData` calls across them. */
    allocations: () => ({
      imageData: surfaces.reduce(
        (count, surface) => count + surface.callLog.filter((entry) => entry.op === 'createImageData').length,
        0
      ),
      surfaces: surfaces.length,
    }),
    /** Every value the tool assigned to `prop` on any surface it drew into. */
    assigned: (prop: string) =>
      surfaces.flatMap((surface) =>
        surface.callLog.filter((entry) => entry.op === 'set' && entry.args[0] === prop).map((entry) => entry.args[1])
      ),
    click: async (x: number, y: number) => {
      tool.onPointerDown?.(ctx, pointer(x, y));
      tool.onPointerUp?.(ctx, pointer(x, y, 0));
      await settle();
    },
    ctx,
    dispatched,
    layers,
    openPresses,
    record: strokeEdit.record,
    refusals,
    requestRasterization,
    reserved,
    stores,
    tool,
  };
};

describe('paint bucket: a click', () => {
  it('fills the region on the selected paint layer as one step, reserving the sample only while handing it over', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    await h.click(40, 40);

    expect(h.record.commits).toHaveLength(1);
    const commit = h.record.commits[0]!;
    expect(commit).toMatchObject({ layerId: 'paint-1', tool: 'paintBucket' });
    expect(commit.createdLayer).toBeUndefined();
    // An empty layer samples the frame, and a one-color sample fills all of it.
    expect(commit.dirtyRect).toEqual({ height: 96, width: 96, x: 0, y: 0 });
    expect(h.layers.peek('paint-1')?.rect).toEqual({ height: 96, width: 96, x: 0, y: 0 });
    // The composite and its read-back pixels are alive together; the worker taking the buffer ends the reservation.
    expect(h.reserved).toEqual({ bytes: 96 * 96 * 8, held: 0 });
    expect(h.assigned('fillStyle')).toContain('#3366ff');
    expect(h.assigned('globalCompositeOperation')).toContain('source-over');
    expect(h.openPresses.size).toBe(0);
  });

  it('draws at the options opacity and records an erase with its own tool name', async () => {
    const fill = createHarness(docWith([paintLayer()], 'paint-1'), { bucket: { opacity: 40 } });
    await fill.click(10, 10);
    expect(fill.assigned('globalAlpha')).toContain(0.4);

    const erase = createHarness(docWith([paintLayer()], 'paint-1'), { bucket: { mode: 'erase' } });
    await erase.click(10, 10);
    expect(erase.record.commits[0]?.tool).toBe('paintBucketErase');
    expect(erase.assigned('globalCompositeOperation')).toContain('destination-out');
  });

  it('colors only existing pixels on a transparency-locked layer', async () => {
    const h = createHarness(docWith([paintLayer({ isTransparencyLocked: true } as never)], 'paint-1'));
    await h.click(10, 10);
    expect(h.record.commits).toHaveLength(1);
    expect(h.assigned('globalCompositeOperation')).toContain('source-atop');
  });

  it('paints a mask with opaque, hard mask coverage and its own expand, whatever the raster options say', async () => {
    const h = createHarness(docWith([layerContract('mask', 'inpaint_mask')], 'mask'), {
      bucket: { antiAlias: true, expandMask: 3, expandRaster: -5, opacity: 30 },
    });
    const begin = vi.spyOn(h.ctx.floodFill!, 'begin');
    await h.click(10, 10);

    expect(h.record.commits[0]).toMatchObject({ layerId: 'mask', tool: 'paintBucket' });
    expect(begin.mock.calls[0]![3]).toMatchObject({ antiAlias: false, expand: 3 });
    expect(h.assigned('fillStyle')).toContain(MASK_PAINT_COLOR);
    expect(h.assigned('fillStyle')).not.toContain('#3366ff');
    expect(h.assigned('globalAlpha')).toContain(1);
    expect(h.assigned('globalAlpha')).not.toContain(0.3);
    // Raster fills draw their coverage smoothed; a mask's never is, so a rotated mask keeps hard edges.
    expect(h.assigned('imageSmoothingEnabled')).not.toContain(true);
  });

  it('creates a paint layer above a group selection and records it with the fill', async () => {
    const h = createHarness(docWith([groupContract('group')], 'group'));
    await h.click(10, 10);

    expect(h.dispatched.map((action) => action.type)).toEqual(['addCanvasLayer']);
    expect(h.record.commits[0]?.layerId).toBe('created');
    expect(h.record.commits[0]?.createdLayer?.layer.id).toBe('created');
  });
});

describe('paint bucket: domain', () => {
  it('fills only inside the selection, and does nothing for a click outside it', async () => {
    const selection = {
      rect: { height: 20, width: 30, x: 10, y: 50 },
      surface: createTestStubRasterBackend().createSurface(30, 20),
    };
    const h = createHarness(docWith([paintLayer()], 'paint-1'), { selectionMask: selection });
    await h.click(5, 5);
    expect(h.record.commits).toHaveLength(0);
    expect(h.layers.peek('paint-1')).toBeUndefined();

    await h.click(20, 60);
    expect(h.record.commits[0]?.dirtyRect).toEqual({ height: 20, width: 30, x: 10, y: 50 });
    expect(h.reserved.bytes).toBe(30 * 20 * 8);
  });

  it('follows content past the frame unless Clip to frame keeps it inside', async () => {
    const content = (clip: boolean) => {
      const h = createHarness(
        docWith(
          [paintLayer({ source: { bitmap: { height: 200, imageName: 'p', width: 200 }, type: 'paint' } })],
          'paint-1'
        ),
        clip ? { clipRect: { height: 96, width: 96, x: 0, y: 0 } } : {}
      );
      h.layers.growToRect('paint-1', { height: 200, width: 200, x: 0, y: 0 }).stale = false;
      return h;
    };
    const free = content(false);
    await free.click(10, 10);
    expect(free.record.commits[0]?.dirtyRect).toEqual({ height: 200, width: 200, x: 0, y: 0 });

    const clipped = content(true);
    await clipped.click(10, 10);
    expect(clipped.record.commits[0]?.dirtyRect).toEqual({ height: 96, width: 96, x: 0, y: 0 });
  });

  it('refuses a domain over the pixel cap with its notice, and a selection makes it fillable', async () => {
    const huge = { bbox: { height: 5000, width: 5000, x: 0, y: 0 } };
    const h = createHarness(docWith([paintLayer()], 'paint-1', huge));
    await h.click(10, 10);
    expect(h.refusals).toEqual([{ status: 'fill-too-large' }]);
    expect(h.record).toMatchObject({ commits: [] });
    expect(h.reserved.bytes).toBe(0);

    const selected = createHarness(docWith([paintLayer()], 'paint-1', huge), {
      selectionMask: {
        rect: { height: 100, width: 100, x: 0, y: 0 },
        surface: createTestStubRasterBackend().createSurface(100, 100),
      },
    });
    await selected.click(10, 10);
    expect(selected.record.commits).toHaveLength(1);
  });

  it('reports a sample the raster budget cannot hold', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'), { reserveRefused: true });
    await h.click(10, 10);
    expect(h.refusals).toEqual(['over-budget']);
    expect(h.record.commits).toHaveLength(0);
    expect(h.record.cancelled).toBeGreaterThan(0);
  });
});

describe('paint bucket: tolerance drag', () => {
  it('scrubs from the press value, shows the readout, and keeps the tolerance it lands on', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    const region = vi.spyOn(h.ctx.floodFill!, 'region');
    const move = (x: number, y: number) => h.tool.onPointerMove?.(h.ctx, pointer(x, y), [pointer(x, y)]);
    h.tool.onPointerDown?.(h.ctx, pointer(50, 50));
    // Inside the drag threshold nothing scrubs.
    move(52, 50);
    expect(h.stores.paintBucketScrub.get()).toBeNull();

    move(90, 80);
    expect(h.stores.paintBucketScrub.get()).toEqual({ screenPoint: { x: 90, y: 80 }, tolerance: 30 });
    move(-450, 80);
    expect(h.stores.paintBucketScrub.get()?.tolerance).toBe(0);
    await settle();
    expect(region.mock.calls.map(([, params]) => params.tolerance)).toEqual([30, 0]);

    move(110, 80);
    h.tool.onPointerUp?.(h.ctx, pointer(110, 80, 0));
    expect(h.stores.paintBucketScrub.get()).toBeNull();
    await settle();

    expect(h.stores.paintBucketOptions.get().tolerance).toBe(35);
    expect(h.record.commits).toHaveLength(1);
    expect(h.openPresses.size).toBe(0);
  });

  it('commits only after the answer to the last request arrives', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    h.tool.onPointerDown?.(h.ctx, pointer(50, 50));
    h.tool.onPointerMove?.(h.ctx, pointer(110, 50), [pointer(110, 50)]);
    h.tool.onPointerUp?.(h.ctx, pointer(110, 50, 0));
    expect(h.record.commits).toHaveLength(0);
    expect(h.openPresses.size).toBe(1);
    // The options form shows the landed tolerance at once, not the press value until the fill lands.
    expect(h.stores.paintBucketOptions.get().tolerance).toBe(35);
    await settle();
    expect(h.record.commits).toHaveLength(1);
  });

  it('redraws each scrub step without allocating another surface or pixel buffer', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    h.tool.onPointerDown?.(h.ctx, pointer(50, 50));
    h.tool.onPointerMove?.(h.ctx, pointer(60, 50), [pointer(60, 50)]);
    await settle();
    const afterFirstRedraw = h.allocations();
    for (const x of [70, 80, 90, 100, 110]) {
      h.tool.onPointerMove?.(h.ctx, pointer(x, 50), [pointer(x, 50)]);
      await settle();
    }
    expect(h.allocations()).toEqual(afterFirstRedraw);
    h.tool.onPointerUp?.(h.ctx, pointer(110, 50, 0));
    await settle();
    expect(h.record.commits).toHaveLength(1);
  });

  it('leaves the stored tolerance alone after a steady click', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'), { bucket: { tolerance: 12 } });
    await h.click(10, 10);
    expect(h.stores.paintBucketOptions.get().tolerance).toBe(12);
  });
});

describe('paint bucket: cancel', () => {
  it('puts the pixels back and records nothing on Escape during the hold', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    h.tool.onPointerDown?.(h.ctx, pointer(10, 10));
    await settle();
    expect(h.layers.peek('paint-1')).toBeDefined();
    h.tool.onPointerMove?.(h.ctx, pointer(40, 10), [pointer(40, 10)]);

    h.tool.onPointerCancel?.(h.ctx);
    await settle();
    expect(h.layers.peek('paint-1')).toBeUndefined();
    expect(h.record.commits).toHaveLength(0);
    expect(h.stores.paintBucketScrub.get()).toBeNull();
    expect(h.stores.paintBucketOptions.get().tolerance).toBe(DEFAULT_PAINT_BUCKET_OPTIONS.tolerance);
    expect(h.openPresses.size).toBe(0);
  });

  it('removes a layer the press created and reselects what was selected', async () => {
    const h = createHarness(docWith([groupContract('group')], 'group'));
    h.tool.onPointerDown?.(h.ctx, pointer(10, 10));
    await settle();
    h.tool.onDeactivate?.(h.ctx);

    expect(h.dispatched.map((action) => action.type)).toEqual([
      'addCanvasLayer',
      'removeCanvasLayers',
      'setCanvasSelectedLayer',
    ]);
    expect(h.dispatched[2]).toMatchObject({ id: 'group' });
    expect(h.record.commits).toHaveLength(0);
  });

  it('puts back a released fill still waiting on the worker when the engine cancels its settling', async () => {
    const h = createHarness(docWith([groupContract('group')], 'group'));
    h.tool.onPointerDown?.(h.ctx, pointer(10, 10));
    await settle();
    h.tool.onPointerMove?.(h.ctx, pointer(60, 10), [pointer(60, 10)]);
    h.tool.onPointerUp?.(h.ctx, pointer(60, 10, 0));

    // Synchronous: whatever runs next (a flush) sees the layer as it was.
    h.tool.cancelSettling?.(h.ctx);
    expect(h.layers.peek('created')).toBeUndefined();
    expect(h.dispatched.map((action) => action.type)).toEqual([
      'addCanvasLayer',
      'removeCanvasLayers',
      'setCanvasSelectedLayer',
    ]);
    expect(h.openPresses.size).toBe(0);
    await settle();
    expect(h.record.commits).toHaveLength(0);
  });

  it('puts the layer back when the worker fails mid-press', async () => {
    let failing = false;
    const inline = createInlineFloodFillExecutor();
    const h = createHarness(docWith([paintLayer()], 'paint-1'), {
      executor: () => ({
        ...inline,
        run: (job) => (failing ? Promise.reject(new Error('worker died')) : inline.run(job)),
      }),
    });
    h.tool.onPointerDown?.(h.ctx, pointer(10, 10));
    await settle();
    expect(h.layers.peek('paint-1')).toBeDefined();

    failing = true;
    h.tool.onPointerMove?.(h.ctx, pointer(60, 10), [pointer(60, 10)]);
    await settle();
    expect(h.layers.peek('paint-1')).toBeUndefined();
    expect(h.openPresses.size).toBe(0);
    expect(h.stores.paintBucketScrub.get()).toBeNull();
    h.tool.onPointerUp?.(h.ctx, pointer(60, 10, 0));
    await settle();
    expect(h.record.commits).toHaveLength(0);
  });

  it('removes a created layer when recording the fill throws', async () => {
    const h = createHarness(docWith([groupContract('group')], 'group'), { commitThrows: true });
    await h.click(10, 10);

    expect(h.dispatched.map((action) => action.type)).toEqual([
      'addCanvasLayer',
      'removeCanvasLayers',
      'setCanvasSelectedLayer',
    ]);
    expect(h.layers.peek('created')).toBeUndefined();
    expect(h.openPresses.size).toBe(0);
  });

  it('keeps a held press through a modifier-hold tool switch', async () => {
    const h = createHarness(docWith([paintLayer()], 'paint-1'));
    h.tool.onPointerDown?.(h.ctx, pointer(10, 10));
    h.tool.onDeactivate?.(h.ctx, { temporary: true });
    h.tool.onPointerUp?.(h.ctx, pointer(10, 10, 0));
    await settle();
    expect(h.record.commits).toHaveLength(1);
  });
});

describe('paint bucket: refusals', () => {
  it.each([
    ['hidden', { isEnabled: false }, 'layer-hidden'],
    ['locked', { isLocked: true }, 'layer-locked'],
  ] as const)('refuses a %s paint layer with its notice', async (_name, over, status) => {
    const h = createHarness(docWith([paintLayer(over)], 'paint-1'));
    expect(h.tool.cursor?.(h.ctx)).toBe('not-allowed');
    await h.click(10, 10);
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status }]);
    expect(h.record.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
  });

  it('asks for pixels that have not loaded and fills nothing', async () => {
    const h = createHarness(
      docWith([paintLayer({ source: { bitmap: { height: 10, imageName: 'p', width: 10 }, type: 'paint' } })], 'paint-1')
    );
    await h.click(10, 10);
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-loading' }]);
    expect(h.requestRasterization).toHaveBeenCalledWith('paint-1');
  });

  it('refuses an erase under transparency lock, as the Eraser does', async () => {
    const h = createHarness(docWith([paintLayer({ isTransparencyLocked: true } as never)], 'paint-1'), {
      bucket: { mode: 'erase' },
    });
    expect(h.tool.cursor?.(h.ctx)).toBe('not-allowed');
    await h.click(10, 10);
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-alpha-locked' }]);
    expect(h.record).toMatchObject({ commits: [] });
    expect(h.layers.peek('paint-1')).toBeUndefined();
  });

  it('refuses an erase quietly where there is nothing paintable to erase', async () => {
    for (const doc of [docWith([groupContract('group')], 'group'), docWith([paintLayer()], null)]) {
      const h = createHarness(doc, { bucket: { mode: 'erase' } });
      expect(h.tool.cursor?.(h.ctx)).toBe('not-allowed');
      await h.click(10, 10);
      expect(h.refusals).toEqual([]);
      expect(h.dispatched).toEqual([]);
      expect(h.record.commits).toHaveLength(0);
    }
  });

  it('shows a crosshair where a fill would land', () => {
    for (const doc of [docWith([paintLayer()], 'paint-1'), docWith([groupContract('group')], 'group')]) {
      const h = createHarness(doc);
      expect(h.tool.cursor?.(h.ctx)).toBe('crosshair');
    }
  });
});
