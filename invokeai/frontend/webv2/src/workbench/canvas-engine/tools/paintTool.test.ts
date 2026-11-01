import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/contracts';
import type { StubRasterBackend, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type {
  PixelEditTransaction,
  StrokeCommittedEvent,
  Tool,
  ToolContext,
} from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { createLayerCacheStore, type LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createBrushTool } from '@workbench/canvas-engine/tools/brushTool';
import { createEraserTool } from '@workbench/canvas-engine/tools/eraserTool';
import { describe, expect, it, vi } from 'vitest';

const paintLayer = (
  id: string,
  opts: { isLocked?: boolean; isEnabled?: boolean; isTransparencyLocked?: boolean } = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: opts.isEnabled ?? true,
  isLocked: opts.isLocked ?? false,
  ...(opts.isTransparencyLocked ? { isTransparencyLocked: true } : {}),
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

const imageLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { image: { height: 10, imageName: id, width: 10 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

/** A raster the brush cannot edit in place, so a stroke auto-creates a paint layer above it. */
const textLayer = (id: string): CanvasLayerContract => ({
  ...(imageLayer(id) as CanvasRasterLayerContractV2),
  source: {
    align: 'left',
    color: '#000000',
    content: id,
    fontFamily: 'sans-serif',
    fontSize: 12,
    fontWeight: 400,
    lineHeight: 1.2,
    type: 'text',
  },
});

const inpaintMaskLayer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
  name: id,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

const controlPaintLayer = (id: string): CanvasControlLayerContract => ({
  adapter: { beginEndStepPct: [0, 1], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 1 },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

const controlImageLayer = (id: string): CanvasControlLayerContract => ({
  ...controlPaintLayer(id),
  source: { image: { height: 20, imageName: id, width: 20 }, type: 'image' },
});

const controlTransaction = (layerId = 'control'): PixelEditTransaction => ({
  cancel: vi.fn(),
  commit: vi.fn(() => true),
  commitPatch: vi.fn(() => true),
  grow: vi.fn(() => true),
  layerId,
});

const makeDoc = (layers: CanvasLayerContract[], selectedLayerId: string | null): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  stacks: stacksFrom(layers),
  selectedLayerId,
  version: 3,
  width: 100,
});

const pointer = (
  x: number,
  y: number,
  opts: { pressure?: number; buttons?: number; shift?: boolean } = {}
): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: opts.shift ?? false },
  pointerType: 'mouse',
  pressure: opts.pressure ?? 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

// Optional-method drivers, so tests read as gestures rather than `?.` noise.
const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput, batch: PointerInput[]): void =>
  t.onPointerMove?.(ctx, i, batch);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);
const cancel = (t: Tool, ctx: ToolContext): void => t.onPointerCancel?.(ctx);

interface Harness {
  beginPixelEdit: ReturnType<typeof vi.fn> | null;
  ctx: ToolContext;
  backend: StubRasterBackend;
  layers: LayerCacheStore;
  dispatched: CanvasProjectMutation[];
  strokes: StrokeCommittedEvent[];
  painted: string[];
  requestLayerRasterization: ReturnType<typeof vi.fn>;
  createdIds: string[];
  /** Every admitted stroke edit, so a test can check each one ended. */
  strokeEdits: { cancel: ReturnType<typeof vi.fn> }[];
  /** What the tool reported through `reportEditRefusal`. */
  refusals: CanvasEditRefusal[];
}

const createHarness = (
  doc: CanvasDocumentContractV3,
  transaction: PixelEditTransaction | null | undefined = undefined,
  refuseStrokes = false
): Harness => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const stores = createEngineStores();
  const dispatched: CanvasProjectMutation[] = [];
  const strokes: StrokeCommittedEvent[] = [];
  const strokeEdits: Harness['strokeEdits'] = [];
  const painted: string[] = [];
  const createdIds: string[] = [];
  const beginPixelEdit = transaction === undefined ? null : vi.fn(() => transaction);
  const requestLayerRasterization = vi.fn();
  const refusals: CanvasEditRefusal[] = [];
  let idCounter = 0;

  const ctx: ToolContext = {
    scheduleFrame: (task) => {
      task();
      return () => undefined;
    },
    backend,
    ...(beginPixelEdit ? { beginPixelEdit } : {}),
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    commitStructural: vi.fn(),
    createLayerId: () => {
      const id = `new-layer-${(idCounter += 1)}`;
      createdIds.push(id);
      return id;
    },
    createPath2D: () => {
      const path = { closePath: () => {}, lineTo: () => {}, moveTo: () => {}, quadraticCurveTo: () => {} };
      return path as unknown as Path2D;
    },
    dispatch: (action) => dispatched.push(action),
    beginStrokeEdit: () => {
      if (refuseStrokes) {
        return null;
      }
      const edit = {
        cancel: vi.fn(),
        commit: (event: StrokeCommittedEvent) => {
          painted.push(event.layerId);
          strokes.push(event);
          return true;
        },
        grow: () => true,
      };
      strokeEdits.push(edit);
      return edit;
    },
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: (layerId) => painted.push(layerId),
    reportEditRefusal: (refusal) => refusals.push(refusal),
    requestLayerRasterization,
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: null as never,
  };

  return {
    backend,
    beginPixelEdit,
    createdIds,
    ctx,
    dispatched,
    layers,
    painted,
    refusals,
    requestLayerRasterization,
    strokeEdits,
    strokes,
  };
};

const cacheOps = (surface: StubRasterSurface): string[] => surface.callLog.map((entry) => entry.op);

/** The value of the last `globalCompositeOperation` assignment recorded on the cache surface. */
const lastCompositeOp = (surface: StubRasterSurface): unknown =>
  surface.callLog
    .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation')
    .map((entry) => entry.args[1])
    .pop();

/** The value of the last `globalAlpha` assignment recorded on the cache surface. */
const lastGlobalAlpha = (surface: StubRasterSurface): unknown =>
  surface.callLog
    .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalAlpha')
    .map((entry) => entry.args[1])
    .pop();

const cacheSurface = (h: Harness, layerId: string): StubRasterSurface =>
  h.layers.get(layerId)!.surface as StubRasterSurface;

describe('brush tool: stroke into an existing paint layer', () => {
  it('paints, emits one strokeCommitted with a content-sized dirty rect, and does not dispatch', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(30, 30), pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(h.dispatched).toHaveLength(0);
    expect(h.strokes).toHaveLength(1);
    const event = h.strokes[0]!;
    expect(event.tool).toBe('brush');
    expect(event.layerId).toBe('paint1');

    // Dirty bounds remain integer and nonempty beyond document edges as paint content grows.
    const { dirtyRect } = event;
    expect(Number.isInteger(dirtyRect.x)).toBe(true);
    expect(Number.isInteger(dirtyRect.width)).toBe(true);
    expect(isEmpty(dirtyRect)).toBe(false);

    // Before/after snapshots are sized exactly to the dirty rect.
    expect(event.beforeImageData.width).toBe(dirtyRect.width);
    expect(event.beforeImageData.height).toBe(dirtyRect.height);
    expect(event.afterImageData.width).toBe(dirtyRect.width);
    expect(event.afterImageData.height).toBe(dirtyRect.height);

    // The layer version was bumped exactly once (on commit).
    expect(h.painted).toEqual(['paint1']);

    // The cache surface received the buffered-stroke composite (source-over).
    const ops = cacheOps(cacheSurface(h, 'paint1'));
    expect(ops).toContain('getImageData');
    expect(ops).toContain('drawImage');
    expect(lastCompositeOp(cacheSurface(h, 'paint1'))).toBe('source-over');
  });

  it('waits for a persisted paint source cache instead of painting over transparent pixels', () => {
    const layer = paintLayer('paint1');
    if (layer.type !== 'raster' || layer.source.type !== 'paint') {
      throw new Error('expected a raster paint layer');
    }
    layer.source.bitmap = { height: 10, imageName: 'persisted-paint', width: 10 };
    const h = createHarness(makeDoc([layer], layer.id));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(5, 5));
    up(brush, h.ctx, pointer(5, 5, { buttons: 0 }));
    expect(h.strokes).toHaveLength(0);
    expect(h.layers.peek(layer.id)).toBeUndefined();
    expect(h.requestLayerRasterization).toHaveBeenCalledWith(layer.id);
    expect(h.refusals).toEqual([{ layerId: layer.id, layerName: layer.name, status: 'layer-loading' }]);

    const entry = h.layers.getOrCreateRect(layer.id, { height: 10, width: 10, x: 0, y: 0 });
    h.layers.publishPixels(layer.id);
    expect(entry.stale).toBe(false);
    down(brush, h.ctx, pointer(5, 5));
    up(brush, h.ctx, pointer(5, 5, { buttons: 0 }));
    expect(h.strokes).toHaveLength(1);
  });
});

describe('eraser tool', () => {
  it('composites the stroke with destination-out and reports tool "eraser"', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    const eraser = createEraserTool();

    down(eraser, h.ctx, pointer(20, 20));
    move(eraser, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(eraser, h.ctx, pointer(40, 40));

    expect(h.strokes).toHaveLength(1);
    expect(h.strokes[0]!.tool).toBe('eraser');
    expect(lastCompositeOp(cacheSurface(h, 'paint1'))).toBe('destination-out');
  });

  /** The committed width of one horizontal eraser stroke drawn at a constant pen pressure. */
  const erasedHeight = (pressure: number, pressureAffectsWidth: boolean): number => {
    const h = createHarness(makeDoc([paintLayer('paint1')], 'paint1'));
    h.ctx.stores.eraserOptions.set({ ...h.ctx.stores.eraserOptions.get(), pressureAffectsWidth, size: 20 });
    const eraser = createEraserTool();
    const input = (x: number) => ({ ...pointer(x, 50, { pressure }), pointerType: 'pen' as const });

    down(eraser, h.ctx, input(20));
    move(eraser, h.ctx, input(60), [input(40), input(60)]);
    up(eraser, h.ctx, input(80));

    expect(h.strokes).toHaveLength(1);
    return h.strokes[0]!.dirtyRect.height;
  };

  it('thins the stroke with pen pressure when width dynamics are on', () => {
    expect(erasedHeight(0.1, true)).toBeLessThan(erasedHeight(1, true));
  });

  it('keeps a constant width regardless of pressure when width dynamics are off', () => {
    expect(erasedHeight(0.1, false)).toBe(erasedHeight(1, false));
  });

  it.each([
    ['eraser', createEraserTool],
    ['brush', createBrushTool],
  ] as const)('%s edits a selected image layer through a materializing pixel transaction', (_, createTool) => {
    const transaction = controlTransaction('img1');
    const h = createHarness(makeDoc([imageLayer('img1')], 'img1'), transaction);
    h.layers.getOrCreateRect('img1', { height: 10, width: 10, x: 0, y: 0 }).stale = false;
    const tool = createTool();

    down(tool, h.ctx, pointer(2, 2));
    move(tool, h.ctx, pointer(6, 6), [pointer(6, 6)]);
    up(tool, h.ctx, pointer(6, 6));

    expect(h.beginPixelEdit).toHaveBeenCalledWith('img1');
    expect(transaction.commit).toHaveBeenCalledOnce();
    expect(h.dispatched).toHaveLength(0);
    expect(h.createdIds).toHaveLength(0);
  });

  it.each([
    ['locked', { isLocked: true }, 'layer-locked'],
    ['hidden', { isEnabled: false }, 'layer-hidden'],
  ] as const)(
    'refuses to brush a %s image layer, offering a new paint layer instead of making one',
    (_, overrides, status) => {
      const transaction = controlTransaction('img1');
      const h = createHarness(
        makeDoc([{ ...imageLayer('img1'), ...overrides } as CanvasLayerContract], 'img1'),
        transaction
      );
      h.layers.getOrCreateRect('img1', { height: 10, width: 10, x: 0, y: 0 }).stale = false;
      const brush = createBrushTool();

      expect(brush.cursor?.(h.ctx)).toBe('not-allowed');
      down(brush, h.ctx, pointer(2, 2));
      up(brush, h.ctx, pointer(2, 2));

      expect(h.beginPixelEdit).not.toHaveBeenCalled();
      expect(h.dispatched).toEqual([]);
      expect(h.strokes).toEqual([]);
      expect(h.refusals).toEqual([{ canPaintNewLayer: true, layerId: 'img1', layerName: 'img1', status }]);
    }
  );

  it('says an image layer is still loading, asks for its pixels and paints once they are in', () => {
    const transaction = controlTransaction('img1');
    const h = createHarness(makeDoc([imageLayer('img1')], 'img1'), transaction);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(2, 2));
    up(brush, h.ctx, pointer(2, 2));
    expect(h.requestLayerRasterization).toHaveBeenCalledWith('img1');
    expect(h.refusals).toEqual([{ layerId: 'img1', layerName: 'img1', status: 'layer-loading' }]);
    expect(h.dispatched).toEqual([]);

    h.layers.getOrCreateRect('img1', { height: 10, width: 10, x: 0, y: 0 }).stale = false;
    down(brush, h.ctx, pointer(2, 2));
    up(brush, h.ctx, pointer(2, 2));
    expect(h.beginPixelEdit).toHaveBeenCalledWith('img1');
    expect(transaction.commit).toHaveBeenCalledOnce();
  });

  it('brushes a transparency-locked image layer in place with source-atop, but refuses to erase it', () => {
    const locked = { ...imageLayer('img1'), isTransparencyLocked: true } as CanvasLayerContract;
    const transaction = controlTransaction('img1');
    const h = createHarness(makeDoc([locked], 'img1'), transaction);
    h.layers.getOrCreateRect('img1', { height: 100, width: 100, x: 0, y: 0 }).stale = false;

    const eraser = createEraserTool();
    expect(eraser.cursor?.(h.ctx)).toBe('not-allowed');
    down(eraser, h.ctx, pointer(2, 2));
    up(eraser, h.ctx, pointer(2, 2));
    expect(h.beginPixelEdit).not.toHaveBeenCalled();
    expect(h.refusals).toEqual([{ layerId: 'img1', layerName: 'img1', status: 'layer-alpha-locked' }]);

    const brush = createBrushTool();
    down(brush, h.ctx, pointer(2, 2));
    move(brush, h.ctx, pointer(6, 6), [pointer(6, 6)]);
    expect(lastCompositeOp(cacheSurface(h, 'img1'))).toBe('source-atop');
    up(brush, h.ctx, pointer(6, 6));
    expect(transaction.commit).toHaveBeenCalledOnce();
    expect(h.dispatched).toHaveLength(0);
  });
});

describe('brush tool: cancel', () => {
  it('restores pixels via putImageData and emits no event', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    h.layers.getOrCreateRect('paint1', { height: 100, width: 100, x: 0, y: 0 }).stale = false;
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    cancel(brush, h.ctx);

    expect(h.strokes).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(cacheOps(cacheSurface(h, 'paint1'))).toContain('putImageData');
  });

  it('removes a cache the cancelled stroke created', () => {
    const h = createHarness(makeDoc([paintLayer('paint1')], 'paint1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    cancel(brush, h.ctx);

    expect(h.layers.peek('paint1')).toBeUndefined();
  });

  it('paints nothing and creates no layer when the stroke is refused admission', () => {
    const h = createHarness(makeDoc([], null), undefined, true);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40, { buttons: 0 }));

    expect(h.dispatched).toEqual([]);
    expect(h.createdIds).toEqual([]);
    expect(h.strokes).toEqual([]);
  });
});

describe('paint tool: target resolution', () => {
  it('commits a selected control stroke through its transaction without adding a raster layer', () => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(10, 10));
    move(brush, h.ctx, pointer(20, 20), [pointer(20, 20)]);
    up(brush, h.ctx, pointer(20, 20, { buttons: 0 }));

    expect(h.beginPixelEdit).toHaveBeenCalledWith('control');
    expect(transaction.commit).toHaveBeenCalledOnce();
    expect(transaction.cancel).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
  });

  it.each(['pointercancel', 'deactivate'] as const)('rolls back control preparation on %s', (ending) => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlImageLayer('control')], 'control'), transaction);
    h.layers.getOrCreateRect('control', { height: 20, width: 20, x: 0, y: 0 }).stale = false;
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    if (ending === 'pointercancel') {
      cancel(brush, h.ctx);
    } else {
      brush.onDeactivate?.(h.ctx);
    }
    expect(transaction.cancel).toHaveBeenCalledOnce();
    expect(transaction.commit).not.toHaveBeenCalled();
  });

  it('aborts the control transaction and releases the gesture after pointer-move painting fails', () => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    const drawImage = vi.spyOn(cacheSurface(h, 'control').ctx, 'drawImage').mockImplementation(() => {
      throw new Error('move paint failed');
    });

    expect(() => move(brush, h.ctx, pointer(20, 20), [pointer(20, 20)])).toThrow('move paint failed');

    expect(transaction.cancel).toHaveBeenCalledOnce();
    expect(transaction.commit).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    drawImage.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('aborts the gesture when a deferred frame render fails', () => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const frames: (() => void)[] = [];
    h.ctx.scheduleFrame = (task) => {
      frames.push(task);
      return () => frames.splice(frames.indexOf(task), 1);
    };
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    move(brush, h.ctx, pointer(20, 20), [pointer(20, 20)]);
    vi.spyOn(h.layers, 'growToRect').mockImplementation(() => {
      throw new Error('frame paint failed');
    });

    expect(() => frames.splice(0).forEach((task) => task())).toThrow('frame paint failed');

    expect(transaction.cancel).toHaveBeenCalledOnce();
    up(brush, h.ctx, pointer(30, 30));
    expect(transaction.commit).not.toHaveBeenCalled();
    expect(h.strokes).toHaveLength(0);
  });

  it('aborts the control transaction and releases the gesture after stroke finalization fails', () => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    const getImageData = vi.spyOn(cacheSurface(h, 'control').ctx, 'getImageData').mockImplementation(() => {
      throw new Error('stroke finalization failed');
    });

    expect(() => up(brush, h.ctx, pointer(10, 10, { buttons: 0 }))).toThrow('stroke finalization failed');

    expect(transaction.cancel).toHaveBeenCalledOnce();
    expect(transaction.commit).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    getImageData.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('still cancels the control transaction and releases the gesture when pixel restoration fails', () => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    const putImageData = vi.spyOn(cacheSurface(h, 'control').ctx, 'putImageData').mockImplementation(() => {
      throw new Error('pixel restoration failed');
    });

    expect(() => cancel(brush, h.ctx)).toThrow('pixel restoration failed');

    expect(transaction.cancel).toHaveBeenCalledOnce();
    expect(transaction.commit).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    putImageData.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('abandons the stroke when its edit fails to publish, leaving the tool ready', () => {
    const transaction = controlTransaction();
    vi.mocked(transaction.commit).mockImplementation(() => {
      throw new Error('publication failed');
    });
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));

    expect(() => up(brush, h.ctx, pointer(10, 10, { buttons: 0 }))).toThrow('publication failed');

    expect(transaction.commit).toHaveBeenCalledOnce();
    expect(transaction.cancel).toHaveBeenCalledOnce();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('does not fall through to raster auto-create when control preparation is rejected', () => {
    const h = createHarness(makeDoc([controlImageLayer('control')], 'control'), null);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    expect(h.dispatched).toHaveLength(0);
  });

  it('auto-creates a paint layer (one dispatch) when the selection is not paintable', () => {
    const doc = makeDoc([textLayer('img1')], 'img1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    up(brush, h.ctx, pointer(20, 20));

    expect(h.dispatched).toHaveLength(1);
    const action = h.dispatched[0]!;
    expect(action.type).toBe('addCanvasLayer');
    if (action.type === 'addCanvasLayer' && action.layer.type === 'raster') {
      expect(action.layer.source.type).toBe('paint');
      expect(action.layer.id).toBe(h.createdIds[0]);
    } else {
      throw new Error('expected an addCanvasLayer with a raster layer');
    }
    // The stroke commits against the freshly-created layer.
    expect(h.strokes).toHaveLength(1);
    expect(h.strokes[0]!.layerId).toBe(h.createdIds[0]);
  });

  it('auto-creates when nothing is selected', () => {
    const doc = makeDoc([paintLayer('paint1')], null);
    const h = createHarness(doc);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(20, 20));
    expect(h.dispatched).toHaveLength(1);
    expect(h.dispatched[0]!.type).toBe('addCanvasLayer');
  });

  it.each([
    ['locked', { isLocked: true }, 'layer-locked'],
    ['hidden', { isEnabled: false }, 'layer-hidden'],
  ] as const)('refuses a %s selected paint layer with a notice and no new layer', (_, opts, status) => {
    const doc = makeDoc([paintLayer('paint1', opts)], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    expect(brush.cursor?.(h.ctx)).toBe('not-allowed');
    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(h.dispatched).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);
    expect(h.refusals).toEqual([{ layerId: 'paint1', layerName: 'paint1', status }]);
  });

  it('shows the crosshair over a paintable selection', () => {
    const h = createHarness(makeDoc([paintLayer('paint1')], 'paint1'));
    expect(createBrushTool().cursor?.(h.ctx)).toBe('crosshair');
  });
});

describe('transparency lock', () => {
  it('brush composites source-atop on a transparency-locked layer (colour only on existing pixels)', () => {
    const doc = makeDoc([paintLayer('paint1', { isTransparencyLocked: true })], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(h.strokes).toHaveLength(1);
    expect(lastCompositeOp(cacheSurface(h, 'paint1'))).toBe('source-atop');
  });

  it('refuses the eraser entirely on a transparency-locked layer (no stroke, no alpha change)', () => {
    const doc = makeDoc([paintLayer('paint1', { isTransparencyLocked: true })], 'paint1');
    const h = createHarness(doc);
    const eraser = createEraserTool();

    down(eraser, h.ctx, pointer(20, 20));
    move(eraser, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(eraser, h.ctx, pointer(40, 40));

    expect(h.strokes).toHaveLength(0);
    expect(h.painted).toEqual([]);
    expect(h.strokeEdits.map((edit) => edit.cancel.mock.calls.length)).toEqual([1]);
    expect(h.refusals).toEqual([{ layerId: 'paint1', layerName: 'paint1', status: 'layer-alpha-locked' }]);
  });

  it('ends the admitted edit and removes the layer it created when the stroke is cancelled', () => {
    const h = createHarness(makeDoc([], null));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    cancel(brush, h.ctx);

    expect(h.strokeEdits.map((edit) => edit.cancel.mock.calls.length)).toEqual([1]);
    expect(h.dispatched).toContainEqual({ ids: [h.createdIds[0]], type: 'removeCanvasLayers' });
  });

  it('ends the admitted edit when a zero-scale layer leaves nothing to paint into', () => {
    const flat = { ...paintLayer('paint1'), transform: { rotation: 0, scaleX: 0, scaleY: 1, x: 0, y: 0 } };
    const h = createHarness(makeDoc([flat], 'paint1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    up(brush, h.ctx, pointer(20, 20));

    expect(h.strokes).toHaveLength(0);
    expect(h.strokeEdits.map((edit) => edit.cancel.mock.calls.length)).toEqual([1]);
  });

  it('brush is unaffected (source-over) when transparency is NOT locked', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(lastCompositeOp(cacheSurface(h, 'paint1'))).toBe('source-over');
  });
});

describe('mask strokes are forced opaque', () => {
  it('waits for a persisted mask cache instead of replacing its coverage', () => {
    const layer = inpaintMaskLayer('mask1');
    if (layer.type !== 'inpaint_mask') {
      throw new Error('expected an inpaint mask');
    }
    layer.mask.bitmap = { height: 10, imageName: 'persisted-mask', width: 10 };
    const h = createHarness(makeDoc([layer], layer.id));
    const eraser = createEraserTool();

    down(eraser, h.ctx, pointer(5, 5));
    up(eraser, h.ctx, pointer(5, 5, { buttons: 0 }));
    expect(h.strokes).toHaveLength(0);
    expect(h.layers.peek(layer.id)).toBeUndefined();
    expect(h.requestLayerRasterization).toHaveBeenCalledWith(layer.id);

    h.layers.getOrCreateRect(layer.id, { height: 10, width: 10, x: 0, y: 0 });
    h.layers.publishPixels(layer.id);
    down(eraser, h.ctx, pointer(5, 5));
    up(eraser, h.ctx, pointer(5, 5, { buttons: 0 }));
    expect(h.strokes).toHaveLength(1);
  });

  it('composites a mask stroke at globalAlpha 1 even when the brush opacity is 0.5', () => {
    const doc = makeDoc([inpaintMaskLayer('mask1')], 'mask1');
    const h = createHarness(doc);
    // Masks force full-alpha stencils even when brush opacity is one half.
    h.ctx.stores.brushOptions.set({ ...h.ctx.stores.brushOptions.get(), opacity: 0.5 });
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(h.strokes).toHaveLength(1);
    expect(h.strokes[0]!.layerId).toBe('mask1');
    expect(lastGlobalAlpha(cacheSurface(h, 'mask1'))).toBe(1);
  });

  it('still respects brush opacity on a normal raster paint layer', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    h.ctx.stores.brushOptions.set({ ...h.ctx.stores.brushOptions.get(), opacity: 0.5 });
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(lastGlobalAlpha(cacheSurface(h, 'paint1'))).toBe(0.5);
  });
});

describe('auto-created layer rollback (a gesture that commits nothing leaves no trace)', () => {
  /**
   * The auto-create dispatch happens at pointer-DOWN, outside history, so a gesture
   * producing no dirty rect must roll the layer back itself or strand it un-undoably.
   */
  const clipped = (doc: CanvasDocumentContractV3) => {
    const h = createHarness(doc);
    // Clip-to-bbox on with the stroke outside the frame, so `commit()` returns null.
    const ctx: ToolContext = { ...h.ctx, getStrokeClipRect: () => ({ height: 10, width: 10, x: 0, y: 0 }) };
    return { ...h, ctx };
  };

  const kinds = (h: { dispatched: CanvasProjectMutation[] }): string[] => h.dispatched.map((action) => action.type);

  it('removes the layer and restores the prior selection when the stroke is fully clipped away', () => {
    const h = clipped(makeDoc([textLayer('img1'), paintLayer('p1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(80, 80));
    move(brush, h.ctx, pointer(90, 90), [pointer(90, 90)]);
    up(brush, h.ctx, pointer(90, 90));

    expect(h.strokes).toHaveLength(0);
    expect(kinds(h)).toEqual(['addCanvasLayer', 'removeCanvasLayers', 'setCanvasSelectedLayer']);
    const removal = h.dispatched[1]!;
    expect(removal.type === 'removeCanvasLayers' && removal.ids).toEqual([h.createdIds[0]]);
    // The reducer's nearest-neighbour fallback would have picked the top layer.
    const reselect = h.dispatched[2]!;
    expect(reselect.type === 'setCanvasSelectedLayer' && reselect.id).toBe('img1');
  });

  it('drops the provisional cache entry too', () => {
    const h = clipped(makeDoc([textLayer('img1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(80, 80));
    expect(h.layers.peek(h.createdIds[0]!)).toBeDefined();
    up(brush, h.ctx, pointer(80, 80));

    expect(h.layers.peek(h.createdIds[0]!)).toBeUndefined();
  });

  it('rolls back on pointercancel mid-drag', () => {
    const h = createHarness(makeDoc([textLayer('img1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(30, 30), [pointer(30, 30)]);
    cancel(brush, h.ctx);

    expect(h.strokes).toHaveLength(0);
    expect(kinds(h)).toEqual(['addCanvasLayer', 'removeCanvasLayers', 'setCanvasSelectedLayer']);
    expect(h.layers.peek(h.createdIds[0]!)).toBeUndefined();
  });

  it('rolls back when the tool is switched away mid-drag', () => {
    const h = createHarness(makeDoc([textLayer('img1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    brush.onDeactivate?.(h.ctx, undefined);

    expect(kinds(h)).toEqual(['addCanvasLayer', 'removeCanvasLayers', 'setCanvasSelectedLayer']);
    expect(h.layers.peek(h.createdIds[0]!)).toBeUndefined();
  });

  it('does NOT roll back for a temporary tool switch (space/alt hold keeps the gesture)', () => {
    const h = createHarness(makeDoc([textLayer('img1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    brush.onDeactivate?.(h.ctx, { temporary: true });

    expect(kinds(h)).toEqual(['addCanvasLayer']);
    expect(h.layers.peek(h.createdIds[0]!)).toBeDefined();
  });

  it('restores a null selection when nothing was selected to begin with', () => {
    const h = clipped(makeDoc([paintLayer('p1')], null));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(80, 80));
    up(brush, h.ctx, pointer(80, 80));

    expect(kinds(h)).toEqual(['addCanvasLayer', 'removeCanvasLayers', 'setCanvasSelectedLayer']);
    const reselect = h.dispatched[2]!;
    expect(reselect.type === 'setCanvasSelectedLayer' && reselect.id).toBeNull();
  });

  it('keeps the layer when the stroke DOES commit pixels', () => {
    const h = createHarness(makeDoc([textLayer('img1')], 'img1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    up(brush, h.ctx, pointer(20, 20));

    expect(h.strokes).toHaveLength(1);
    expect(kinds(h)).toEqual(['addCanvasLayer']);
    expect(h.layers.peek(h.createdIds[0]!)).toBeDefined();
  });

  it('never rolls back a layer it did not create (an existing paint target)', () => {
    const h = clipped(makeDoc([paintLayer('p1')], 'p1'));
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(80, 80));
    up(brush, h.ctx, pointer(80, 80));

    expect(h.dispatched).toHaveLength(0);
  });
});

describe('Shift-click straight lines', () => {
  /**
   * A harness with an engine-owned line memory and a path recorder: `span()` is the horizontal extent of the stroke
   * outlines traced since the last call, so a dab and a line read apart regardless of cache growth chunks.
   */
  const lineHarness = (layers: CanvasLayerContract[], selectedLayerId: string | null) => {
    const h = createHarness(makeDoc(layers, selectedLayerId));
    const paintLineEnds = new Map<string, { x: number; y: number }>();
    let xs: number[] = [];
    const record = (x: number): void => {
      xs.push(x);
    };
    const ctx: ToolContext = {
      ...h.ctx,
      createPath2D: () =>
        ({
          closePath: () => {},
          lineTo: record,
          moveTo: record,
          quadraticCurveTo: (_cx: number, _cy: number, x: number) => record(x),
        }) as unknown as Path2D,
      paintLineEnds,
    };
    ctx.stores.brushOptions.set({ ...ctx.stores.brushOptions.get(), size: 4 });
    ctx.stores.eraserOptions.set({ ...ctx.stores.eraserOptions.get(), size: 4 });
    const span = (): number => {
      const extent = Math.max(...xs) - Math.min(...xs);
      xs = [];
      return extent;
    };
    return { ...h, ctx, paintLineEnds, span };
  };

  it('draws from the last stroke end to a Shift press, as one committed stroke', () => {
    const h = lineHarness([paintLayer('paint1')], 'paint1');
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(10, 10));
    up(brush, h.ctx, pointer(10, 10));
    expect(h.paintLineEnds.get('paint1')).toEqual({ x: 10, y: 10 });
    // A 4 px dab is about one brush wide.
    expect(h.span()).toBeLessThanOrEqual(5);

    down(brush, h.ctx, pointer(60, 10, { shift: true }));
    up(brush, h.ctx, pointer(60, 10, { shift: true }));

    expect(h.strokes).toHaveLength(2);
    // The line runs 50 px from x=10 to x=60, plus the brush radius at each end.
    expect(h.span()).toBeGreaterThanOrEqual(50);
    expect(h.paintLineEnds.get('paint1')).toEqual({ x: 60, y: 10 });
  });

  it('keeps the line start on the layer when the layer moves', () => {
    const layer = paintLayer('paint1');
    const h = lineHarness([layer], 'paint1');
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    up(brush, h.ctx, pointer(10, 10));
    h.span();

    // Nudge the layer 100 px right: its last stroke end now sits at document x=110.
    const nudged = makeDoc([{ ...layer, transform: { ...layer.transform, x: 100 } }], 'paint1');
    const ctx: ToolContext = { ...h.ctx, getDocument: () => nudged };
    down(brush, ctx, pointer(160, 10, { shift: true }));
    up(brush, ctx, pointer(160, 10, { shift: true }));

    // 50 px from x=110 to x=160 plus the brush radius at each end, not 150 px from where the end used to be.
    const span = h.span();
    expect(span).toBeGreaterThanOrEqual(50);
    expect(span).toBeLessThan(60);
    expect(h.paintLineEnds.get('paint1')).toEqual({ x: 60, y: 10 });
  });

  it('paints a dab on a Shift press without a remembered point, or on a layer with none', () => {
    const h = lineHarness([paintLayer('paint1'), paintLayer('paint2')], 'paint2');
    h.paintLineEnds.set('paint1', { x: 10, y: 10 });
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(60, 10, { shift: true }));
    up(brush, h.ctx, pointer(60, 10, { shift: true }));

    expect(h.strokes).toHaveLength(1);
    expect(h.span()).toBeLessThanOrEqual(5);
  });

  it('ignores the memory on a press without Shift', () => {
    const h = lineHarness([paintLayer('paint1')], 'paint1');
    h.paintLineEnds.set('paint1', { x: 10, y: 10 });
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(60, 10));
    up(brush, h.ctx, pointer(60, 10));

    expect(h.span()).toBeLessThanOrEqual(5);
  });

  it('shares one point per layer between brush and eraser', () => {
    const h = lineHarness([paintLayer('paint1')], 'paint1');
    h.layers.getOrCreateRect('paint1', { height: 100, width: 100, x: 0, y: 0 }).stale = false;
    const brush = createBrushTool();
    const eraser = createEraserTool();

    down(brush, h.ctx, pointer(10, 10));
    up(brush, h.ctx, pointer(10, 10));
    // Switching tools keeps the memory; only the engine's lifecycle events forget it.
    brush.onDeactivate?.(h.ctx, undefined);
    h.span();
    down(eraser, h.ctx, pointer(60, 10, { shift: true }));
    up(eraser, h.ctx, pointer(60, 10, { shift: true }));

    expect(h.strokes.map((stroke) => stroke.tool)).toEqual(['brush', 'eraser']);
    expect(h.span()).toBeGreaterThanOrEqual(50);
  });

  it('remembers the end of a stroke that auto-created its layer under the new id', () => {
    const h = lineHarness([textLayer('text1')], 'text1');
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    up(brush, h.ctx, pointer(25, 20));

    expect([...h.paintLineEnds]).toEqual([[h.createdIds[0], { x: 25, y: 20 }]]);
  });

  it('remembers nothing for a cancelled stroke', () => {
    const h = lineHarness([paintLayer('paint1')], 'paint1');
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    cancel(brush, h.ctx);

    expect(h.paintLineEnds.size).toBe(0);
  });
});
