import type { StubRasterBackend, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type {
  ControlPixelEditTransaction,
  StrokeCommittedEvent,
  Tool,
  ToolContext,
} from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasControlLayerContract, CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

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

const controlTransaction = (): ControlPixelEditTransaction => ({
  cancel: vi.fn(),
  commitPatch: vi.fn(),
  commitStroke: vi.fn(),
  layerId: 'control',
});

const makeDoc = (layers: CanvasLayerContract[], selectedLayerId: string | null): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 100, width: 100, x: 0, y: 0 },
  height: 100,
  layers,
  selectedLayerId,
  version: 2,
  width: 100,
});

const pointer = (x: number, y: number, opts: { pressure?: number; buttons?: number } = {}): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
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
  beginControlPixelEdit: ReturnType<typeof vi.fn> | null;
  ctx: ToolContext;
  backend: StubRasterBackend;
  layers: LayerCacheStore;
  dispatched: CanvasProjectMutation[];
  strokes: StrokeCommittedEvent[];
  painted: string[];
  createdIds: string[];
}

const createHarness = (
  doc: CanvasDocumentContractV2,
  transaction: ControlPixelEditTransaction | null | undefined = undefined
): Harness => {
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  const stores = createEngineStores();
  const dispatched: CanvasProjectMutation[] = [];
  const strokes: StrokeCommittedEvent[] = [];
  const painted: string[] = [];
  const createdIds: string[] = [];
  const beginControlPixelEdit = transaction === undefined ? null : vi.fn(() => transaction);
  let idCounter = 0;

  const ctx: ToolContext = {
    backend,
    ...(beginControlPixelEdit ? { beginControlPixelEdit } : {}),
    commitStructural: vi.fn(),
    createLayerId: () => {
      const id = `new-layer-${(idCounter += 1)}`;
      createdIds.push(id);
      return id;
    },
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    emitStrokeCommitted: (event) => {
      painted.push(event.layerId);
      strokes.push(event);
    },
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: (layerId) => painted.push(layerId),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: null as never,
  };

  return { backend, beginControlPixelEdit, createdIds, ctx, dispatched, layers, painted, strokes };
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

    // Dirty rect is integer and non-empty. Content-sized: it is the stroke's TRUE
    // bounds (no document clamp), so it can extend past the document edges — a
    // paint layer grows with its strokes.
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
});

describe('brush tool: cancel', () => {
  it('restores pixels via putImageData and emits no event', () => {
    const doc = makeDoc([paintLayer('paint1')], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    cancel(brush, h.ctx);

    expect(h.strokes).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(cacheOps(cacheSurface(h, 'paint1'))).toContain('putImageData');
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

    expect(h.beginControlPixelEdit).toHaveBeenCalledWith('control');
    expect(transaction.commitStroke).toHaveBeenCalledOnce();
    expect(transaction.cancel).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
  });

  it.each(['pointercancel', 'deactivate'] as const)('rolls back control preparation on %s', (ending) => {
    const transaction = controlTransaction();
    const h = createHarness(makeDoc([controlImageLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    if (ending === 'pointercancel') {
      cancel(brush, h.ctx);
    } else {
      brush.onDeactivate?.(h.ctx);
    }
    expect(transaction.cancel).toHaveBeenCalledOnce();
    expect(transaction.commitStroke).not.toHaveBeenCalled();
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
    expect(transaction.commitStroke).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    drawImage.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginControlPixelEdit).toHaveBeenCalledTimes(2);
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
    expect(transaction.commitStroke).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    getImageData.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginControlPixelEdit).toHaveBeenCalledTimes(2);
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
    expect(transaction.commitStroke).not.toHaveBeenCalled();
    expect(h.dispatched).toHaveLength(0);
    expect(h.painted).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);

    putImageData.mockRestore();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginControlPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('does not roll back an accepted stroke when transaction publication throws', () => {
    const transaction = controlTransaction();
    vi.mocked(transaction.commitStroke).mockImplementation(() => {
      throw new Error('listener publication failed');
    });
    const h = createHarness(makeDoc([controlPaintLayer('control')], 'control'), transaction);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));

    expect(() => up(brush, h.ctx, pointer(10, 10, { buttons: 0 }))).toThrow('listener publication failed');

    expect(transaction.commitStroke).toHaveBeenCalledOnce();
    expect(transaction.cancel).not.toHaveBeenCalled();
    down(brush, h.ctx, pointer(30, 30));
    expect(h.beginControlPixelEdit).toHaveBeenCalledTimes(2);
  });

  it('does not fall through to raster auto-create when control preparation is rejected', () => {
    const h = createHarness(makeDoc([controlImageLayer('control')], 'control'), null);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(10, 10));
    expect(h.dispatched).toHaveLength(0);
  });

  it('auto-creates a paint layer (one dispatch) when the selection is not paintable', () => {
    const doc = makeDoc([imageLayer('img1')], 'img1');
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

  it('no-ops on a locked selected paint layer (no dispatch, no paint)', () => {
    const doc = makeDoc([paintLayer('paint1', { isLocked: true })], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();

    down(brush, h.ctx, pointer(20, 20));
    move(brush, h.ctx, pointer(40, 40), [pointer(40, 40)]);
    up(brush, h.ctx, pointer(40, 40));

    expect(h.dispatched).toHaveLength(0);
    expect(h.strokes).toHaveLength(0);
  });

  it('no-ops on a disabled selected paint layer', () => {
    const doc = makeDoc([paintLayer('paint1', { isEnabled: false })], 'paint1');
    const h = createHarness(doc);
    const brush = createBrushTool();
    down(brush, h.ctx, pointer(20, 20));
    up(brush, h.ctx, pointer(20, 20));
    expect(h.strokes).toHaveLength(0);
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
  it('composites a mask stroke at globalAlpha 1 even when the brush opacity is 0.5', () => {
    const doc = makeDoc([inpaintMaskLayer('mask1')], 'mask1');
    const h = createHarness(doc);
    // A half-opacity brush: on a raster layer this would land alpha ~128, but a
    // mask is an all-or-nothing alpha stencil and must be forced fully opaque.
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
