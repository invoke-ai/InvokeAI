import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend, type StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createShapeTool, polygonShapeFrom, rectFromDrag } from './shapeTool';

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

const makeDoc = (over: Partial<CanvasDocumentContractV3> = {}): CanvasDocumentContractV3 => ({
  background: 'transparent',
  bbox: { height: 96, width: 96, x: 0, y: 0 },
  height: 512,
  stacks: stacksFrom([]),
  selectedLayerId: null,
  version: 3,
  width: 512,
  ...over,
});

const identityViewport = {
  documentToScreen: (p: Vec2): Vec2 => ({ x: p.x, y: p.y }),
} as unknown as Viewport;

const pointer = (
  x: number,
  y: number,
  opts: { shift?: boolean; buttons?: number; timeStamp?: number } = {}
): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: opts.shift ?? false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: opts.timeStamp ?? 0,
});

interface StructuralCommit {
  label: string;
  forward: CanvasProjectMutation;
  inverse: CanvasProjectMutation;
}

const createHarness = (doc: CanvasDocumentContractV3) => {
  const strokeEdit = createRecordingStrokeEdit();
  const admission = { refuse: false };
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const stores = createEngineStores();
  const backend = createTestStubRasterBackend();
  const layers = createLayerCacheStore(backend);
  let idCounter = 0;
  const ctx: ToolContext = {
    scheduleFrame: () => () => undefined,
    backend,
    commitStructural: (label, forward, inverse) => {
      commits.push({ forward, inverse, label });
      return { status: 'committed' as const };
    },
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    createLayerId: () => `shape-${++idCounter}`,
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    beginStrokeEdit: () => (admission.refuse ? null : strokeEdit.edit),
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: identityViewport,
  };
  return {
    commits,
    ctx,
    dispatched,
    layers,
    previewOf: () => stores.shapePreview.get(),
    admission,
    stores,
    strokeEdit,
  };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('rectFromDrag', () => {
  it('normalizes a drag to a positive integer rect', () => {
    expect(rectFromDrag({ x: 50, y: 60 }, { x: 20, y: 100 }, false)).toEqual({
      height: 40,
      width: 30,
      x: 20,
      y: 60,
    });
  });

  it('constrains to a square using the larger dimension, preserving direction', () => {
    expect(rectFromDrag({ x: 0, y: 0 }, { x: 30, y: 80 }, true)).toEqual({ height: 80, width: 80, x: 0, y: 0 });
    // Dragging up-left: the square extends toward negative both axes.
    expect(rectFromDrag({ x: 100, y: 100 }, { x: 70, y: 20 }, true)).toEqual({
      height: 80,
      width: 80,
      x: 20,
      y: 20,
    });
  });
});

describe('shape tool: creation', () => {
  it('previews on move then commits ONE addCanvasLayer on up', () => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    expect(h.previewOf()).toEqual({ kind: 'rect', rect: { height: 40, width: 60, x: 10, y: 10 } });

    up(tool, h.ctx, pointer(70, 50));

    // Creation goes through commitStructural (undoable), never a bare dispatch.
    expect(h.dispatched).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    expect(forward?.type).toBe('addCanvasLayer');
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect(forward.layer.source).toEqual({
        fill: '#000000',
        height: 40,
        kind: 'rect',
        stroke: null,
        strokeWidth: 8,
        type: 'shape',
        width: 60,
      });
      expect(forward.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 10 });
    }
    // Inverse removes the created layer.
    expect(h.commits[0]?.inverse).toEqual({ ids: ['shape-1'], type: 'removeCanvasLayers' });
    expect(h.previewOf()).toBeNull();
  });

  it('resolves the pair at gesture start for the enabled fill and stroke', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({
      fillEnabled: true,
      kind: 'ellipse',
      strokeEnabled: true,
      strokeWidth: 4,
      target: 'new',
    });
    h.stores.colorPair.set({ background: '#0000ff', foreground: '#ff0000' });
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 100));
    up(tool, h.ctx, pointer(100, 100));

    const forward = h.commits[0]?.forward;
    if (
      forward?.type === 'addCanvasLayer' &&
      forward.layer.type === 'raster' &&
      forward.layer.source.type === 'shape'
    ) {
      expect(forward.layer.source.kind).toBe('ellipse');
      expect(forward.layer.source.fill).toBe('#ff0000');
      expect(forward.layer.source.stroke).toBe('#0000ff');
      expect(forward.layer.source.strokeWidth).toBe(4);
    } else {
      throw new Error('expected an ellipse shape layer');
    }
  });

  it('constrains to a square while shift is held', () => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(0, 0, { shift: true }));
    move(tool, h.ctx, pointer(30, 80, { shift: true }));
    up(tool, h.ctx, pointer(30, 80, { shift: true }));
    const forward = h.commits[0]?.forward;
    if (
      forward?.type === 'addCanvasLayer' &&
      forward.layer.type === 'raster' &&
      forward.layer.source.type === 'shape'
    ) {
      expect(forward.layer.source.width).toBe(80);
      expect(forward.layer.source.height).toBe(80);
    } else {
      throw new Error('expected a shape layer');
    }
  });

  it('commits nothing for a zero-area drag', () => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));
    expect(h.commits).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it('escape (cancel key command) drops the gesture without committing', () => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    expect(h.previewOf()).not.toBeNull();
    tool.onKeyCommand?.(h.ctx, 'cancel');
    expect(h.previewOf()).toBeNull();
    // A subsequent up does nothing (gesture already dropped).
    up(tool, h.ctx, pointer(70, 50));
    expect(h.commits).toHaveLength(0);
  });
});

describe('shape tool: polygon and freehand kinds', () => {
  const click = (tool: Tool, ctx: ToolContext, x: number, y: number, timeStamp: number): void => {
    down(tool, ctx, pointer(x, y, { timeStamp }));
    up(tool, ctx, pointer(x, y, { timeStamp }));
  };

  it('places polygon vertices click by click and commits one layer on close, points relative to its box', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();

    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    move(tool, h.ctx, pointer(50, 40));
    expect(h.stores.lassoPreview.get()).toMatchObject({
      kind: 'polygon',
      points: [
        { x: 10, y: 10 },
        { x: 50, y: 10 },
      ],
    });
    click(tool, h.ctx, 50, 40, 2000);
    expect(h.commits).toHaveLength(0);
    tool.onKeyCommand?.(h.ctx, 'apply');

    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect(forward.layer.source).toMatchObject({
        height: 30,
        kind: 'polygon',
        points: [
          { x: 0, y: 0 },
          { x: 40, y: 0 },
          { x: 40, y: 30 },
        ],
        width: 40,
      });
      expect(forward.layer.transform).toMatchObject({ x: 10, y: 10 });
    } else {
      throw new Error('expected a polygon layer');
    }
    expect(h.stores.lassoPreview.get()).toBeNull();
  });

  it('traces a freehand drag into a closed polygon', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'freehand' });
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(30, 0));
    move(tool, h.ctx, pointer(30, 30));
    expect(h.stores.lassoPreview.get()).toMatchObject({ kind: 'freehand' });
    up(tool, h.ctx, pointer(0, 30));

    const forward = h.commits[0]?.forward;
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect(forward.layer.source).toMatchObject({ height: 30, kind: 'polygon', width: 30 });
      expect((forward.layer.source as { points?: unknown[] }).points).toHaveLength(4);
    } else {
      throw new Error('expected a polygon layer');
    }
  });

  it('decimates a freehand trace by on-screen travel and keeps the release point', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'freehand' });
    const tool = createShapeTool();
    const zoomedOut = (x: number, y: number): PointerInput => ({
      ...pointer(x, y),
      screenPoint: { x: x / 10, y: y / 10 },
    });

    down(tool, h.ctx, zoomedOut(0, 0));
    // 10 document units are 1 CSS px at 0.1× zoom: decimated away.
    move(tool, h.ctx, zoomedOut(10, 0));
    move(tool, h.ctx, zoomedOut(30, 0));
    move(tool, h.ctx, zoomedOut(30, 30));
    expect(h.stores.lassoPreview.get()).toMatchObject({
      points: [
        { x: 0, y: 0 },
        { x: 30, y: 0 },
        { x: 30, y: 30 },
      ],
    });
    up(tool, h.ctx, zoomedOut(5, 30));

    const forward = h.commits[0]?.forward;
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect((forward.layer.source as { points?: unknown[] }).points).toHaveLength(4);
      expect(forward.layer.source).toMatchObject({ height: 30, width: 30 });
    } else {
      throw new Error('expected a polygon layer');
    }
  });

  it('commits nothing for a flat polygon', () => {
    // The third point survives the 1px dedupe; only the sub-pixel bounds reject it.
    expect(
      polygonShapeFrom(
        [
          { x: 0, y: 0 },
          { x: 40, y: 0 },
          { x: 20, y: 0.5 },
        ],
        { fill: '#000', stroke: null, strokeWidth: 0 }
      )
    ).toBeNull();
  });

  it('closes the polygon with a click on the first vertex', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();

    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    click(tool, h.ctx, 50, 40, 2000);
    move(tool, h.ctx, pointer(13, 12));
    expect(h.stores.lassoPreview.get()).toMatchObject({ closeArmed: true, kind: 'polygon' });
    expect(tool.cursor?.(h.ctx)).toBe('pointer');
    click(tool, h.ctx, 13, 12, 3000);

    expect(h.commits).toHaveLength(1);
    expect(h.stores.lassoPreview.get()).toBeNull();
  });

  it('drops an open polygon when the kind changes mid-session', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();

    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'rect' });
    move(tool, h.ctx, pointer(50, 40));
    expect(h.stores.lassoPreview.get()).toBeNull();

    // The next press starts a fresh rect drag rather than extending the polygon.
    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(20, 20));
    up(tool, h.ctx, pointer(20, 20));
    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect(forward.layer.source).toMatchObject({ kind: 'rect', width: 20 });
    } else {
      throw new Error('expected a rect layer');
    }
  });
});

describe('shape tool: placement', () => {
  it('draws onto the selected paint layer as one stroke event, growing its cache to the shape', () => {
    const layer = paintLayer({ transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 50 } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'paint-1' }));
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(110, 60));
    move(tool, h.ctx, pointer(170, 100));
    up(tool, h.ctx, pointer(170, 100));

    expect(h.commits).toHaveLength(0);
    expect(h.strokeEdit.record.commits).toHaveLength(1);
    const event = h.strokeEdit.record.commits[0]!;
    // Layer-local: the document rect minus the layer's offset.
    expect(event.dirtyRect).toEqual({ height: 40, width: 60, x: 10, y: 10 });
    expect(event.tool).toBe('shape');
    expect(event.layerId).toBe('paint-1');
    expect(h.layers.get('paint-1')?.rect).toEqual({ height: 40, width: 60, x: 10, y: 10 });
    const surface = h.layers.get('paint-1')?.surface as StubRasterSurface;
    expect(surface.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
  });

  it('leaves the paint layer untouched when the shape is refused admission', () => {
    const layer = paintLayer({ transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 50 } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'paint-1' }));
    h.admission.refuse = true;
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(110, 60));
    move(tool, h.ctx, pointer(170, 100));
    up(tool, h.ctx, pointer(170, 100));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
    expect(h.layers.peek('paint-1')).toBeUndefined();
  });

  it('restores the pixels and cache extent when the shape cannot be recorded', () => {
    const layer = paintLayer({ transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 50 } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'paint-1' }));
    h.layers.getOrCreateRect('paint-1', { height: 4, width: 4, x: 0, y: 0 }).stale = false;
    const cancel = vi.spyOn(h.strokeEdit.edit, 'cancel');
    h.strokeEdit.edit.commit = () => false;
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(110, 60));
    move(tool, h.ctx, pointer(170, 100));
    up(tool, h.ctx, pointer(170, 100));

    expect(h.layers.peek('paint-1')?.rect).toEqual({ height: 4, width: 4, x: 0, y: 0 });
    const writes = (h.layers.peek('paint-1')!.surface as StubRasterSurface).callLog.map((entry) => entry.op);
    expect(writes.lastIndexOf('putImageData')).toBeGreaterThan(writes.lastIndexOf('drawImage'));
    expect(cancel).toHaveBeenCalledOnce();
    expect(h.commits).toHaveLength(0);
  });

  it('maps the shape through a scaled layer and clamps the cache growth to the selection', () => {
    const layer = paintLayer({ transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 100, y: 50 } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'paint-1' }));
    const mask = h.ctx.backend.createSurface(40, 40);
    h.ctx.getSelectionMask = () => ({ rect: { height: 40, width: 40, x: 100, y: 50 }, surface: mask });
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(110, 60));
    move(tool, h.ctx, pointer(170, 100));
    up(tool, h.ctx, pointer(170, 100));

    const event = h.strokeEdit.record.commits[0]!;
    // Document 110..140 × 60..90 (the selection's extent) → local (5..20, 5..20) at half scale.
    expect(event.dirtyRect).toEqual({ height: 15, width: 15, x: 5, y: 5 });
    expect(h.layers.get('paint-1')?.rect).toEqual(event.dirtyRect);
  });

  it('clips to the bbox and skips a shape that lands entirely outside the selection', () => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' }));
    const mask = h.ctx.backend.createSurface(10, 10);
    h.ctx.getSelectionMask = () => ({ rect: { height: 10, width: 10, x: 0, y: 0 }, surface: mask });
    h.ctx.getStrokeClipRect = () => ({ height: 8, width: 8, x: 0, y: 0 });
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(4, 4));
    move(tool, h.ctx, pointer(60, 60));
    up(tool, h.ctx, pointer(60, 60));
    expect(h.strokeEdit.record.commits[0]!.dirtyRect).toEqual({
      height: 4,
      width: 4,
      x: 4,
      y: 4,
    });

    down(tool, h.ctx, pointer(20, 20));
    move(tool, h.ctx, pointer(60, 60));
    up(tool, h.ctx, pointer(60, 60));
    // Nothing to draw: no stroke event, no history, no new layer.
    expect(h.strokeEdit.record.commits).toHaveLength(1);
    expect(h.commits).toHaveLength(0);
    expect(h.layers.get('paint-1')?.rect).toEqual({ height: 4, width: 4, x: 4, y: 4 });
  });

  it('refuses a locked paint layer instead of spawning a layer over it', () => {
    const h = createHarness(
      makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' })
    );
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
  });

  it('waits for a durable paint layer whose pixels are not cached yet', () => {
    const durable = paintLayer({ source: { bitmap: { height: 8, imageName: 'b', width: 8 }, type: 'paint' } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([durable]), selectedLayerId: 'paint-1' }));
    h.ctx.requestLayerRasterization = vi.fn();
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));

    expect(h.ctx.requestLayerRasterization).toHaveBeenCalledWith('paint-1');
    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
  });

  it('falls back to a new shape layer when the selection is not a paint layer', () => {
    const image = paintLayer({
      id: 'image-1',
      source: { image: { height: 8, imageName: 'i', width: 8 }, type: 'image' },
    });
    const h = createHarness(makeDoc({ stacks: stacksFrom([image]), selectedLayerId: 'image-1' }));
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits[0]?.forward.type).toBe('addCanvasLayer');
  });

  it('creates a shape layer over a paint layer when the target is a new layer', () => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' }));
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), target: 'new' });
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits[0]?.forward.type).toBe('addCanvasLayer');
  });
});
