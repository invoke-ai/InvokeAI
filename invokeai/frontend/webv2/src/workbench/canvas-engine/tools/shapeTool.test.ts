import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ShapeToolKind } from '@workbench/canvas-engine/engineStores';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend, type StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createShapeTool, polygonShapeFrom } from './shapeTool';

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
  opts: { shift?: boolean; alt?: boolean; ctrl?: boolean; meta?: boolean; buttons?: number; timeStamp?: number } = {}
): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: opts.alt ?? false, ctrl: opts.ctrl ?? false, meta: opts.meta ?? false, shift: opts.shift ?? false },
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
  const refusals: CanvasEditRefusal[] = [];
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
    reportEditRefusal: (refusal) => refusals.push(refusal),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: identityViewport,
  };
  return {
    commits,
    refusals,
    ctx,
    dispatched,
    layers,
    previewOf: () => stores.shapePreview.get(),
    admission,
    stores,
    strokeEdit,
    /** Every value the tool assigned to `prop` on any surface it drew into. */
    assigned: (prop: string) =>
      surfaces.flatMap((surface) =>
        surface.callLog.filter((entry) => entry.op === 'set' && entry.args[0] === prop).map((entry) => entry.args[1])
      ),
  };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

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

  it.each(['busy', 'not-ready', 'over-budget'] as const)(
    'says why the release added no layer when the document refuses it as %s',
    (status) => {
      const h = createHarness(makeDoc());
      h.ctx.commitStructural = () => ({ status });
      const tool = createShapeTool();

      down(tool, h.ctx, pointer(10, 10));
      move(tool, h.ctx, pointer(70, 50));
      up(tool, h.ctx, pointer(70, 50));

      expect(h.refusals).toEqual([status]);
    }
  );

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
    // The closing press only arms the close: the commit waits for its release, outside the gesture.
    down(tool, h.ctx, pointer(13, 12, { timeStamp: 3000 }));
    expect(h.commits).toHaveLength(0);
    up(tool, h.ctx, pointer(13, 12, { timeStamp: 3000 }));

    expect(h.commits).toHaveLength(1);
    expect(h.stores.lassoPreview.get()).toBeNull();
  });

  it('closes on the release of a double-click, and a cancelled closing press closes nothing', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();

    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    click(tool, h.ctx, 50, 40, 2000);
    down(tool, h.ctx, pointer(50, 40, { timeStamp: 2100 }));
    tool.onPointerCancel?.(h.ctx);
    up(tool, h.ctx, pointer(50, 40, { timeStamp: 2100 }));
    expect(h.commits).toHaveLength(0);
    expect(h.stores.lassoPreview.get()).toBeNull();

    click(tool, h.ctx, 10, 10, 3000);
    click(tool, h.ctx, 50, 10, 4000);
    click(tool, h.ctx, 50, 40, 5000);
    down(tool, h.ctx, pointer(50, 40, { timeStamp: 5100 }));
    expect(h.commits).toHaveLength(0);
    up(tool, h.ctx, pointer(50, 40, { timeStamp: 5100 }));
    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]?.forward).toMatchObject({ layer: { source: { kind: 'polygon', width: 40 } } });
  });

  it('closes on the release when Enter comes during a held press', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();

    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    down(tool, h.ctx, pointer(50, 40, { timeStamp: 2000 }));
    tool.onKeyCommand?.(h.ctx, 'apply');
    expect(h.commits).toHaveLength(0);
    up(tool, h.ctx, pointer(50, 40, { timeStamp: 2000 }));

    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]?.forward).toMatchObject({ layer: { source: { height: 30, kind: 'polygon', width: 40 } } });
  });

  it('keeps the vertices when the closing press is refused, so the polygon can close once the layer is fixed', () => {
    let doc = makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' });
    const h = createHarness(doc);
    h.ctx.getDocument = () => doc;
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();
    doc = makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' });
    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    click(tool, h.ctx, 50, 40, 2000);

    doc = makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' });
    click(tool, h.ctx, 10, 10, 3000);
    expect(h.refusals).toEqual([expect.objectContaining({ status: 'layer-locked' })]);
    expect(h.stores.lassoPreview.get()).toMatchObject({ points: [{}, {}, {}] });

    doc = makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' });
    click(tool, h.ctx, 10, 10, 4000);
    expect(h.strokeEdit.record.commits).toHaveLength(1);
  });

  it.each([
    ['Enter', false],
    ['Enter during a held press', true],
  ])('keeps the vertices when a close by %s is refused, closing once the layer is fixed', (_label, held) => {
    let doc = makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' });
    const h = createHarness(doc);
    h.ctx.getDocument = () => doc;
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createShapeTool();
    click(tool, h.ctx, 10, 10, 0);
    click(tool, h.ctx, 50, 10, 1000);
    click(tool, h.ctx, 50, 40, 2000);
    /** Enter, alone or between a later press and its release (that press placing a fourth vertex). */
    const enter = (timeStamp: number) => {
      if (held) {
        down(tool, h.ctx, pointer(30, 60, { timeStamp }));
      }
      tool.onKeyCommand?.(h.ctx, 'apply');
      if (held) {
        up(tool, h.ctx, pointer(30, 60, { timeStamp }));
      }
    };

    doc = makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' });
    enter(3000);
    expect(h.refusals).toEqual([expect.objectContaining({ status: 'layer-locked' })]);
    expect(h.stores.lassoPreview.get()).toMatchObject({ kind: 'polygon' });

    doc = makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' });
    tool.onKeyCommand?.(h.ctx, 'apply');
    expect(h.strokeEdit.record.commits).toHaveLength(1);
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
    // Said at the press, before anything is drawn; the refused press starts no preview.
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-locked' }]);
    move(tool, h.ctx, pointer(70, 50));
    expect(h.previewOf()).toBeNull();
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
    expect(h.refusals).toHaveLength(1);
  });

  it('says why the shape placed nothing when its layer is locked mid-drag', () => {
    let doc = makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' });
    const h = createHarness(doc);
    h.ctx.getDocument = () => doc;
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    doc = makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' });
    up(tool, h.ctx, pointer(70, 50));

    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-locked' }]);
    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
  });

  it('waits for a durable paint layer whose pixels are not cached yet', () => {
    const durable = paintLayer({ source: { bitmap: { height: 8, imageName: 'b', width: 8 }, type: 'paint' } });
    const h = createHarness(makeDoc({ stacks: stacksFrom([durable]), selectedLayerId: 'paint-1' }));
    h.ctx.requestLayerRasterization = vi.fn();
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    expect(h.ctx.requestLayerRasterization).toHaveBeenCalledWith('paint-1');
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-loading' }]);
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);

    // Once the pixels are cached, the next press draws.
    h.layers.getOrCreateRect('paint-1', { height: 8, width: 8, x: 0, y: 0 }).stale = false;
    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50));
    expect(h.strokeEdit.record.commits).toHaveLength(1);
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

describe('shape tool: mask placement', () => {
  const drawKind = (tool: Tool, ctx: ToolContext, kind: ShapeToolKind): void => {
    if (kind === 'polygon') {
      for (const [x, y, timeStamp] of [
        [110, 60, 0],
        [170, 60, 1000],
        [170, 100, 2000],
      ] as const) {
        down(tool, ctx, pointer(x, y, { timeStamp }));
        up(tool, ctx, pointer(x, y, { timeStamp }));
      }
      tool.onKeyCommand?.(ctx, 'apply');
      return;
    }
    down(tool, ctx, pointer(110, 60));
    move(tool, ctx, pointer(170, 60));
    move(tool, ctx, pointer(170, 100));
    up(tool, ctx, pointer(kind === 'freehand' ? 110 : 170, 100));
  };

  const maskHarness = (mask: CanvasLayerContract, kind: ShapeToolKind = 'rect') => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([mask]), selectedLayerId: mask.id }));
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind, strokeEnabled: true });
    h.stores.colorPair.set({ background: '#00ff00', foreground: '#ff0000' });
    return h;
  };

  it.each(['rect', 'ellipse', 'polygon', 'freehand'] as const)(
    'paints a %s into the selected inpaint mask as opaque coverage, never a new layer',
    (kind) => {
      const h = maskHarness(layerContract('mask-1', 'inpaint_mask'), kind);
      const tool = createShapeTool();

      drawKind(tool, h.ctx, kind);

      expect(h.commits).toHaveLength(0);
      expect(h.dispatched).toHaveLength(0);
      expect(h.strokeEdit.record.commits).toHaveLength(1);
      const event = h.strokeEdit.record.commits[0]!;
      expect(event).toMatchObject({ layerId: 'mask-1', tool: 'shape' });
      expect(event.dirtyRect).toEqual({ height: 40, width: 60, x: 110, y: 60 });
      // Coverage is a stencil: the pair's colors never reach mask alpha.
      expect(h.assigned('fillStyle')).toContain('#ffffff');
      expect(h.assigned('strokeStyle')).toContain('#ffffff');
      expect([...h.assigned('fillStyle'), ...h.assigned('strokeStyle')]).not.toContain('#ff0000');
      expect(h.assigned('strokeStyle')).not.toContain('#00ff00');
    }
  );

  it('maps the shape through a transformed regional mask into its local space', () => {
    const mask = layerContract('region-1', 'regional_guidance', {
      transform: { rotation: 0, scaleX: 2, scaleY: 2, x: 100, y: 50 },
    });
    const h = maskHarness(mask);
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'rect');

    // Document 110..170 × 60..100 → local (5..35, 5..25) at half scale.
    expect(h.strokeEdit.record.commits[0]).toMatchObject({
      dirtyRect: { height: 20, width: 30, x: 5, y: 5 },
      layerId: 'region-1',
    });
    expect(h.commits).toHaveLength(0);
  });

  it('clips mask coverage to the selection and bbox like a brush stroke', () => {
    const h = maskHarness(layerContract('mask-1', 'inpaint_mask'));
    const selection = h.ctx.backend.createSurface(40, 40);
    h.ctx.getSelectionMask = () => ({ rect: { height: 40, width: 40, x: 100, y: 50 }, surface: selection });
    h.ctx.getStrokeClipRect = () => ({ height: 200, width: 130, x: 0, y: 0 });
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'rect');

    expect(h.strokeEdit.record.commits[0]!.dirtyRect).toEqual({ height: 30, width: 20, x: 110, y: 60 });
  });

  it.each([
    ['a locked mask', [layerContract('mask-1', 'inpaint_mask', { isLocked: true })]],
    ['a disabled mask', [layerContract('mask-1', 'inpaint_mask', { isEnabled: false })]],
    [
      'a mask under a disabled group',
      [groupContract('group-1', [layerContract('mask-1', 'inpaint_mask')], { isEnabled: false })],
    ],
    [
      'a mask under a locked group',
      [groupContract('group-1', [layerContract('mask-1', 'regional_guidance')], { isLocked: true })],
    ],
  ])('refuses %s without spawning a layer', (label, nodes) => {
    const h = createHarness(makeDoc({ stacks: stacksFrom(nodes), selectedLayerId: 'mask-1' }));
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'rect');

    expect(h.refusals).toEqual([
      expect.objectContaining({
        layerId: 'mask-1',
        status: label.includes('disabled') ? 'layer-hidden' : 'layer-locked',
      }),
    ]);
    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.layers.peek('mask-1')).toBeUndefined();
  });

  it('refuses a mask whose admission is refused without spawning a layer', () => {
    const h = maskHarness(layerContract('mask-1', 'inpaint_mask'));
    h.admission.refuse = true;
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'ellipse');

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
  });

  it('waits for a persisted mask whose alpha is not cached yet', () => {
    const mask = layerContract('mask-1', 'inpaint_mask', {
      mask: { bitmap: { height: 8, imageName: 'm', width: 8 }, fill: { color: '#e07575', style: 'diagonal' } },
    } as Partial<CanvasLayerContract>);
    const h = maskHarness(mask);
    h.ctx.requestLayerRasterization = vi.fn();
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'rect');

    expect(h.ctx.requestLayerRasterization).toHaveBeenCalledWith('mask-1');
    expect(h.refusals).toEqual([expect.objectContaining({ layerId: 'mask-1', status: 'layer-loading' })]);
    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
  });

  it('still creates a shape layer over a mask when the target is a new layer', () => {
    const h = maskHarness(layerContract('mask-1', 'inpaint_mask'));
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), target: 'new' });
    const tool = createShapeTool();

    drawKind(tool, h.ctx, 'rect');

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    const forward = h.commits[0]?.forward;
    expect(forward?.type).toBe('addCanvasLayer');
    if (forward?.type === 'addCanvasLayer' && forward.layer.type === 'raster') {
      expect(forward.layer.source).toMatchObject({ fill: '#ff0000', stroke: '#00ff00', type: 'shape' });
    }
  });

  it.each([
    ['fill and stroke off', { fillEnabled: false, strokeEnabled: false }],
    ['fill off and a zero-width stroke', { fillEnabled: false, strokeEnabled: true, strokeWidth: 0 }],
  ])('records nothing for a shape with %s, on a paint layer or a mask', (_label, options) => {
    for (const target of [paintLayer(), layerContract('mask-1', 'inpaint_mask')]) {
      const h = createHarness(makeDoc({ stacks: stacksFrom([target]), selectedLayerId: target.id }));
      h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), ...options });
      const tool = createShapeTool();

      drawKind(tool, h.ctx, 'rect');

      expect(h.strokeEdit.record.commits).toHaveLength(0);
      expect(h.strokeEdit.record.admitted).toBe(0);
      expect(h.commits).toHaveLength(0);
      expect(h.layers.peek(target.id)).toBeUndefined();
    }
  });
});

describe('shape tool: Alt draws from the center', () => {
  const dragWith = (modifiers: { alt?: boolean; shift?: boolean }) => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(50, 50, modifiers));
    move(tool, h.ctx, pointer(60, 70, modifiers));
    const preview = h.previewOf();
    up(tool, h.ctx, pointer(60, 70, modifiers));
    return { forward: h.commits[0]?.forward, preview };
  };

  it('grows the box from the press point under Alt, previewing what it commits', () => {
    const { forward, preview } = dragWith({ alt: true });
    expect(preview?.rect).toEqual({ height: 40, width: 20, x: 40, y: 30 });
    expect(forward).toMatchObject({
      layer: { source: { height: 40, width: 20 }, transform: { x: 40, y: 30 } },
      type: 'addCanvasLayer',
    });
  });

  it('makes a square about the press point under Shift+Alt', () => {
    const { forward } = dragWith({ alt: true, shift: true });
    expect(forward).toMatchObject({ layer: { source: { height: 40, width: 40 }, transform: { x: 30, y: 30 } } });
  });

  it('reads Alt live: let go before the release, the box goes back to a corner drag', () => {
    const h = createHarness(makeDoc());
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(50, 50, { alt: true }));
    move(tool, h.ctx, pointer(60, 70, { alt: true }));
    up(tool, h.ctx, pointer(60, 70));
    expect(h.commits[0]?.forward).toMatchObject({
      layer: { source: { height: 20, width: 10 }, transform: { x: 50, y: 50 } },
    });
  });

  it('leaves a freehand trace as drawn', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'freehand' });
    const tool = createShapeTool();
    down(tool, h.ctx, pointer(50, 50, { alt: true }));
    move(tool, h.ctx, pointer(80, 50, { alt: true }));
    up(tool, h.ctx, pointer(80, 80, { alt: true }));
    expect(h.commits[0]?.forward).toMatchObject({
      layer: { source: { height: 30, width: 30 }, transform: { x: 50, y: 50 } },
    });
  });
});

describe('shape tool: the mod key cuts the shape out', () => {
  /** Off macOS, so Ctrl is the mod key; the platform cases below cover ⌘. */
  const createTool = () => createShapeTool({ isMacOs: false });

  /** A selected layer with cached pixels over 0..100², so a cutout has something to erase. */
  const cutoutHarness = (layer: CanvasLayerContract, target: 'selected' | 'new' = 'selected') => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: layer.id }));
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), target });
    h.layers.getOrCreateRect(layer.id, { height: 100, width: 100, x: 0, y: 0 }).stale = false;
    return h;
  };

  /** The composite operations the tool set on the layer's own cache surface. */
  const layerComposites = (h: ReturnType<typeof createHarness>, layerId: string): unknown[] =>
    (h.layers.peek(layerId)!.surface as StubRasterSurface).callLog
      .filter((entry) => entry.op === 'set' && entry.args[0] === 'globalCompositeOperation')
      .map((entry) => entry.args[1]);

  it.each([
    ['Ctrl off macOS', false, { ctrl: true }],
    ['⌘ on macOS', true, { meta: true }],
  ])('erases from the selected paint layer with %s at the release, whatever Place on says', (_l, isMacOs, modifier) => {
    const h = cutoutHarness(paintLayer(), 'new');
    const tool = createShapeTool({ isMacOs });

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50, modifier));

    expect(h.commits).toHaveLength(0);
    expect(h.strokeEdit.record.commits).toHaveLength(1);
    expect(h.strokeEdit.record.commits[0]).toMatchObject({
      dirtyRect: { height: 40, width: 60, x: 10, y: 10 },
      layerId: 'paint-1',
      tool: 'shape-cutout',
    });
    expect(layerComposites(h, 'paint-1')).toContain('destination-out');
    // The footprint is opaque coverage, not the pair's colors.
    expect(h.assigned('fillStyle')).toContain('#ffffff');
  });

  it.each([
    ['Ctrl on macOS, which opens the context menu there', true, { ctrl: true }],
    ['⌘ off macOS', false, { meta: true }],
  ])('draws rather than cuts with %s', (_label, isMacOs, modifier) => {
    const h = cutoutHarness(paintLayer());
    const tool = createShapeTool({ isMacOs });

    down(tool, h.ctx, pointer(10, 10, modifier));
    move(tool, h.ctx, pointer(70, 50, modifier));
    up(tool, h.ctx, pointer(70, 50, modifier));

    expect(h.strokeEdit.record.commits[0]?.tool).toBe('shape');
    expect(tool.cursor?.(h.ctx)).toBe('crosshair');
  });

  it('erases mask coverage from a selected mask', () => {
    const h = cutoutHarness(layerContract('mask-1', 'inpaint_mask'));
    const tool = createTool();

    down(tool, h.ctx, pointer(10, 10, { ctrl: true }));
    move(tool, h.ctx, pointer(70, 50, { ctrl: true }));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));

    expect(h.strokeEdit.record.commits[0]).toMatchObject({ layerId: 'mask-1', tool: 'shape-cutout' });
    expect(layerComposites(h, 'mask-1')).toContain('destination-out');
    expect(h.commits).toHaveLength(0);
  });

  it('erases only cached pixels, never growing the layer', () => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([paintLayer()]), selectedLayerId: 'paint-1' }));
    h.layers.getOrCreateRect('paint-1', { height: 20, width: 20, x: 0, y: 0 }).stale = false;
    const tool = createTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));
    expect(h.strokeEdit.record.commits[0]?.dirtyRect).toEqual({ height: 10, width: 10, x: 10, y: 10 });
    expect(h.layers.peek('paint-1')?.rect).toEqual({ height: 20, width: 20, x: 0, y: 0 });

    // A shape wholly past the pixels erases nothing and records nothing.
    down(tool, h.ctx, pointer(40, 40));
    move(tool, h.ctx, pointer(70, 70));
    up(tool, h.ctx, pointer(70, 70, { ctrl: true }));
    expect(h.strokeEdit.record.commits).toHaveLength(1);
    expect(h.refusals).toHaveLength(0);
  });

  it('clips the cutout to the selection and bbox like a shape drawn there', () => {
    const h = cutoutHarness(paintLayer());
    const selection = h.ctx.backend.createSurface(40, 40);
    h.ctx.getSelectionMask = () => ({ rect: { height: 40, width: 40, x: 0, y: 0 }, surface: selection });
    h.ctx.getStrokeClipRect = () => ({ height: 30, width: 30, x: 0, y: 0 });
    const tool = createTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));

    expect(h.strokeEdit.record.commits[0]?.dirtyRect).toEqual({ height: 20, width: 20, x: 10, y: 10 });
  });

  it('cuts a click-closed polygon out under the modifiers of its closing press', () => {
    const h = cutoutHarness(paintLayer(), 'new');
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), kind: 'polygon' });
    const tool = createTool();
    for (const [x, y, timeStamp] of [
      [10, 10, 0],
      [50, 10, 1000],
      [50, 40, 2000],
    ] as const) {
      down(tool, h.ctx, pointer(x, y, { timeStamp }));
      up(tool, h.ctx, pointer(x, y, { timeStamp }));
    }
    down(tool, h.ctx, pointer(10, 10, { ctrl: true, timeStamp: 3000 }));
    up(tool, h.ctx, pointer(10, 10, { timeStamp: 3000 }));

    expect(h.commits).toHaveLength(0);
    expect(h.strokeEdit.record.commits[0]).toMatchObject({ layerId: 'paint-1', tool: 'shape-cutout' });
  });

  it.each([
    ['nothing is selected', makeDoc()],
    [
      'the selection is an image layer',
      makeDoc({
        stacks: stacksFrom([
          paintLayer({ id: 'image-1', source: { image: { height: 8, imageName: 'i', width: 8 }, type: 'image' } }),
        ]),
        selectedLayerId: 'image-1',
      }),
    ],
  ])('refuses, drawing nothing and adding no layer, when %s', (_label, doc) => {
    const h = createHarness(doc);
    const tool = createTool();

    // Known at the press: the refused press starts nothing.
    down(tool, h.ctx, pointer(10, 10, { ctrl: true }));
    expect(h.refusals).toEqual([{ status: 'cutout-no-target' }]);
    move(tool, h.ctx, pointer(70, 50, { ctrl: true }));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));

    // Learned only at the release: the shape still places nothing.
    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));

    expect(h.refusals).toEqual([{ status: 'cutout-no-target' }, { status: 'cutout-no-target' }]);
    expect(h.commits).toHaveLength(0);
    expect(h.strokeEdit.record.commits).toHaveLength(0);
  });

  it('refuses a transparency-locked paint layer like the eraser', () => {
    const h = cutoutHarness(paintLayer({ isTransparencyLocked: true } as Partial<CanvasLayerContract>));
    const tool = createTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(70, 50));
    up(tool, h.ctx, pointer(70, 50, { ctrl: true }));

    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-alpha-locked' }]);
    expect(h.strokeEdit.record.commits).toHaveLength(0);
  });

  it('cues the cutout on the cursor while the mod key is held, and not-allowed when it would be refused', () => {
    const h = cutoutHarness(paintLayer());
    const tool = createTool();
    expect(tool.cursor?.(h.ctx)).toBe('crosshair');

    move(tool, h.ctx, pointer(10, 10, { ctrl: true }));
    expect(h.ctx.updateCursor).toHaveBeenCalledOnce();
    expect(tool.cursor?.(h.ctx)).toMatch(/^url\("data:image\/svg\+xml,.*, crosshair$/);

    move(tool, h.ctx, pointer(12, 10));
    expect(h.ctx.updateCursor).toHaveBeenCalledTimes(2);
    expect(tool.cursor?.(h.ctx)).toBe('crosshair');

    const empty = createHarness(makeDoc());
    move(tool, empty.ctx, pointer(10, 10, { ctrl: true }));
    expect(tool.cursor?.(empty.ctx)).toBe('not-allowed');
  });
});
