import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ShapeToolKind } from '@workbench/canvas-engine/engineStores';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend, type StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createShapeTool } from './shapeTool';

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
    expect(h.commits[0]?.inverse).toEqual({ ids: ['shape-1'], type: 'removeCanvasLayers' });
    expect(h.previewOf()).toBeNull();
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
    expect(event.dirtyRect).toEqual({ height: 40, width: 60, x: 10, y: 10 });
    expect(event.tool).toBe('shape');
    expect(event.layerId).toBe('paint-1');
    expect(h.layers.get('paint-1')?.rect).toEqual({ height: 40, width: 60, x: 10, y: 10 });
    const surface = h.layers.get('paint-1')?.surface as StubRasterSurface;
    expect(surface.callLog.some((entry) => entry.op === 'drawImage')).toBe(true);
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

  it('refuses a locked paint layer instead of spawning a layer over it', () => {
    const h = createHarness(
      makeDoc({ stacks: stacksFrom([paintLayer({ isLocked: true })]), selectedLayerId: 'paint-1' })
    );
    const tool = createShapeTool();

    down(tool, h.ctx, pointer(10, 10));
    expect(h.refusals).toEqual([{ layerId: 'paint-1', layerName: 'Paint', status: 'layer-locked' }]);
    move(tool, h.ctx, pointer(70, 50));
    expect(h.previewOf()).toBeNull();
    up(tool, h.ctx, pointer(70, 50));

    expect(h.strokeEdit.record.commits).toHaveLength(0);
    expect(h.commits).toHaveLength(0);
    expect(h.refusals).toHaveLength(1);
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
      expect(h.assigned('fillStyle')).toContain('#ffffff');
      expect(h.assigned('strokeStyle')).toContain('#ffffff');
      expect([...h.assigned('fillStyle'), ...h.assigned('strokeStyle')]).not.toContain('#ff0000');
      expect(h.assigned('strokeStyle')).not.toContain('#00ff00');
    }
  );
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
});

describe('shape tool: the mod key cuts the shape out', () => {
  const createTool = () => createShapeTool({ isMacOs: false });

  const cutoutHarness = (layer: CanvasLayerContract, target: 'selected' | 'new' = 'selected') => {
    const h = createHarness(makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: layer.id }));
    h.stores.shapeOptions.set({ ...h.stores.shapeOptions.get(), target });
    h.layers.getOrCreateRect(layer.id, { height: 100, width: 100, x: 0, y: 0 }).stale = false;
    return h;
  };

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
    expect(h.assigned('fillStyle')).toContain('#ffffff');
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

    down(tool, h.ctx, pointer(40, 40));
    move(tool, h.ctx, pointer(70, 70));
    up(tool, h.ctx, pointer(70, 70, { ctrl: true }));
    expect(h.strokeEdit.record.commits).toHaveLength(1);
    expect(h.refusals).toHaveLength(0);
  });
});
