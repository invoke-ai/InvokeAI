import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { createGradientTool, placementFromDrag } from './gradientTool';

const gradientLayer = (over: Partial<CanvasLayerContract> = {}): CanvasLayerContract =>
  ({
    blendMode: 'normal',
    id: 'grad-existing',
    isEnabled: true,
    isLocked: false,
    name: 'Gradient',
    opacity: 1,
    source: {
      angle: 0,
      kind: 'linear',
      stops: [
        { color: '#000000', offset: 0 },
        { color: '#ffffff', offset: 1 },
      ],
      type: 'gradient',
    },
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

const pointer = (x: number, y: number, buttons = 1): PointerInput => ({
  buttons,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

interface StructuralCommit {
  label: string;
  forward: CanvasProjectMutation;
  inverse: CanvasProjectMutation;
}

const createHarness = (doc: CanvasDocumentContractV3) => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const stores = createEngineStores();
  let idCounter = 0;
  const ctx: ToolContext = {
    scheduleFrame: () => () => undefined,
    backend: null as never,
    commitStructural: (label, forward, inverse) => {
      commits.push({ forward, inverse, label });
      return { status: 'committed' as const };
    },
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    createLayerId: () => `grad-${++idCounter}`,
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    beginStrokeEdit: () => null,
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers: null as never,
    notifyLayerPainted: vi.fn(),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: identityViewport,
  };
  return { commits, ctx, dispatched, previewOf: () => stores.gradientPreview.get(), stores };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('placementFromDrag', () => {
  it('centers a linear ramp on the drag midpoint, as long as the drag, along its angle', () => {
    expect(placementFromDrag('linear', { x: 10, y: 20 }, { x: 10, y: 120 })).toEqual({
      angle: 90,
      center: { x: 10, y: 70 },
      span: 100,
    });
  });

  it('centers a radial gradient on the press point with the drag length as radius', () => {
    expect(placementFromDrag('radial', { x: 10, y: 20 }, { x: 40, y: 60 })).toEqual({
      angle: expect.any(Number),
      center: { x: 10, y: 20 },
      span: 50,
    });
  });

  it('never yields a zero span', () => {
    expect(placementFromDrag('linear', { x: 5, y: 5 }, { x: 5, y: 5 }).span).toBe(1);
  });
});

describe('gradient tool: create when no gradient selected', () => {
  it('creates a bbox-sized gradient layer placed where the drag went', () => {
    const h = createHarness(makeDoc({ bbox: { height: 96, width: 96, x: 100, y: 200 } }));
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(110, 210));
    move(tool, h.ctx, pointer(110, 310));
    expect(h.previewOf()).toEqual({ end: { x: 110, y: 310 }, kind: 'linear', start: { x: 110, y: 210 } });

    up(tool, h.ctx, pointer(110, 310));

    expect(h.dispatched).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    expect(forward?.type).toBe('addCanvasLayer');
    if (
      forward?.type === 'addCanvasLayer' &&
      forward.layer.type === 'raster' &&
      forward.layer.source.type === 'gradient'
    ) {
      expect(forward.layer.source.angle).toBeCloseTo(90);
      expect(forward.layer.source.kind).toBe('linear');
      // Layer-local: the drag minus the bbox origin the layer sits at.
      expect(forward.layer.source.center).toEqual({ x: 10, y: 60 });
      expect(forward.layer.source.span).toBe(100);
      expect(forward.layer.source.width).toBe(96);
      expect(forward.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 200 });
    } else {
      throw new Error('expected a gradient layer');
    }
    expect(h.commits[0]?.inverse).toEqual({ ids: ['grad-1'], type: 'removeCanvasLayers' });
    expect(h.previewOf()).toBeNull();
  });

  it('resolves the FG→BG pair preset at gesture start', () => {
    const h = createHarness(makeDoc());
    h.stores.colorPair.set({ background: '#0000ff', foreground: '#ff0000' });
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    const forward = h.commits[0]?.forward;
    if (
      forward?.type === 'addCanvasLayer' &&
      forward.layer.type === 'raster' &&
      forward.layer.source.type === 'gradient'
    ) {
      expect(forward.layer.source.stops).toEqual([
        { color: '#ff0000ff', offset: 0 },
        { color: '#0000ffff', offset: 1 },
      ]);
    } else {
      throw new Error('expected a gradient layer');
    }
  });

  it('uses the explicit custom stops verbatim, independent of the pair', () => {
    const h = createHarness(makeDoc());
    h.stores.colorPair.set({ background: '#0000ff', foreground: '#ff0000' });
    h.stores.gradientOptions.set({
      angle: 0,
      kind: 'linear',
      preset: 'custom',
      stops: [
        { color: '#11223344', offset: 0 },
        { color: '#55667788', offset: 1 },
      ],
    });
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    const forward = h.commits[0]?.forward;
    if (
      forward?.type === 'addCanvasLayer' &&
      forward.layer.type === 'raster' &&
      forward.layer.source.type === 'gradient'
    ) {
      expect(forward.layer.source.stops).toEqual([
        { color: '#11223344', offset: 0 },
        { color: '#55667788', offset: 1 },
      ]);
    } else {
      throw new Error('expected a gradient layer');
    }
  });
});

describe('gradient tool: edit selected gradient layer', () => {
  it('commits ONE updateCanvasLayerSource with the new placement (kind/stops preserved)', () => {
    const layer = gradientLayer();
    const doc = makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    const inverse = h.commits[0]?.inverse;
    expect(forward?.type).toBe('updateCanvasLayerSource');
    if (forward?.type === 'updateCanvasLayerSource' && forward.source.type === 'gradient') {
      expect(forward.id).toBe('grad-existing');
      expect(forward.source.angle).toBeCloseTo(0);
      expect(forward.source.center).toEqual({ x: 50, y: 0 });
      expect(forward.source.span).toBe(100);
      expect(forward.source.kind).toBe('linear');
      expect(forward.source.stops).toHaveLength(2);
    } else {
      throw new Error('expected an updateCanvasLayerSource gradient edit');
    }
    // Inverse restores the exact original source object.
    if (inverse?.type === 'updateCanvasLayerSource' && layer.type === 'raster') {
      expect(inverse.source).toBe(layer.source);
    } else {
      throw new Error('expected an inverse source restore');
    }
  });

  it('is a no-op when the selected gradient layer is locked', () => {
    const layer = gradientLayer({ isLocked: true });
    const doc = makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it("reads the drag in the layer's local space when it is moved, scaled and rotated", () => {
    // Local +x points down the document after a quarter turn: a downward
    // document drag of 200px is a 100px local drag along +x at scale 2.
    const layer = gradientLayer({ transform: { rotation: Math.PI / 2, scaleX: 2, scaleY: 2, x: 100, y: 100 } });
    const doc = makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(100, 100));
    move(tool, h.ctx, pointer(100, 300));
    up(tool, h.ctx, pointer(100, 300));

    const forward = h.commits[0]?.forward;
    if (forward?.type === 'updateCanvasLayerSource' && forward.source.type === 'gradient') {
      expect(forward.source.angle).toBeCloseTo(0);
      expect(forward.source.center?.x).toBeCloseTo(50);
      expect(forward.source.center?.y).toBeCloseTo(0);
      expect(forward.source.span).toBeCloseTo(100);
    } else {
      throw new Error('expected an updateCanvasLayerSource gradient edit');
    }
  });

  it('re-centers a selected radial gradient on the press point with the drag as its radius', () => {
    const layer = gradientLayer({
      source: {
        angle: 0,
        kind: 'radial',
        stops: [
          { color: '#000000', offset: 0 },
          { color: '#ffffff', offset: 1 },
        ],
        type: 'gradient',
      },
    } as Partial<CanvasLayerContract>);
    const doc = makeDoc({ stacks: stacksFrom([layer]), selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(20, 30));
    move(tool, h.ctx, pointer(50, 70));
    expect(h.previewOf()?.kind).toBe('radial');
    up(tool, h.ctx, pointer(50, 70));

    expect(h.commits).toHaveLength(1);
    const forward = h.commits[0]?.forward;
    if (forward?.type === 'updateCanvasLayerSource' && forward.source.type === 'gradient') {
      expect(forward.source.kind).toBe('radial');
      expect(forward.source.center).toEqual({ x: 20, y: 30 });
      expect(forward.source.span).toBe(50);
    } else {
      throw new Error('expected an updateCanvasLayerSource gradient edit');
    }
    expect(h.previewOf()).toBeNull();
  });

  it('creates a new gradient when the selected layer is not a gradient', () => {
    const paint = gradientLayer({
      id: 'paint-1',
      source: { bitmap: null, type: 'paint' },
    } as Partial<CanvasLayerContract>);
    const doc = makeDoc({ stacks: stacksFrom([paint]), selectedLayerId: 'paint-1' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(50, 50));
    up(tool, h.ctx, pointer(50, 50));

    expect(h.commits[0]?.forward.type).toBe('addCanvasLayer');
  });
});
