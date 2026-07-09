import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { angleFromDrag, createGradientTool } from './gradientTool';

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

const makeDoc = (over: Partial<CanvasDocumentContractV2> = {}): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 96, width: 96, x: 0, y: 0 },
  height: 512,
  layers: [],
  selectedLayerId: null,
  version: 2,
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
  forward: WorkbenchAction;
  inverse: WorkbenchAction;
}

const createHarness = (doc: CanvasDocumentContractV2) => {
  const dispatched: WorkbenchAction[] = [];
  const commits: StructuralCommit[] = [];
  const stores = createEngineStores();
  let idCounter = 0;
  const ctx: ToolContext = {
    backend: null as never,
    commitStructural: (label, forward, inverse) => commits.push({ forward, inverse, label }),
    createLayerId: () => `grad-${++idCounter}`,
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    emitStrokeCommitted: vi.fn(),
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

describe('angleFromDrag', () => {
  it('is 0° for a left→right drag and 90° for a top→bottom drag', () => {
    expect(angleFromDrag({ x: 0, y: 0 }, { x: 10, y: 0 })).toBeCloseTo(0);
    expect(angleFromDrag({ x: 0, y: 0 }, { x: 0, y: 10 })).toBeCloseTo(90);
  });
});

describe('gradient tool: create when no gradient selected', () => {
  it('creates a document-covering gradient layer with the drag angle', () => {
    const h = createHarness(makeDoc());
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(10, 110));
    expect(h.previewOf()).toEqual({ end: { x: 10, y: 110 }, start: { x: 10, y: 10 } });

    up(tool, h.ctx, pointer(10, 110));

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
      expect(forward.layer.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
    } else {
      throw new Error('expected a gradient layer');
    }
    expect(h.commits[0]?.inverse).toEqual({ ids: ['grad-1'], type: 'removeCanvasLayers' });
    expect(h.previewOf()).toBeNull();
  });
});

describe('gradient tool: edit selected gradient layer', () => {
  it('commits ONE updateCanvasLayerSource with the new angle (kind/stops preserved)', () => {
    const layer = gradientLayer();
    const doc = makeDoc({ layers: [layer], selectedLayerId: 'grad-existing' });
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
    const doc = makeDoc({ layers: [layer], selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it('is a no-op when the selected gradient layer is radial (angle has no visual effect)', () => {
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
    const doc = makeDoc({ layers: [layer], selectedLayerId: 'grad-existing' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(100, 0));
    up(tool, h.ctx, pointer(100, 0));

    // A radial gradient ignores `angle`, so dragging on one would only ever
    // produce an angle-only, visually-inert commit. Skipped entirely: no
    // commit, no dispatch, no dangling preview.
    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it('creates a new gradient when the selected layer is not a gradient', () => {
    const paint = gradientLayer({
      id: 'paint-1',
      source: { bitmap: null, type: 'paint' },
    } as Partial<CanvasLayerContract>);
    const doc = makeDoc({ layers: [paint], selectedLayerId: 'paint-1' });
    const h = createHarness(doc);
    const tool = createGradientTool();

    down(tool, h.ctx, pointer(0, 0));
    move(tool, h.ctx, pointer(50, 50));
    up(tool, h.ctx, pointer(50, 50));

    expect(h.commits[0]?.forward.type).toBe('addCanvasLayer');
  });
});
