import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { createShapeTool, rectFromDrag } from './shapeTool';

const makeDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 96, width: 96, x: 0, y: 0 },
  height: 512,
  layers: [],
  selectedLayerId: null,
  version: 2,
  width: 512,
});

const identityViewport = {
  documentToScreen: (p: Vec2): Vec2 => ({ x: p.x, y: p.y }),
} as unknown as Viewport;

const pointer = (x: number, y: number, opts: { shift?: boolean; buttons?: number } = {}): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: opts.shift ?? false },
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

const createHarness = (doc: CanvasDocumentContractV2) => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const stores = createEngineStores();
  let idCounter = 0;
  const ctx: ToolContext = {
    backend: null as never,
    commitStructural: (label, forward, inverse) => commits.push({ forward, inverse, label }),
    createLayerId: () => `shape-${++idCounter}`,
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
  return { commits, ctx, dispatched, previewOf: () => stores.shapePreview.get(), stores };
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

  it('uses the shape options kind/fill/stroke for the created layer', () => {
    const h = createHarness(makeDoc());
    h.stores.shapeOptions.set({ fill: '#ff0000', kind: 'ellipse', stroke: '#0000ff', strokeWidth: 4 });
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
