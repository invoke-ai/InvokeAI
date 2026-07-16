import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { constrainDelta, createMoveTool } from './moveTool';

const imageLayer = (
  id: string,
  opts: { x?: number; y?: number; width?: number; height?: number; isLocked?: boolean; isEnabled?: boolean } = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: opts.isEnabled ?? true,
  isLocked: opts.isLocked ?? false,
  name: id,
  opacity: 1,
  source: { image: { height: opts.height ?? 40, imageName: id, width: opts.width ?? 40 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: opts.x ?? 0, y: opts.y ?? 0 },
  type: 'raster',
});

const shapeLayer = (
  id: string,
  opts: { x?: number; y?: number; width?: number; height?: number } = {}
): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: {
    fill: '#ffffff',
    height: opts.height ?? 40,
    kind: 'rect',
    stroke: null,
    strokeWidth: 0,
    type: 'shape',
    width: opts.width ?? 40,
  },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: opts.x ?? 0, y: opts.y ?? 0 },
  type: 'raster',
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

interface Harness {
  ctx: ToolContext;
  dispatched: CanvasProjectMutation[];
  commits: StructuralCommit[];
  overrides: { layerId: string; override: { x: number; y: number } | null }[];
}

const createHarness = (doc: CanvasDocumentContractV2): Harness => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const overrides: { layerId: string; override: { x: number; y: number } | null }[] = [];
  const ctx: ToolContext = {
    backend: null as never,
    commitStructural: (label, forward, inverse) => commits.push({ forward, inverse, label }),
    createLayerId: () => 'x',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    emitStrokeCommitted: vi.fn(),
    getDocument: () => doc,
    invalidate: vi.fn(),
    // A real (empty) layer cache so the move tool's live-cache-rect hit-test seam
    // resolves; no test here relies on cached content, so entries stay absent.
    layers: createLayerCacheStore(createTestStubRasterBackend()),
    notifyLayerPainted: vi.fn(),
    setLayerTransformOverride: (layerId, override) => overrides.push({ layerId, override }),
    setOverlayCursor: vi.fn(),
    stores: createEngineStores(),
    updateCursor: vi.fn(),
    viewport: null as never,
  };
  return { commits, ctx, dispatched, overrides };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);
const cancel = (t: Tool, ctx: ToolContext): void => t.onPointerCancel?.(ctx);

describe('constrainDelta', () => {
  it('passes the raw delta through without shift', () => {
    expect(constrainDelta(3, 7, false)).toEqual({ x: 3, y: 7 });
  });
  it('zeroes the smaller axis under shift', () => {
    expect(constrainDelta(10, 3, true)).toEqual({ x: 10, y: 0 });
    expect(constrainDelta(2, -8, true)).toEqual({ x: 0, y: -8 });
  });
});

describe('move tool: click selection', () => {
  it('selects the top-most visible layer under the point (one dispatch, no commit)', () => {
    const doc = makeDoc(
      [imageLayer('top', { width: 50, height: 50 }), imageLayer('bottom', { width: 50, height: 50 })],
      null
    );
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toEqual([{ id: 'top', type: 'setCanvasSelectedLayer' }]);
  });

  it('clears the selection when clicking empty space', () => {
    const doc = makeDoc([imageLayer('a', { width: 10, height: 10 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(80, 80));
    up(tool, h.ctx, pointer(80, 80));

    expect(h.dispatched).toEqual([{ id: null, type: 'setCanvasSelectedLayer' }]);
  });

  it('can click-select a locked layer', () => {
    const doc = makeDoc([imageLayer('locked', { width: 50, height: 50, isLocked: true })], null);
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));

    expect(h.dispatched).toEqual([{ id: 'locked', type: 'setCanvasSelectedLayer' }]);
  });

  it('does not click-select a hidden layer', () => {
    const doc = makeDoc([imageLayer('hidden', { width: 50, height: 50, isEnabled: false })], null);
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));

    // Hidden layer is not hit → empty-space clear.
    expect(h.dispatched).toEqual([{ id: null, type: 'setCanvasSelectedLayer' }]);
  });
});

describe('move tool: drag', () => {
  it('previews via override on move then commits one structural transform on up', () => {
    const doc = makeDoc([imageLayer('a', { x: 0, y: 0, width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    // Preview override applied during the move, cleared on commit.
    expect(h.overrides).toEqual([
      { layerId: 'a', override: { x: 20, y: 15 } },
      { layerId: 'a', override: null },
    ]);
    // Exactly one structural commit; the target was already selected → no extra dispatch.
    expect(h.dispatched).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toEqual({
      id: 'a',
      patch: { transform: { x: 20, y: 15 } },
      type: 'updateCanvasLayer',
    });
    expect(h.commits[0]!.inverse).toEqual({
      id: 'a',
      patch: { transform: { x: 0, y: 0 } },
      type: 'updateCanvasLayer',
    });
  });

  it('drags a parametric shape layer, committing one structural transform', () => {
    // Regression: shape/gradient/text layers were not hit-testable, so the move
    // tool skipped them (Phase 5 "param for parametric" hit-testing).
    const doc = makeDoc([shapeLayer('s', { x: 0, y: 0, width: 50, height: 50 })], 's');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toEqual({
      id: 's',
      patch: { transform: { x: 20, y: 15 } },
      type: 'updateCanvasLayer',
    });
  });

  it('auto-selects the pressed unlocked layer before committing the move', () => {
    const doc = makeDoc(
      [imageLayer('top', { width: 50, height: 50 }), imageLayer('bottom', { width: 50, height: 50 })],
      'bottom'
    );
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 10));
    up(tool, h.ctx, pointer(30, 10));

    // Selection switched to the pressed layer, then the move committed on it.
    expect(h.dispatched).toEqual([{ id: 'top', type: 'setCanvasSelectedLayer' }]);
    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toMatchObject({ id: 'top' });
  });

  it('moves the selected layer when the press lands on empty space', () => {
    const doc = makeDoc([imageLayer('a', { x: 0, y: 0, width: 10, height: 10 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    // Press at empty (80,80): no layer hit there, but 'a' is the selected movable layer.
    down(tool, h.ctx, pointer(80, 80));
    move(tool, h.ctx, pointer(90, 85));
    up(tool, h.ctx, pointer(90, 85));

    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toEqual({
      id: 'a',
      patch: { transform: { x: 10, y: 5 } },
      type: 'updateCanvasLayer',
    });
  });

  it('does not drag a locked layer (auto-select finds nothing, no selected fallback)', () => {
    const doc = makeDoc([imageLayer('locked', { width: 50, height: 50, isLocked: true })], null);
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 40));
    up(tool, h.ctx, pointer(40, 40));

    expect(h.commits).toHaveLength(0);
    expect(h.overrides).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
  });

  it('constrains to the dominant axis under shift', () => {
    const doc = makeDoc([imageLayer('a', { x: 0, y: 0, width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 15, { shift: true }));
    up(tool, h.ctx, pointer(40, 15, { shift: true }));

    expect(h.commits[0]!.forward).toEqual({
      id: 'a',
      patch: { transform: { x: 30, y: 0 } },
      type: 'updateCanvasLayer',
    });
  });

  it('commits nothing for a sub-threshold press+release (treated as a click)', () => {
    const doc = makeDoc([imageLayer('a', { width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(11, 11)); // < 3px screen
    up(tool, h.ctx, pointer(11, 11));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toEqual([{ id: 'a', type: 'setCanvasSelectedLayer' }]);
  });
});

describe('move tool: cancel', () => {
  it('drops the override and commits nothing on Esc mid-drag', () => {
    const doc = makeDoc([imageLayer('a', { width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 40));
    cancel(tool, h.ctx);

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    // Last override op clears the preview.
    expect(h.overrides.at(-1)).toEqual({ layerId: 'a', override: null });

    // A subsequent up (from the released pointer) does nothing.
    up(tool, h.ctx, pointer(40, 40));
    expect(h.commits).toHaveLength(0);
  });
});
