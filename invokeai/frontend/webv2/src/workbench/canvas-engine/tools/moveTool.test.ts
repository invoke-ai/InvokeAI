import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

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
  /** The fake floating selection this harness hands the tool, if any. */
  float: { current: FloatingSelection | null };
  lifts: string[];
  commitFloat: ReturnType<typeof vi.fn>;
  transforms: LayerTransform[];
}

interface HarnessOptions {
  /** Whether a live selection contains every point (the ants are "under" the cursor). */
  selectionContainsPoint?: boolean;
  /** Whether a lift succeeds when the tool asks for one. */
  canLift?: boolean;
  /** A float that is already in flight before the gesture starts. */
  existingFloat?: boolean;
}

const IDENTITY: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

/** A minimal stand-in for a lifted float — only the fields the move tool reads. */
const fakeFloat = (layerId: string): FloatingSelection =>
  ({ layerId, transform: { ...IDENTITY } }) as FloatingSelection;

const createHarness = (doc: CanvasDocumentContractV2, options: HarnessOptions = {}): Harness => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const overrides: { layerId: string; override: { x: number; y: number } | null }[] = [];
  const transforms: LayerTransform[] = [];
  const lifts: string[] = [];
  const float: { current: FloatingSelection | null } = {
    current: options.existingFloat ? fakeFloat(doc.selectedLayerId ?? 'a') : null,
  };
  const commitFloat = vi.fn();
  const ctx: ToolContext = {
    backend: null as never,
    commitFloatingSelection: commitFloat,
    commitStructural: (label, forward, inverse) => commits.push({ forward, inverse, label }),
    createLayerId: () => 'x',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    emitStrokeCommitted: vi.fn(),
    getDocument: () => doc,
    getFloatingSelection: () => float.current,
    invalidate: vi.fn(),
    isPointInSelection: () => options.selectionContainsPoint === true,
    // A real (empty) layer cache so the tool's cache seam resolves; no test here
    // relies on cached content, so entries stay absent.
    layers: createLayerCacheStore(createTestStubRasterBackend()),
    liftFloatingSelection: (layerId) => {
      if (options.canLift === false) {
        return false;
      }
      lifts.push(layerId);
      float.current = fakeFloat(layerId);
      return true;
    },
    notifyLayerPainted: vi.fn(),
    setFloatingTransform: (transform) => {
      transforms.push(transform);
      if (float.current) {
        float.current.transform = transform;
      }
    },
    setLayerTransformOverride: (layerId, override) => overrides.push({ layerId, override }),
    setOverlayCursor: vi.fn(),
    stores: createEngineStores(),
    updateCursor: vi.fn(),
    viewport: null as never,
  };
  return { commitFloat, commits, ctx, dispatched, float, lifts, overrides, transforms };
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

describe('move tool: the layers panel owns selection', () => {
  it('does not select the layer under the pointer on click', () => {
    const doc = makeDoc(
      [imageLayer('top', { width: 50, height: 50 }), imageLayer('bottom', { width: 50, height: 50 })],
      null
    );
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
  });

  it('does not clear the selection when clicking empty space', () => {
    const doc = makeDoc([imageLayer('a', { width: 10, height: 10 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(80, 80));
    up(tool, h.ctx, pointer(80, 80));

    expect(h.dispatched).toHaveLength(0);
  });

  it('does not steal the selection when pressing another layer above the selected one', () => {
    // The exact complaint this contract exists to fix: 'bottom' is selected in
    // the panel and 'top' covers the press point.
    const doc = makeDoc(
      [imageLayer('top', { width: 50, height: 50 }), imageLayer('bottom', { width: 50, height: 50 })],
      'bottom'
    );
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    up(tool, h.ctx, pointer(10, 10));

    expect(h.dispatched).toHaveLength(0);
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

  it('moves the SELECTED layer even when another layer covers the press point', () => {
    // 'top' is composited over the press point, but 'bottom' is what the panel
    // selected — the panel wins, and no selection dispatch is emitted.
    const doc = makeDoc(
      [imageLayer('top', { width: 50, height: 50 }), imageLayer('bottom', { width: 50, height: 50 })],
      'bottom'
    );
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 10));
    up(tool, h.ctx, pointer(30, 10));

    expect(h.dispatched).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toMatchObject({ id: 'bottom' });
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

  it('does not drag a locked selected layer', () => {
    const doc = makeDoc([imageLayer('locked', { width: 50, height: 50, isLocked: true })], 'locked');
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
    expect(h.dispatched).toHaveLength(0);
  });

  it('commits nothing on a zero-delta drag, only clearing the preview', () => {
    const doc = makeDoc([imageLayer('a', { width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(20, 20)); // past the threshold
    up(tool, h.ctx, pointer(10, 10)); // ...and back to the origin

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.overrides.at(-1)).toEqual({ layerId: 'a', override: null });
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

describe('move tool: dragging a floating selection', () => {
  const doc = () => makeDoc([imageLayer('a', { x: 0, y: 0, width: 50, height: 50 })], 'a');

  it('lifts the selected pixels when the press lands inside the ants', () => {
    const h = createHarness(doc(), { selectionContainsPoint: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.lifts).toEqual(['a']);
    // The float moved, and the LAYER did not.
    expect(h.transforms.at(-1)).toMatchObject({ x: 20, y: 15 });
    expect(h.commits).toHaveLength(0);
    expect(h.overrides).toHaveLength(0);
  });

  it('moves the layer when the press lands outside the ants', () => {
    const h = createHarness(doc(), { selectionContainsPoint: false });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.lifts).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.forward).toMatchObject({ id: 'a' });
  });

  it('moves the layer when there is no selection at all', () => {
    const h = createHarness(doc());
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.lifts).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
  });

  it('drags an existing float even when the press is outside the ants', () => {
    // Once pixels are in flight they stay grabbable — the ants have moved with
    // them, and re-targeting the layer mid-move would be a trap.
    const h = createHarness(doc(), { existingFloat: true, selectionContainsPoint: false });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 10));
    up(tool, h.ctx, pointer(40, 10));

    expect(h.lifts).toHaveLength(0);
    expect(h.transforms.at(-1)).toMatchObject({ x: 30, y: 0 });
    expect(h.commits).toHaveLength(0);
  });

  it('accumulates across drags, each starting from the float current position', () => {
    const h = createHarness(doc(), { selectionContainsPoint: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 10));
    up(tool, h.ctx, pointer(30, 10));
    expect(h.transforms.at(-1)).toMatchObject({ x: 20, y: 0 });

    down(tool, h.ctx, pointer(30, 10));
    move(tool, h.ctx, pointer(45, 10));
    up(tool, h.ctx, pointer(45, 10));
    expect(h.transforms.at(-1)).toMatchObject({ x: 35, y: 0 });
  });

  it('falls back to moving the layer when the lift finds nothing to take', () => {
    const h = createHarness(doc(), { canLift: false, selectionContainsPoint: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.commits).toHaveLength(1);
  });

  it('constrains a float drag to the dominant axis under shift', () => {
    const h = createHarness(doc(), { selectionContainsPoint: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 15, { shift: true }));
    up(tool, h.ctx, pointer(40, 15, { shift: true }));

    expect(h.transforms.at(-1)).toMatchObject({ x: 30, y: 0 });
  });

  it('reverts only the current drag on pointercancel, keeping the float', () => {
    const h = createHarness(doc(), { existingFloat: true });
    const tool = createMoveTool();
    h.float.current!.transform = { rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 4 };

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 10));
    cancel(tool, h.ctx);

    expect(h.transforms.at(-1)).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 12, y: 4 });
    expect(h.float.current).not.toBeNull();
  });

  it('banks the float on Enter', () => {
    const h = createHarness(doc(), { existingFloat: true });
    const tool = createMoveTool();

    tool.onKeyCommand?.(h.ctx, 'apply');

    expect(h.commitFloat).toHaveBeenCalledTimes(1);
  });

  it('ignores Enter mid-drag, so a held pointer is never torn out from under', () => {
    const h = createHarness(doc(), { existingFloat: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 10));
    tool.onKeyCommand?.(h.ctx, 'apply');

    expect(h.commitFloat).not.toHaveBeenCalled();
  });
});
