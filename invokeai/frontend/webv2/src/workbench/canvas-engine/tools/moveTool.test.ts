import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { FloatingSelection, FloatLiftResult } from '@workbench/canvas-engine/selection/floatingSelection';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestInsertionAnchorCapture } from '@workbench/canvas-engine/document/insertionAnchors.testStub';
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
  opts: { shift?: boolean; alt?: boolean; buttons?: number } = {}
): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: opts.alt ?? false, ctrl: false, meta: false, shift: opts.shift ?? false },
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
  float: { current: FloatingSelection | null };
  lifts: string[];
  commitFloat: ReturnType<typeof vi.fn>;
  transforms: LayerTransform[];
  refusals: CanvasEditRefusal[];
  selectedLayerEdits: ReturnType<typeof vi.fn>;
}

interface HarnessOptions {
  selectedLayerIds?: readonly string[];
  selectionContainsPoint?: boolean;
  lift?: FloatLiftResult;
  existingFloat?: boolean;
  snapToGrid?: boolean;
  bboxGrid?: number;
}

const IDENTITY: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

const fakeFloat = (layerId: string): FloatingSelection =>
  ({ layerId, transform: { ...IDENTITY } }) as FloatingSelection;

const createHarness = (doc: CanvasDocumentContractV3, options: HarnessOptions = {}): Harness => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const overrides: { layerId: string; override: { x: number; y: number } | null }[] = [];
  const transforms: LayerTransform[] = [];
  const lifts: string[] = [];
  const float: { current: FloatingSelection | null } = {
    current: options.existingFloat ? fakeFloat(doc.selectedLayerId ?? 'a') : null,
  };
  const commitFloat = vi.fn();
  const refusals: CanvasEditRefusal[] = [];
  const selectedLayerEdits = vi.fn();
  const stores = createEngineStores();
  stores.snapToGrid.set(options.snapToGrid ?? false);
  if (options.bboxGrid !== undefined) {
    stores.bboxGrid.set(options.bboxGrid);
  }
  const ctx: ToolContext = {
    scheduleFrame: () => () => undefined,
    backend: null as never,
    commitFloatingSelection: commitFloat,
    commitStructural: (label, forward, inverse) => {
      commits.push({ forward, inverse, label });
      return { status: 'committed' as const };
    },
    captureInsertionAnchor: createTestInsertionAnchorCapture('p'),
    createLayerId: () => 'x',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: (action) => dispatched.push(action),
    beginSelectedLayerEdit: selectedLayerEdits,
    beginStrokeEdit: () => null,
    getDocument: () => doc,
    getSelectedLayerIds: () => options.selectedLayerIds ?? (doc.selectedLayerId ? [doc.selectedLayerId] : []),
    getFloatingSelection: () => float.current,
    invalidate: vi.fn(),
    isPointInSelection: () => options.selectionContainsPoint === true,
    layers: createLayerCacheStore(createTestStubRasterBackend()),
    liftFloatingSelection: (layerId) => {
      const result = options.lift ?? 'lifted';
      lifts.push(layerId);
      if (result === 'lifted') {
        float.current = fakeFloat(layerId);
      }
      return result;
    },
    notifyLayerPainted: vi.fn(),
    reportEditRefusal: (refusal) => refusals.push(refusal),
    setFloatingTransform: (transform) => {
      transforms.push(transform);
      if (float.current) {
        float.current.transform = transform;
      }
    },
    setLayerTransformOverride: (layerId, override) => overrides.push({ layerId, override }),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    viewport: null as never,
  };
  return { commitFloat, commits, ctx, dispatched, float, lifts, overrides, refusals, selectedLayerEdits, transforms };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);
const cancel = (t: Tool, ctx: ToolContext): void => t.onPointerCancel?.(ctx);

describe('constrainDelta', () => {
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
});

describe('move tool: drag', () => {
  it('previews and commits every selected layer with one shared snapped delta', () => {
    const doc = makeDoc(
      [imageLayer('a', { x: 0, y: 0 }), imageLayer('middle', { x: 50, y: 50 }), imageLayer('b', { x: 7, y: 11 })],
      'a'
    );
    const h = createHarness(doc, { bboxGrid: 8, selectedLayerIds: ['a', 'b'], snapToGrid: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(25, 25));
    up(tool, h.ctx, pointer(25, 25));

    expect(h.overrides).toEqual([
      { layerId: 'a', override: { x: 16, y: 16 } },
      { layerId: 'b', override: { x: 23, y: 27 } },
      { layerId: 'a', override: null },
      { layerId: 'b', override: null },
    ]);
    expect(h.commits).toEqual([
      {
        forward: {
          type: 'setCanvasLayerPositions',
          updates: [
            { id: 'a', x: 16, y: 16 },
            { id: 'b', x: 23, y: 27 },
          ],
        },
        inverse: {
          type: 'setCanvasLayerPositions',
          updates: [
            { id: 'a', x: 0, y: 0 },
            { id: 'b', x: 7, y: 11 },
          ],
        },
        label: 'Move layers',
      },
    ]);
  });

  it('moves none of a selected group when any selected layer is locked', () => {
    const doc = makeDoc([imageLayer('a'), imageLayer('b', { isLocked: true })], 'a');
    const h = createHarness(doc, { selectedLayerIds: ['a', 'b'] });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 30));
    up(tool, h.ctx, pointer(30, 30));

    expect(h.overrides).toEqual([]);
    expect(h.commits).toEqual([]);
    expect(h.refusals).toEqual([{ layerId: 'b', layerName: 'b', status: 'layer-locked' }]);
  });

  it('previews via override on move then commits one structural transform on up', () => {
    const doc = makeDoc([imageLayer('a', { x: 0, y: 0, width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    up(tool, h.ctx, pointer(30, 25));

    expect(h.overrides).toEqual([
      { layerId: 'a', override: { x: 20, y: 15 } },
      { layerId: 'a', override: null },
    ]);
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

  it('moves the SELECTED layer even when another layer covers the press point', () => {
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
    expect(h.refusals).toEqual([{ layerId: 'locked', layerName: 'locked', status: 'layer-locked' }]);
  });

  it('commits nothing for a sub-threshold press+release (treated as a click)', () => {
    const doc = makeDoc([imageLayer('a', { width: 50, height: 50 })], 'a');
    const h = createHarness(doc);
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(11, 11));
    up(tool, h.ctx, pointer(11, 11));

    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
  });
});

describe('move tool: editing the selected layer in place', () => {
  const press = (h: Harness, tool: Tool, path: readonly number[]): number[] => {
    h.selectedLayerEdits.mockClear();
    down(tool, h.ctx, pointer(10, 10));
    const reported = path.map((x) => {
      const before = h.selectedLayerEdits.mock.calls.length;
      move(tool, h.ctx, pointer(x, 10));
      return h.selectedLayerEdits.mock.calls.length - before;
    });
    up(tool, h.ctx, pointer(path.at(-1)!, 10));
    return reported;
  };

  it('reports a layer drag once, as it crosses the threshold, and never a click', () => {
    const h = createHarness(makeDoc([imageLayer('a')], 'a'));
    const tool = createMoveTool();
    expect(press(h, tool, [11, 12])).toEqual([0, 0]);
    expect(press(h, tool, [11, 30, 35])).toEqual([0, 1, 0]);
  });
});

describe('move tool: grid snapping', () => {
  const snapDoc = () => makeDoc([imageLayer('a', { x: 0, y: 0, width: 50, height: 50 })], 'a');

  it('snaps the committed position to the grid, and previews the same value', () => {
    const h = createHarness(snapDoc(), { bboxGrid: 8, snapToGrid: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(31, 25)); // raw delta (21, 15) → snapped (24, 16)
    up(tool, h.ctx, pointer(31, 25));

    expect(h.overrides).toEqual([
      { layerId: 'a', override: { x: 24, y: 16 } },
      { layerId: 'a', override: null },
    ]);
    expect(h.commits[0]!.forward).toEqual({
      id: 'a',
      patch: { transform: { x: 24, y: 16 } },
      type: 'updateCanvasLayer',
    });
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
    expect(h.overrides.at(-1)).toEqual({ layerId: 'a', override: null });

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
    expect(h.transforms.at(-1)).toMatchObject({ x: 20, y: 15 });
    expect(h.commits).toHaveLength(0);
    expect(h.overrides).toHaveLength(0);
  });

  it('moves nothing when the lift is refused, never the whole layer', () => {
    const h = createHarness(doc(), { lift: 'refused', selectionContainsPoint: true });
    const tool = createMoveTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(30, 25));
    move(tool, h.ctx, pointer(40, 30));
    up(tool, h.ctx, pointer(40, 30));

    expect(h.lifts).toEqual(['a']);
    expect(h.commits).toHaveLength(0);
    expect(h.overrides).toHaveLength(0);
    expect(h.transforms).toHaveLength(0);
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
});
