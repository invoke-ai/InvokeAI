import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { TRANSFORM_ROTATE_NUB_PX, transformOverlayGeometry } from '@workbench/canvas-engine/transform/transformMath';
import { describe, expect, it, vi } from 'vitest';

import { createTransformTool } from './transformTool';

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
  source: { image: { height: opts.height ?? 100, imageName: id, width: opts.width ?? 100 }, type: 'image' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: opts.x ?? 0, y: opts.y ?? 0 },
  type: 'raster',
});

const makeDoc = (layers: CanvasLayerContract[], selectedLayerId: string | null): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height: 200, width: 200, x: 0, y: 0 },
  height: 200,
  layers,
  selectedLayerId,
  version: 2,
  width: 200,
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

/** A pointer input at an explicit SCREEN point, with the doc point derived at `zoom`. */
const pointerAtScreen = (screen: Vec2, zoom: number): PointerInput => ({
  buttons: 1,
  documentPoint: { x: screen.x / zoom, y: screen.y / zoom },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: screen,
  timeStamp: 0,
});

/** The screen-space tip of the drawn rotation nub for `transform` at `zoom`. */
const nubTipScreen = (
  transform: LayerTransform,
  sz: { x: number; y: number; width: number; height: number },
  zoom: number
): Vec2 => {
  const geo = transformOverlayGeometry(transform, sz);
  const toScreen = (p: Vec2) => ({ x: zoom * p.x, y: zoom * p.y });
  const a = toScreen(geo.rotationAnchor);
  const c = toScreen(geo.center);
  const dx = a.x - c.x;
  const dy = a.y - c.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: a.x + (dx / len) * TRANSFORM_ROTATE_NUB_PX, y: a.y + (dy / len) * TRANSFORM_ROTATE_NUB_PX };
};

/** Rotates `p` about `pivot` by `rad` (screen space). */
const rotateAbout = (p: Vec2, pivot: Vec2, rad: number): Vec2 => {
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const dx = p.x - pivot.x;
  const dy = p.y - pivot.y;
  return { x: pivot.x + cos * dx - sin * dy, y: pivot.y + sin * dx + cos * dy };
};

interface Harness {
  ctx: ToolContext;
  applyCount: () => number;
  session: () => ReturnType<ReturnType<typeof createEngineStores>['transformSession']['get']>;
  overrides: { layerId: string; override: unknown }[];
}

/**
 * A ToolContext whose transform-session seams mutate a real `transformSession`
 * store (mirroring the engine), so the tool's reads reflect its own writes across
 * a down→move→up drag. The viewport projects document→screen 1:1.
 */
const createHarness = (doc: CanvasDocumentContractV2, zoom = 1): Harness => {
  const stores = createEngineStores();
  const overrides: { layerId: string; override: unknown }[] = [];
  const state = { applyCount: 0 };

  const beginTransformSession = (layerId: string): void => {
    const layer = doc.layers.find((entry) => entry.id === layerId);
    if (!layer) {
      return;
    }
    const start: LayerTransform = { ...layer.transform };
    stores.transformSession.set({ layerId, startTransform: start, transform: start });
    overrides.push({ layerId, override: start });
  };
  const updateTransformSession = (transform: LayerTransform): void => {
    const session = stores.transformSession.get();
    if (!session) {
      return;
    }
    stores.transformSession.set({ ...session, transform });
    overrides.push({ layerId: session.layerId, override: transform });
  };
  const cancelTransform = (): void => {
    const session = stores.transformSession.get();
    if (session) {
      overrides.push({ layerId: session.layerId, override: null });
    }
    stores.transformSession.set(null);
  };

  const ctx: ToolContext = {
    applyTransform: () => {
      state.applyCount += 1;
    },
    backend: null as never,
    beginTransformSession,
    cancelTransform,
    commitStructural: vi.fn(),
    createLayerId: () => 'x',
    createPath2D: (d) => ({ d }) as unknown as Path2D,
    dispatch: vi.fn(),
    emitStrokeCommitted: vi.fn(),
    getDocument: () => doc,
    invalidate: vi.fn(),
    layers: null as never,
    notifyLayerPainted: vi.fn(),
    setLayerTransformOverride: vi.fn(),
    setOverlayCursor: vi.fn(),
    stores,
    updateCursor: vi.fn(),
    updateTransformSession,
    viewport: {
      documentToScreen: (p: Vec2) => ({ x: zoom * p.x, y: zoom * p.y }),
      screenToDocument: (p: Vec2) => ({ x: p.x / zoom, y: p.y / zoom }),
    } as never,
  };

  return {
    applyCount: () => state.applyCount,
    ctx,
    overrides,
    session: () => stores.transformSession.get(),
  };
};

const activate = (t: Tool, ctx: ToolContext): void => t.onActivate?.(ctx);
const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('transform tool: session lifecycle', () => {
  it('opens a session on the selected eligible layer when activated', () => {
    const doc = makeDoc([imageLayer('a')], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();

    activate(tool, h.ctx);

    const s = h.session();
    expect(s?.layerId).toBe('a');
    expect(s?.transform).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
  });

  it('opens no session on a locked selected layer', () => {
    const doc = makeDoc([imageLayer('a', { isLocked: true })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();

    activate(tool, h.ctx);

    expect(h.session()).toBeNull();
  });

  it('opens no session on a hidden selected layer', () => {
    const doc = makeDoc([imageLayer('a', { isEnabled: false })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();

    activate(tool, h.ctx);

    expect(h.session()).toBeNull();
  });

  it('clicking a layer with no session starts a session (move gesture)', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], null);
    const h = createHarness(doc);
    const tool = createTransformTool();

    down(tool, h.ctx, pointer(50, 50));
    expect(h.session()?.layerId).toBe('a');
  });
});

describe('transform tool: gestures', () => {
  it('scales via a corner handle drag (updates the session, no dispatch)', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    // se corner is at doc (100,100); drag it to (150,150).
    down(tool, h.ctx, pointer(100, 100));
    move(tool, h.ctx, pointer(150, 150));
    up(tool, h.ctx, pointer(150, 150));

    const s = h.session();
    expect(s?.transform.scaleX).toBeCloseTo(1.5, 5);
    expect(s?.transform.scaleY).toBeCloseTo(1.5, 5);
    // No structural dispatch — the session holds the preview until Apply.
    expect(h.ctx.commitStructural).not.toHaveBeenCalled();
  });

  it('moves via an interior drag', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    down(tool, h.ctx, pointer(50, 50));
    move(tool, h.ctx, pointer(70, 65));
    up(tool, h.ctx, pointer(70, 65));

    const s = h.session();
    expect(s?.transform.x).toBeCloseTo(20, 5);
    expect(s?.transform.y).toBeCloseTo(15, 5);
  });

  it('rotates via a corner rotate zone drag', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    // Just outside the se corner (100,100) is a rotate zone.
    down(tool, h.ctx, pointer(112, 112));
    move(tool, h.ctx, pointer(100, 130));
    up(tool, h.ctx, pointer(100, 130));

    const s = h.session();
    expect(s?.transform.rotation).not.toBe(0);
  });

  it('ignores a sub-threshold drag (no session change)', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);
    const startOverrides = h.overrides.length;

    down(tool, h.ctx, pointer(100, 100));
    move(tool, h.ctx, pointer(101, 101));
    up(tool, h.ctx, pointer(101, 101));

    // No update beyond the initial begin override.
    expect(h.overrides.length).toBe(startOverrides);
    expect(h.session()?.transform.scaleX).toBe(1);
  });
});

describe('transform tool: rotation nub (regression — a nub press must rotate, not reset)', () => {
  const transformed: LayerTransform = { rotation: 0.4, scaleX: 1.6, scaleY: 1.3, x: 30, y: 20 };
  const layerSize = { height: 100, width: 100, x: 0, y: 0 };

  const run = (zoom: number): void => {
    const layer: CanvasLayerContract = { ...imageLayer('a'), transform: transformed };
    const doc = makeDoc([layer], 'a');
    const h = createHarness(doc, zoom);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    const startTransform = h.session()?.transform;
    expect(startTransform).toEqual(transformed);
    const overridesBefore = h.overrides.length;

    // Pointer-down EXACTLY on the drawn rotation nub (above the top edge).
    const tip = nubTipScreen(transformed, layerSize, zoom);
    down(tool, h.ctx, pointerAtScreen(tip, zoom));

    // (a) Gesture start must NOT touch the session/override values. The bug read
    // the nub press as off-frame and re-opened the session, resetting its live
    // transform back to the committed one.
    expect(h.session()?.transform).toEqual(startTransform);
    expect(h.overrides.length).toBe(overridesBefore);

    // (b) A subsequent move begins a ROTATION: rotation changes by the swept
    // angle, scale is untouched (not a move/scale/reset).
    const geo = transformOverlayGeometry(transformed, layerSize);
    const centerScreen: Vec2 = { x: zoom * geo.center.x, y: zoom * geo.center.y };
    const moved = rotateAbout(tip, centerScreen, 0.5);
    move(tool, h.ctx, pointerAtScreen(moved, zoom));

    const after = h.session()?.transform;
    expect(after).toBeDefined();
    expect(after?.scaleX).toBeCloseTo(transformed.scaleX, 6);
    expect(after?.scaleY).toBeCloseTo(transformed.scaleY, 6);
    expect(after?.rotation).toBeCloseTo(transformed.rotation + 0.5, 6);
  };

  it('rotates (does not reset) at zoom 1', () => run(1));
  it('rotates (does not reset) at a non-1 zoom', () => run(2.5));
});

describe('transform tool: apply / cancel', () => {
  it('Enter applies the session', () => {
    const doc = makeDoc([imageLayer('a')], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    tool.onKeyCommand?.(h.ctx, 'apply');

    // The engine apply seam was invoked exactly once.
    expect(h.applyCount()).toBe(1);
  });

  it('Escape cancels the session', () => {
    const doc = makeDoc([imageLayer('a')], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);
    expect(h.session()).not.toBeNull();

    tool.onKeyCommand?.(h.ctx, 'cancel');

    expect(h.session()).toBeNull();
  });

  it('tool switch (deactivate) cancels the session', () => {
    const doc = makeDoc([imageLayer('a')], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    tool.onDeactivate?.(h.ctx);

    expect(h.session()).toBeNull();
  });

  it('pointercancel reverts the drag but keeps the session', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    down(tool, h.ctx, pointer(100, 100));
    move(tool, h.ctx, pointer(160, 160));
    expect(h.session()?.transform.scaleX).toBeGreaterThan(1);

    tool.onPointerCancel?.(h.ctx);

    // Session persists, reverted to the start transform.
    const s = h.session();
    expect(s).not.toBeNull();
    expect(s?.transform.scaleX).toBe(1);
  });

  it('Enter mid-drag no-ops: the gesture continues (a further move still updates the session) and pointer-up still works', () => {
    const doc = makeDoc([imageLayer('a', { width: 100, height: 100 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);

    // se corner drag, past the threshold — a gesture is now in progress and
    // holds (conceptually) pointer capture.
    down(tool, h.ctx, pointer(100, 100));
    move(tool, h.ctx, pointer(120, 120));
    const midDrag = h.session()?.transform.scaleX;
    expect(midDrag).toBeGreaterThan(1);

    tool.onKeyCommand?.(h.ctx, 'apply');

    // No-op: the engine apply seam was NOT invoked (unlike the "Enter applies
    // the session" test above, which has no live gesture).
    expect(h.applyCount()).toBe(0);

    // The gesture is still alive — it did not silently freeze mid-drag.
    move(tool, h.ctx, pointer(150, 150));
    expect(h.session()?.transform.scaleX).toBeGreaterThan(midDrag!);

    // Pointer-up still ends the gesture normally, keeping the session.
    up(tool, h.ctx, pointer(150, 150));
    expect(h.session()).not.toBeNull();
  });
});

describe('transform tool: temp-tool switch (space/alt hold)', () => {
  it('a temporary deactivate preserves the session; the matching temporary activate leaves it untouched', () => {
    const doc = makeDoc([imageLayer('a', { x: 5, y: 5 })], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);
    h.ctx.updateTransformSession?.({ rotation: 0, scaleX: 2, scaleY: 1, x: 40, y: 5 });
    const edited = h.session()?.transform;

    // Space down: a temporary deactivate must not cancel the session.
    tool.onDeactivate?.(h.ctx, { temporary: true });
    expect(h.session()?.transform).toEqual(edited);

    // Space up: a temporary activate must not re-open the session from the
    // current selection (which would stomp the preserved edit with the
    // layer's committed transform).
    tool.onActivate?.(h.ctx, { temporary: true });
    expect(h.session()?.transform).toEqual(edited);
  });

  it('a temporary activate does not resurrect a session the engine already cancelled while held (e.g. its layer was deleted)', () => {
    const doc = makeDoc([imageLayer('a')], 'a');
    const h = createHarness(doc);
    const tool = createTransformTool();
    activate(tool, h.ctx);
    expect(h.session()).not.toBeNull();

    tool.onDeactivate?.(h.ctx, { temporary: true });
    // Simulates the engine's layer-change teardown (Task 26 finding #3)
    // cancelling the session out-of-band while temp-switched away.
    h.ctx.cancelTransform?.();
    expect(h.session()).toBeNull();

    tool.onActivate?.(h.ctx, { temporary: true });
    expect(h.session()).toBeNull();
  });
});
