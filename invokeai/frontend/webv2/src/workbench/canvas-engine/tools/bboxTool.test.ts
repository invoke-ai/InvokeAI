import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/contracts';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it, vi } from 'vitest';

import { createBboxTool } from './bboxTool';

// Grid-aligned start frame (grid 8) so a zero-net-move gesture is a true no-op.
const bbox = { height: 96, width: 96, x: 16, y: 16 };

const makeDoc = (): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { ...bbox },
  height: 512,
  layers: [],
  selectedLayerId: null,
  version: 2,
  width: 512,
});

// Identity viewport: screen coordinates equal document coordinates (zoom 1, no pan),
// so pointer inputs can use the same value for both spaces.
const identityViewport = {
  documentToScreen: (p: Vec2): Vec2 => ({ x: p.x, y: p.y }),
} as unknown as Viewport;

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
  previewOf: () => CanvasDocumentContractV2['bbox'] | null;
}

const createHarness = (doc: CanvasDocumentContractV2): Harness => {
  const dispatched: CanvasProjectMutation[] = [];
  const commits: StructuralCommit[] = [];
  const stores = createEngineStores();
  const ctx: ToolContext = {
    backend: null as never,
    commitStructural: (label, forward, inverse) => commits.push({ forward, inverse, label }),
    createLayerId: () => 'x',
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
  return { commits, ctx, dispatched, previewOf: () => stores.bboxPreview.get() };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);
const cancel = (t: Tool, ctx: ToolContext): void => t.onPointerCancel?.(ctx);

describe('bbox tool: move gesture', () => {
  it('previews on move then commits one setCanvasBbox on up (grid-snapped)', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();

    down(tool, h.ctx, pointer(60, 60)); // inside the frame → move
    move(tool, h.ctx, pointer(75, 75)); // delta (15,15) → moved
    expect(h.previewOf()).not.toBeNull();

    up(tool, h.ctx, pointer(75, 75));

    expect(h.dispatched).toHaveLength(0);
    expect(h.commits).toHaveLength(1);
    // origin snapped: 16+15 = 31 → grid 8 → 32.
    expect(h.commits[0]?.forward).toEqual({ bbox: { height: 96, width: 96, x: 32, y: 32 }, type: 'setCanvasBbox' });
    expect(h.commits[0]?.inverse).toEqual({ bbox: { ...bbox }, type: 'setCanvasBbox' });
    // Preview cleared after commit.
    expect(h.previewOf()).toBeNull();
  });

  it('bypasses snapping while alt is held', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(60, 60, { alt: true }));
    move(tool, h.ctx, pointer(75, 75, { alt: true }));
    up(tool, h.ctx, pointer(75, 75, { alt: true }));
    expect(h.commits[0]?.forward).toEqual({ bbox: { height: 96, width: 96, x: 31, y: 31 }, type: 'setCanvasBbox' });
  });
});

describe('bbox tool: resize gesture', () => {
  it('resizes from the SE handle and commits the grid-snapped frame', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();

    down(tool, h.ctx, pointer(112, 112)); // SE corner handle
    move(tool, h.ctx, pointer(130, 130)); // delta (18,18)
    up(tool, h.ctx, pointer(130, 130));

    expect(h.commits).toHaveLength(1);
    // right/bottom = 112+18 = 130 → grid 8 → 128; anchored at (16,16).
    expect(h.commits[0]?.forward).toEqual({ bbox: { height: 112, width: 112, x: 16, y: 16 }, type: 'setCanvasBbox' });
  });

  it('constrains the ratio while shift is held on an unlocked frame', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    // Start frame is square (96x96) → shift preserves 1:1.
    down(tool, h.ctx, pointer(112, 112));
    move(tool, h.ctx, pointer(150, 112, { shift: true })); // widen only, ratio 1:1 keeps square
    up(tool, h.ctx, pointer(150, 112, { shift: true }));
    const out = h.commits[0]?.forward as { bbox: { width: number; height: number } };
    expect(out.bbox.width).toBe(out.bbox.height);
  });
});

describe('bbox tool: no-op cases', () => {
  it('commits nothing on a zero-delta gesture that returns to the origin', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(60, 60));
    move(tool, h.ctx, pointer(70, 70)); // crosses threshold → moved
    move(tool, h.ctx, pointer(60, 60)); // back to origin
    up(tool, h.ctx, pointer(60, 60));
    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it('does nothing when pressing outside the frame (never deselects)', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(400, 400));
    move(tool, h.ctx, pointer(420, 420));
    up(tool, h.ctx, pointer(420, 420));
    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });

  it('ignores a sub-threshold press (no drag)', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(60, 60));
    up(tool, h.ctx, pointer(61, 61));
    expect(h.commits).toHaveLength(0);
  });
});

describe('bbox tool: cancel', () => {
  it('drops the preview and commits nothing on escape/cancel', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(112, 112));
    move(tool, h.ctx, pointer(130, 130));
    expect(h.previewOf()).not.toBeNull();
    cancel(tool, h.ctx);
    expect(h.commits).toHaveLength(0);
    expect(h.dispatched).toHaveLength(0);
    expect(h.previewOf()).toBeNull();
  });
});

describe('bbox tool: hover cursors', () => {
  it('reflects the hovered handle/interior in the cursor and refreshes it on change', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    // Hover (no gesture) then read the tool's CSS cursor. Frame is {16,16,96,96}.
    const cursorAt = (x: number, y: number): string | undefined => {
      move(tool, h.ctx, pointer(x, y));
      return tool.cursor?.(h.ctx);
    };

    expect(cursorAt(112, 112)).toBe('nwse-resize'); // SE corner
    expect(cursorAt(16, 112)).toBe('nesw-resize'); // SW corner
    expect(cursorAt(16, 16)).toBe('nwse-resize'); // NW corner
    expect(cursorAt(112, 16)).toBe('nesw-resize'); // NE corner
    expect(cursorAt(64, 16)).toBe('ns-resize'); // N edge midpoint
    expect(cursorAt(112, 64)).toBe('ew-resize'); // E edge midpoint
    expect(cursorAt(64, 64)).toBe('move'); // interior
    expect(cursorAt(400, 400)).toBe('default'); // off the frame

    // The tool asked the engine to re-apply the cursor as the hover target changed.
    expect(h.ctx.updateCursor).toHaveBeenCalled();
  });

  it('holds the grabbed handle cursor during a resize drag', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(112, 112)); // grab SE corner
    move(tool, h.ctx, pointer(130, 130)); // dragging
    expect(tool.cursor?.(h.ctx)).toBe('nwse-resize');
  });
});

describe('bbox tool: undo/redo contract', () => {
  it('carries an inverse that restores the prior bbox', () => {
    const h = createHarness(makeDoc());
    const tool = createBboxTool();
    down(tool, h.ctx, pointer(112, 112));
    move(tool, h.ctx, pointer(130, 130));
    up(tool, h.ctx, pointer(130, 130));
    // The engine's commitStructural dispatches `inverse` on undo — restoring the original frame.
    expect(h.commits[0]?.inverse).toEqual({ bbox: { ...bbox }, type: 'setCanvasBbox' });
  });
});
