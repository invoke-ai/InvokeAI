import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { WorkbenchAction } from '@workbench/workbenchState.testing';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLassoTool } from '@workbench/canvas-engine/tools/lassoTool';
import { describe, expect, it, vi } from 'vitest';

const pointer = (
  x: number,
  y: number,
  opts: { shift?: boolean; alt?: boolean; buttons?: number; timeStamp?: number } = {}
): PointerInput => ({
  buttons: opts.buttons ?? 1,
  documentPoint: { x, y },
  modifiers: { alt: opts.alt ?? false, ctrl: false, meta: false, shift: opts.shift ?? false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: opts.timeStamp ?? 0,
});

const createHarness = () => {
  const stores = createEngineStores();
  const commits: SelectionCommit[] = [];
  const dispatched: WorkbenchAction[] = [];
  const ctx = {
    commitSelection: (commit: SelectionCommit) => commits.push(commit),
    createPath2D: (d?: string) => ({ d }) as unknown as Path2D,
    dispatch: (action: WorkbenchAction) => dispatched.push(action),
    invalidate: vi.fn(),
    stores,
    // Identity viewport: document and screen coordinates coincide, so the
    // polygon's close-on-first-vertex hit test is expressed in the same numbers.
    viewport: { documentToScreen: (p: { x: number; y: number }) => p },
  } as unknown as ToolContext;
  return { commits, ctx, dispatched, stores };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('lassoTool: drag → commit', () => {
  const drag = (tool: Tool, ctx: ToolContext, upOpts: { shift?: boolean; alt?: boolean } = {}): void => {
    down(tool, ctx, pointer(0, 0));
    move(tool, ctx, pointer(20, 0));
    move(tool, ctx, pointer(20, 20));
    up(tool, ctx, pointer(0, 20, upOpts));
  };

  it('commits a replace by default and publishes/clears the live preview', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createHarness();

    down(tool, ctx, pointer(0, 0));
    move(tool, ctx, pointer(20, 0));
    expect(stores.lassoPreview.get()).not.toBeNull();

    move(tool, ctx, pointer(20, 20));
    up(tool, ctx, pointer(0, 20));

    expect(commits).toHaveLength(1);
    expect(commits[0]!.op).toBe('replace');
    expect(commits[0]!.bounds).toEqual({ height: 20, width: 20, x: 0, y: 0 });
    // Preview cleared on commit.
    expect(stores.lassoPreview.get()).toBeNull();
  });

  it('resolves each modifier op', () => {
    for (const [mods, op] of [
      [{ shift: true }, 'add'],
      [{ alt: true }, 'subtract'],
      [{ shift: true, alt: true }, 'intersect'],
    ] as const) {
      const tool = createLassoTool();
      const { commits, ctx } = createHarness();
      drag(tool, ctx, mods);
      expect(commits[0]!.op).toBe(op);
    }
  });

  it('uses the persistent lasso op mode when no modifier is held', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createHarness();
    stores.lassoOptions.set({ mode: 'subtract', shape: 'freehand' });
    drag(tool, ctx);
    expect(commits[0]!.op).toBe('subtract');
  });

  it('never dispatches to the reducer (selection is transient)', () => {
    const tool = createLassoTool();
    const { ctx, dispatched } = createHarness();
    drag(tool, ctx);
    expect(dispatched).toHaveLength(0);
  });
});

describe('lassoTool: cancel + guards', () => {
  it('Escape/pointercancel drops the in-progress path without committing', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createHarness();
    down(tool, ctx, pointer(0, 0));
    move(tool, ctx, pointer(20, 0));
    move(tool, ctx, pointer(20, 20));
    tool.onPointerCancel?.(ctx);
    expect(commits).toHaveLength(0);
    expect(stores.lassoPreview.get()).toBeNull();
  });

  it('a click (too few points) commits nothing', () => {
    const tool = createLassoTool();
    const { commits, ctx } = createHarness();
    down(tool, ctx, pointer(5, 5));
    up(tool, ctx, pointer(5, 5));
    expect(commits).toHaveLength(0);
  });

  it('decimates points closer than the minimum distance', () => {
    const tool = createLassoTool();
    const { ctx, stores } = createHarness();
    down(tool, ctx, pointer(0, 0));
    // These are all within 1px of each other → decimated away.
    move(tool, ctx, pointer(0.5, 0));
    move(tool, ctx, pointer(1, 0));
    const preview = stores.lassoPreview.get();
    expect(preview).toHaveLength(1);
  });
});

describe('lassoTool: polygon mode', () => {
  const createPolygonHarness = () => {
    const h = createHarness();
    h.stores.lassoOptions.set({ mode: 'replace', shape: 'polygon' });
    return h;
  };

  /** One vertex placement: a press and release at the same spot. */
  const click = (tool: Tool, ctx: ToolContext, x: number, y: number, timeStamp: number): void => {
    down(tool, ctx, pointer(x, y, { timeStamp }));
    up(tool, ctx, pointer(x, y, { timeStamp }));
  };

  it('accumulates vertices across clicks without committing', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);

    expect(commits).toHaveLength(0);
    expect(stores.lassoPreview.get()).toHaveLength(3);
  });

  it('shows a rubber-band segment to the cursor between clicks', () => {
    const tool = createLassoTool();
    const { ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    move(tool, ctx, pointer(30, 10));

    // The placed vertex plus the live cursor endpoint.
    expect(stores.lassoPreview.get()).toEqual([
      { x: 0, y: 0 },
      { x: 30, y: 10 },
    ]);
  });

  it('closes and commits on Enter', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    tool.onKeyCommand?.(ctx, 'apply');

    expect(commits).toHaveLength(1);
    expect(commits[0]!.bounds).toEqual({ height: 40, width: 40, x: 0, y: 0 });
    expect(stores.lassoPreview.get()).toBeNull();
  });

  it('closes on a click landing on the first vertex', () => {
    const tool = createLassoTool();
    const { commits, ctx } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    // Within POLYGON_CLOSE_HIT_PX of the first vertex (identity viewport).
    click(tool, ctx, 3, 2, 3000);

    expect(commits).toHaveLength(1);
    expect(commits[0]!.bounds).toEqual({ height: 40, width: 40, x: 0, y: 0 });
  });

  it('does not close on the first vertex before the polygon is fillable', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    // A second click back near the start must add a vertex, not close a 2-gon.
    click(tool, ctx, 2, 0, 1000);

    expect(commits).toHaveLength(0);
    expect(stores.lassoPreview.get()).toHaveLength(2);
  });

  it('closes on a double-click', () => {
    const tool = createLassoTool();
    const { commits, ctx } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    // A second press at the same spot, inside the double-click window.
    click(tool, ctx, 40, 40, 2100);

    expect(commits).toHaveLength(1);
  });

  it('treats a slow second click at the same spot as a new vertex, not a close', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    click(tool, ctx, 40, 40, 9000);

    expect(commits).toHaveLength(0);
    expect(stores.lassoPreview.get()).toHaveLength(4);
  });

  it('drops the session on Escape without committing', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    tool.onKeyCommand?.(ctx, 'cancel');

    expect(commits).toHaveLength(0);
    expect(stores.lassoPreview.get()).toBeNull();
  });

  it('survives a temporary tool switch so the user can pan mid-polygon', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    tool.onDeactivate?.(ctx, { temporary: true });
    expect(stores.lassoPreview.get()).toHaveLength(2);

    tool.onActivate?.(ctx, { temporary: true });
    click(tool, ctx, 40, 40, 2000);
    tool.onKeyCommand?.(ctx, 'apply');

    expect(commits).toHaveLength(1);
  });

  it('drops the session on a real tool switch', () => {
    const tool = createLassoTool();
    const { commits, ctx, stores } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    tool.onDeactivate?.(ctx);

    expect(stores.lassoPreview.get()).toBeNull();
    expect(commits).toHaveLength(0);
  });

  it('applies the modifiers held on the closing click', () => {
    const tool = createLassoTool();
    const { commits, ctx } = createPolygonHarness();

    click(tool, ctx, 0, 0, 0);
    click(tool, ctx, 40, 0, 1000);
    click(tool, ctx, 40, 40, 2000);
    down(tool, ctx, pointer(1, 1, { shift: true, timeStamp: 3000 }));

    expect(commits[0]!.op).toBe('add');
  });
});
