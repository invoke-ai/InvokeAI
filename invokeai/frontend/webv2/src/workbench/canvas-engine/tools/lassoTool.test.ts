import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { createLassoTool, lassoOpFor } from '@workbench/canvas-engine/tools/lassoTool';
import { describe, expect, it, vi } from 'vitest';

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
  } as unknown as ToolContext;
  return { commits, ctx, dispatched, stores };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

describe('lassoOpFor', () => {
  it('maps modifiers to ops, falling back to the persistent mode', () => {
    const none = { alt: false, ctrl: false, meta: false, shift: false };
    expect(lassoOpFor(none, 'replace')).toBe('replace');
    expect(lassoOpFor(none, 'add')).toBe('add');
    expect(lassoOpFor({ ...none, shift: true }, 'replace')).toBe('add');
    expect(lassoOpFor({ ...none, alt: true }, 'replace')).toBe('subtract');
    expect(lassoOpFor({ ...none, alt: true, shift: true }, 'replace')).toBe('intersect');
  });
});

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
    stores.lassoOptions.set({ mode: 'subtract' });
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
