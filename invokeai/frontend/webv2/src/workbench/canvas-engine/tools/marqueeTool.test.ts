import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { ellipsePathData, rectPathData } from '@workbench/canvas-engine/selection/selectionPaths';
import { createMarqueeTool } from '@workbench/canvas-engine/tools/marqueeTool';
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
  const ctx = {
    commitSelection: (commit: SelectionCommit) => commits.push(commit),
    createPath2D: (d?: string) => ({ d }) as unknown as Path2D,
    getDocument: () => ({ layers: [] }),
    invalidate: vi.fn(),
    stores,
  } as unknown as ToolContext;
  return { commits, ctx, stores };
};

const down = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerDown?.(ctx, i);
const move = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerMove?.(ctx, i, [i]);
const up = (t: Tool, ctx: ToolContext, i: PointerInput): void => t.onPointerUp?.(ctx, i);

/** A press at `from`, a move past the drag threshold, and a release at `to`. */
const drag = (tool: Tool, ctx: ToolContext, from: PointerInput, to: PointerInput): void => {
  down(tool, ctx, from);
  move(tool, ctx, to);
  up(tool, ctx, to);
};

describe('marqueeTool: drag → commit', () => {
  it('commits a rect path with the persistent op mode', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(10, 10), pointer(40, 30));

    expect(h.commits).toHaveLength(1);
    expect(h.commits[0]!.op).toBe('replace');
    expect(h.commits[0]!.bounds).toEqual({ height: 20, width: 30, x: 10, y: 10 });
    expect(h.commits[0]!.path).toEqual({ d: rectPathData({ height: 20, width: 30, x: 10, y: 10 }) });
  });

  it('commits an ellipse path when the options say so', () => {
    const h = createHarness();
    h.stores.marqueeOptions.set({ kind: 'ellipse', mode: 'replace' });
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(0, 0), pointer(40, 20));

    expect(h.commits[0]!.path).toEqual({ d: ellipsePathData({ height: 20, width: 40, x: 0, y: 0 }) });
  });

  it('shapes with a modifier pressed mid-drag without changing the op', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 30, { shift: true }));
    up(tool, h.ctx, pointer(40, 30, { shift: true }));

    expect(h.commits[0]!.op).toBe('replace');
    expect(h.commits[0]!.bounds).toEqual({ height: 30, width: 30, x: 10, y: 10 });
  });

  it('adds with shift held before the press, without constraining the rect', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(10, 10, { shift: true }), pointer(40, 30, { shift: true }));

    expect(h.commits[0]!.op).toBe('add');
    expect(h.commits[0]!.bounds).toEqual({ height: 20, width: 30, x: 10, y: 10 });
  });

  it('lets an op key released mid-drag shape the rect when pressed again', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10, { shift: true }));
    move(tool, h.ctx, pointer(20, 15));
    move(tool, h.ctx, pointer(40, 30, { shift: true }));
    up(tool, h.ctx, pointer(40, 30, { shift: true }));

    expect(h.commits[0]!.op).toBe('add');
    expect(h.commits[0]!.bounds).toEqual({ height: 30, width: 30, x: 10, y: 10 });
  });

  it('subtracts with alt held before the press, without drawing from the center', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(10, 10, { alt: true }), pointer(40, 30, { alt: true }));

    expect(h.commits[0]!.op).toBe('subtract');
    expect(h.commits[0]!.bounds).toEqual({ height: 20, width: 30, x: 10, y: 10 });
  });

  it('resolves the boolean op from the modifiers held at the press', () => {
    const cases: [{ shift?: boolean; alt?: boolean }, string][] = [
      [{ shift: true }, 'add'],
      [{ alt: true }, 'subtract'],
      [{ alt: true, shift: true }, 'intersect'],
    ];
    for (const [modifiers, expected] of cases) {
      const h = createHarness();
      const tool = createMarqueeTool();
      drag(tool, h.ctx, pointer(10, 10, modifiers), pointer(40, 30, modifiers));
      expect(h.commits[0]!.op).toBe(expected);
    }
  });

  it('honours the persistent mode when no modifier is held', () => {
    const h = createHarness();
    h.stores.marqueeOptions.set({ kind: 'rect', mode: 'subtract' });
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(10, 10), pointer(40, 30));

    expect(h.commits[0]!.op).toBe('subtract');
  });

  it('publishes a live preview during the drag and clears it on commit', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 30));
    expect(h.stores.marqueePreview.get()).toEqual({ kind: 'rect', rect: { height: 20, width: 30, x: 10, y: 10 } });

    up(tool, h.ctx, pointer(40, 30));
    expect(h.stores.marqueePreview.get()).toBeNull();
  });

  it('stays below the drag threshold without previewing or committing', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(11, 11));
    up(tool, h.ctx, pointer(11, 11));

    expect(h.stores.marqueePreview.get()).toBeNull();
    expect(h.commits).toHaveLength(0);
  });

  it('commits nothing for a degenerate drag rather than wiping the selection', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    // Past the threshold vertically, but the rect collapses to zero width.
    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(10, 30));
    up(tool, h.ctx, pointer(10, 30));

    expect(h.commits).toHaveLength(0);
  });

  it('ignores a non-primary button press', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    drag(tool, h.ctx, pointer(10, 10, { buttons: 2 }), pointer(40, 30, { buttons: 2 }));

    expect(h.commits).toHaveLength(0);
  });
});

describe('marqueeTool: cancel', () => {
  it('drops the preview and commits nothing on pointercancel', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 30));
    tool.onPointerCancel?.(h.ctx);

    expect(h.stores.marqueePreview.get()).toBeNull();
    expect(h.commits).toHaveLength(0);

    // A trailing pointerup from the released pointer does nothing.
    up(tool, h.ctx, pointer(40, 30));
    expect(h.commits).toHaveLength(0);
  });

  it('drops the preview on Escape', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 30));
    tool.onKeyCommand?.(h.ctx, 'cancel');

    expect(h.stores.marqueePreview.get()).toBeNull();
    expect(h.commits).toHaveLength(0);
  });

  it('drops the preview when the tool is deactivated mid-drag', () => {
    const h = createHarness();
    const tool = createMarqueeTool();

    down(tool, h.ctx, pointer(10, 10));
    move(tool, h.ctx, pointer(40, 30));
    tool.onDeactivate?.(h.ctx);

    expect(h.stores.marqueePreview.get()).toBeNull();
    expect(h.commits).toHaveLength(0);
  });
});
