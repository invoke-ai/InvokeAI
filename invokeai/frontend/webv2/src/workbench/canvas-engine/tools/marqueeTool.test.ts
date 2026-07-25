import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { createEngineStores } from '@workbench/canvas-engine/engineStores';
import { ellipsePathData, rectPathData } from '@workbench/canvas-engine/selection/selectionPaths';
import { createMarqueeTool, marqueeRect } from '@workbench/canvas-engine/tools/marqueeTool';
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

describe('marqueeRect', () => {
  const plain = { fromCenter: false, square: false };

  it('normalizes a drag in any direction', () => {
    expect(marqueeRect({ x: 10, y: 10 }, { x: 40, y: 30 }, plain)).toEqual({ height: 20, width: 30, x: 10, y: 10 });
    // Dragging up and to the left produces the same rect.
    expect(marqueeRect({ x: 40, y: 30 }, { x: 10, y: 10 }, plain)).toEqual({ height: 20, width: 30, x: 10, y: 10 });
  });

  it('constrains to a square under shift, keeping the drag direction', () => {
    expect(marqueeRect({ x: 0, y: 0 }, { x: 30, y: 10 }, { fromCenter: false, square: true })).toEqual({
      height: 30,
      width: 30,
      x: 0,
      y: 0,
    });
    // Up-left: the square grows away from the press point on both axes.
    expect(marqueeRect({ x: 50, y: 50 }, { x: 20, y: 40 }, { fromCenter: false, square: true })).toEqual({
      height: 30,
      width: 30,
      x: 20,
      y: 20,
    });
  });

  it('treats the press point as the centre under alt, doubling the extent', () => {
    expect(marqueeRect({ x: 50, y: 50 }, { x: 60, y: 70 }, { fromCenter: true, square: false })).toEqual({
      height: 40,
      width: 20,
      x: 40,
      y: 30,
    });
  });

  it('composes shift and alt into a square centred on the press point', () => {
    expect(marqueeRect({ x: 50, y: 50 }, { x: 60, y: 70 }, { fromCenter: true, square: true })).toEqual({
      height: 40,
      width: 40,
      x: 30,
      y: 30,
    });
  });

  it('rounds to whole pixels', () => {
    expect(marqueeRect({ x: 10.4, y: 10.6 }, { x: 20.4, y: 20.6 }, plain)).toEqual({
      height: 10,
      width: 10,
      x: 10,
      y: 11,
    });
  });
});

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

  it('resolves the boolean op from the modifiers held at release', () => {
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
