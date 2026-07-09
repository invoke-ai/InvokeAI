import type { PointerPipelineDeps } from '@workbench/canvas-engine/input/pointerPipeline';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId } from '@workbench/canvas-engine/types';

import { createPointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it, vi } from 'vitest';

// ---- Fakes -----------------------------------------------------------------

interface FakePointerInit {
  button?: number;
  buttons?: number;
  clientX?: number;
  clientY?: number;
  pointerId?: number;
  pointerType?: string;
  pressure?: number;
  ctrlKey?: boolean;
  altKey?: boolean;
  coalesced?: FakePointerInit[];
}

const makePointerEvent = (init: FakePointerInit = {}): PointerEvent => {
  const self: Record<string, unknown> = {
    altKey: init.altKey ?? false,
    button: init.button ?? 0,
    buttons: init.buttons ?? 1,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    ctrlKey: init.ctrlKey ?? false,
    metaKey: false,
    pointerId: init.pointerId ?? 1,
    pointerType: init.pointerType ?? 'mouse',
    preventDefault: vi.fn(),
    pressure: init.pressure ?? 0,
    shiftKey: false,
    timeStamp: 0,
  };
  self.getCoalescedEvents = () => (init.coalesced ?? []).map(makePointerEvent);
  return self as unknown as PointerEvent;
};

const makeKeyEvent = (init: { code?: string; key?: string; repeat?: boolean; target?: unknown }): KeyboardEvent =>
  ({
    code: init.code ?? '',
    key: init.key ?? '',
    preventDefault: vi.fn(),
    repeat: init.repeat ?? false,
    target: init.target ?? null,
  }) as unknown as KeyboardEvent;

const createHarness = (
  opts: {
    tools?: ToolId[];
    handleEscape?: (o: { gestureWasActive: boolean }) => void;
    maybeCommitModalSession?: () => boolean;
  } = {}
) => {
  const registered = new Set<ToolId>(opts.tools ?? ['view', 'brush']);
  const tool: Tool & {
    downs: PointerInput[];
    moves: { input: PointerInput; batch: readonly PointerInput[] }[];
    ups: PointerInput[];
    cancels: number;
  } = {
    cancels: 0,
    downs: [],
    id: 'brush',
    moves: [],
    onPointerCancel: () => {
      tool.cancels += 1;
    },
    onPointerDown: (_ctx, input) => {
      tool.downs.push(input);
    },
    onPointerMove: (_ctx, input, batch) => {
      tool.moves.push({ batch, input });
    },
    onPointerUp: (_ctx, input) => {
      tool.ups.push(input);
    },
    ups: [],
  };

  const captures: number[] = [];
  const releases: number[] = [];
  const element = {
    getBoundingClientRect: () => ({ left: 0, top: 0 }) as DOMRect,
    releasePointerCapture: (id: number) => releases.push(id),
    setPointerCapture: (id: number) => captures.push(id),
  } as unknown as HTMLElement;

  let activeToolId: ToolId = 'brush';
  const setTool = vi.fn((id: ToolId) => {
    activeToolId = id;
  });
  const ctx = {} as ToolContext;

  const deps: PointerPipelineDeps = {
    getActiveTool: () => tool,
    getActiveToolId: () => activeToolId,
    getInputElement: () => element,
    getToolContext: () => ctx,
    handleEscape: opts.handleEscape,
    hasTool: (id) => registered.has(id),
    maybeCommitModalSession: opts.maybeCommitModalSession,
    setTool,
    updateCursor: vi.fn(),
    viewport: createViewport(),
  };

  return { captures, deps, pipeline: createPointerPipeline(deps), releases, setTool, tool };
};

// ---- Tests -----------------------------------------------------------------

describe('pointer pipeline: pressure + capture', () => {
  it('captures the pointer on primary down and defaults mouse pressure to 0.5', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 7, pointerType: 'mouse', pressure: 0 }));

    expect(h.captures).toEqual([7]);
    expect(h.tool.downs).toHaveLength(1);
    expect(h.tool.downs[0]!.pressure).toBe(0.5);
  });

  it('preserves reported pen pressure', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerType: 'pen', pressure: 0.73 }));
    expect(h.tool.downs[0]!.pressure).toBeCloseTo(0.73);
    expect(h.tool.downs[0]!.pointerType).toBe('pen');
  });

  it('releases capture on pointer up and forwards the up sample', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 3 }));
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 3 }));
    expect(h.releases).toEqual([3]);
    expect(h.tool.ups).toHaveLength(1);
  });
});

describe('pointer pipeline: coalesced batching', () => {
  it('expands coalesced events into a batch whose last element is the primary sample', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({}));
    h.pipeline.onPointerMove(
      makePointerEvent({
        clientX: 30,
        coalesced: [
          { clientX: 10, clientY: 10 },
          { clientX: 20, clientY: 15 },
          { clientX: 30, clientY: 20 },
        ],
      })
    );

    expect(h.tool.moves).toHaveLength(1);
    const { batch, input } = h.tool.moves[0]!;
    expect(batch).toHaveLength(3);
    expect(input).toBe(batch[2]);
    expect(batch[0]!.screenPoint).toEqual({ x: 10, y: 10 });
  });

  it('falls back to a single-sample batch when no coalesced events exist', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({}));
    h.pipeline.onPointerMove(makePointerEvent({ clientX: 42, coalesced: [] }));
    expect(h.tool.moves[0]!.batch).toHaveLength(1);
    expect(h.tool.moves[0]!.input.screenPoint.x).toBe(42);
  });
});

describe('pointer pipeline: modal (text-edit) session commit on pointerdown', () => {
  it('commits and swallows a primary press when a modal session is open (no capture/gesture/tool routing)', () => {
    const maybeCommitModalSession = vi.fn(() => true);
    const h = createHarness({ maybeCommitModalSession });
    const event = makePointerEvent({ pointerId: 5 });

    h.pipeline.onPointerDown(event);

    expect(maybeCommitModalSession).toHaveBeenCalledTimes(1);
    expect(h.captures).toEqual([]); // no pointer capture
    expect(h.tool.downs).toHaveLength(0); // not routed to the active tool
    expect(h.pipeline.isGestureActive()).toBe(false); // ran before the gesture flag is set
    expect(event.preventDefault).toHaveBeenCalled();
  });

  it('starts a normal gesture when no modal session is open (returns false)', () => {
    const maybeCommitModalSession = vi.fn(() => false);
    const h = createHarness({ maybeCommitModalSession });

    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 6 }));

    expect(maybeCommitModalSession).toHaveBeenCalledTimes(1);
    expect(h.tool.downs).toHaveLength(1);
    expect(h.pipeline.isGestureActive()).toBe(true);
    expect(h.captures).toEqual([6]);
  });
});

describe('pointer pipeline: Escape priority (handleEscape)', () => {
  it('runs handleEscape with gestureWasActive=false when Escape is pressed idle', () => {
    const handleEscape = vi.fn();
    const h = createHarness({ handleEscape });
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape' }));
    expect(handleEscape).toHaveBeenCalledTimes(1);
    expect(handleEscape).toHaveBeenCalledWith({ gestureWasActive: false });
  });

  it('cancels the active gesture AND flags it to handleEscape (gestureWasActive=true)', () => {
    const handleEscape = vi.fn();
    const h = createHarness({ handleEscape });
    h.pipeline.onPointerDown(makePointerEvent({}));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape' }));
    expect(h.tool.cancels).toBe(1);
    expect(handleEscape).toHaveBeenCalledWith({ gestureWasActive: true });
  });

  it('does not run handleEscape when Escape targets an editable field', () => {
    const handleEscape = vi.fn();
    const h = createHarness({ handleEscape });
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape', target: { tagName: 'INPUT' } }));
    expect(handleEscape).not.toHaveBeenCalled();
  });
});

describe('pointer pipeline: gesture cancel + secondary buttons', () => {
  it('routes pointercancel to the tool and releases capture', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 5 }));
    h.pipeline.onPointerCancel(makePointerEvent({ pointerId: 5 }));
    expect(h.tool.cancels).toBe(1);
    expect(h.releases).toEqual([5]);
  });

  it('cancels the active gesture on Escape', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 9 }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape' }));
    expect(h.tool.cancels).toBe(1);
    // A subsequent up is ignored (the gesture already ended).
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 9 }));
    expect(h.tool.ups).toHaveLength(0);
  });

  it('ignores secondary-button presses during an active gesture', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ button: 0 }));
    h.pipeline.onPointerDown(makePointerEvent({ button: 2, buttons: 3 }));
    expect(h.tool.downs).toHaveLength(1);
  });

  it('ignores a standalone secondary-button press', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ button: 2, buttons: 2 }));
    expect(h.tool.downs).toHaveLength(0);
    expect(h.captures).toHaveLength(0);
  });
});

describe('pointer pipeline: middle-mouse pan', () => {
  it('pans the viewport on middle drag without routing to the tool', () => {
    const h = createHarness();
    const panBy = vi.spyOn(h.deps.viewport, 'panBy');
    h.pipeline.onPointerDown(makePointerEvent({ button: 1, buttons: 4, clientX: 0 }));
    h.pipeline.onPointerMove(makePointerEvent({ buttons: 4, clientX: 25 }));
    expect(panBy).toHaveBeenCalledWith({ x: 25, y: 0 });
    expect(h.tool.moves).toHaveLength(0);
  });
});

describe('pointer pipeline: multi-pointer isolation during an active gesture', () => {
  it('ignores move events from a second pointer while the first pointer is gesturing', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    h.pipeline.onPointerMove(makePointerEvent({ pointerId: 2, clientX: 99 }));
    expect(h.tool.moves).toHaveLength(0);
    h.pipeline.onPointerMove(makePointerEvent({ pointerId: 1, clientX: 10 }));
    expect(h.tool.moves).toHaveLength(1);
  });

  it('ignores a second pointer up but ends the gesture on the first pointer up', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 2 }));
    expect(h.tool.ups).toHaveLength(0);
    expect(h.releases).toHaveLength(0);
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 1 }));
    expect(h.tool.ups).toHaveLength(1);
    expect(h.releases).toEqual([1]);
  });

  it('ignores cancel events from a second pointer during an active gesture', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    h.pipeline.onPointerCancel(makePointerEvent({ pointerId: 2 }));
    expect(h.tool.cancels).toBe(0);
    h.pipeline.onPointerCancel(makePointerEvent({ pointerId: 1 }));
    expect(h.tool.cancels).toBe(1);
  });

  it('ignores a second pointerdown while a gesture is active', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 2 }));
    expect(h.tool.downs).toHaveLength(1);
    expect(h.captures).toEqual([1]);
  });
});

describe('pointer pipeline: gesture-active state', () => {
  it('reports the primary-button gesture from down to up', () => {
    const h = createHarness();
    expect(h.pipeline.isGestureActive()).toBe(false);
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    expect(h.pipeline.isGestureActive()).toBe(true);
    h.pipeline.onPointerMove(makePointerEvent({ pointerId: 1, clientX: 10 }));
    expect(h.pipeline.isGestureActive()).toBe(true);
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 1 }));
    expect(h.pipeline.isGestureActive()).toBe(false);
  });

  it('clears on pointercancel, Escape, and reset', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    h.pipeline.onPointerCancel(makePointerEvent({ pointerId: 1 }));
    expect(h.pipeline.isGestureActive()).toBe(false);

    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 2 }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape' }));
    expect(h.pipeline.isGestureActive()).toBe(false);

    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 3 }));
    h.pipeline.reset();
    expect(h.pipeline.isGestureActive()).toBe(false);
  });

  it('does not report a middle-mouse pan as a gesture', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ button: 1, buttons: 4 }));
    expect(h.pipeline.isGestureActive()).toBe(false);
  });
});

describe('pointer pipeline: cancelActiveGesture + reset run onPointerCancel', () => {
  it('cancelActiveGesture releases capture and runs the tool cancel once; a no-op with no gesture', () => {
    const h = createHarness();
    // No gesture: a no-op (the tool cancel must not fire).
    h.pipeline.cancelActiveGesture();
    expect(h.tool.cancels).toBe(0);

    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 5 }));
    h.pipeline.cancelActiveGesture();
    expect(h.tool.cancels).toBe(1);
    expect(h.releases).toContain(5);
    expect(h.pipeline.isGestureActive()).toBe(false);

    // A second call is a no-op — the gesture is already cancelled.
    h.pipeline.cancelActiveGesture();
    expect(h.tool.cancels).toBe(1);
  });

  it('reset cancels an in-flight gesture (runs onPointerCancel + releases capture), unlike a bare flag clear', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 9 }));
    expect(h.pipeline.isGestureActive()).toBe(true);

    h.pipeline.reset();
    expect(h.tool.cancels).toBe(1);
    expect(h.releases).toContain(9);
    expect(h.pipeline.isGestureActive()).toBe(false);
  });
});

describe('pointer pipeline: temporary modifier tools', () => {
  it('holds view while space is down and restores the prior tool on release', () => {
    const h = createHarness();
    h.pipeline.onPointerEnter();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space' }));
    // Flagged `temporary` so a session-bearing tool (transform) can tell this
    // apart from a real switch and keep its session alive across the hold.
    expect(h.setTool).toHaveBeenLastCalledWith('view', { temporary: true });
    h.pipeline.onKeyUp(makeKeyEvent({ code: 'Space' }));
    expect(h.setTool).toHaveBeenLastCalledWith('brush', { temporary: true });
  });

  it('does not switch on space when the pointer is not over the canvas', () => {
    const h = createHarness();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space' }));
    expect(h.setTool).not.toHaveBeenCalled();
  });

  it('alt-hold is a no-op switch when colorPicker is not registered', () => {
    const h = createHarness({ tools: ['view', 'brush'] });
    h.pipeline.onPointerEnter();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'AltLeft' }));
    h.pipeline.onKeyUp(makeKeyEvent({ code: 'AltLeft' }));
    expect(h.setTool).not.toHaveBeenCalled();
  });

  it('alt-hold switches to colorPicker when registered and restores on release', () => {
    const h = createHarness({ tools: ['view', 'brush', 'colorPicker'] });
    h.pipeline.onPointerEnter();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'AltLeft' }));
    expect(h.setTool).toHaveBeenLastCalledWith('colorPicker', { temporary: true });
    h.pipeline.onKeyUp(makeKeyEvent({ code: 'AltLeft' }));
    expect(h.setTool).toHaveBeenLastCalledWith('brush', { temporary: true });
  });

  it('ignores space/alt holds when the key event targets an editable element', () => {
    const h = createHarness({ tools: ['view', 'brush', 'colorPicker'] });
    h.pipeline.onPointerEnter();

    // Typing space/alt in an input, textarea, or contenteditable belongs to that
    // field, not the canvas — the temp-tool hold must not fire.
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: { tagName: 'INPUT' } }));
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: { tagName: 'textarea' } }));
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'AltLeft', target: { isContentEditable: true } }));
    expect(h.setTool).not.toHaveBeenCalled();

    // A non-editable target (e.g. the canvas) still triggers the hold.
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: { tagName: 'CANVAS' } }));
    expect(h.setTool).toHaveBeenLastCalledWith('view', { temporary: true });
  });
});
