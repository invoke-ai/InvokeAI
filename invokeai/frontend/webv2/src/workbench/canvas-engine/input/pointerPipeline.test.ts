import type { CanvasKeyOwner } from '@workbench/canvas-engine/capabilities';
import type { PointerPipelineDeps } from '@workbench/canvas-engine/input/pointerPipeline';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId } from '@workbench/canvas-engine/types';

import { registerModalPresence } from '@platform/ui/modalPresence';
import { createPointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { afterEach, describe, expect, it, vi } from 'vitest';

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

const makeKeyEvent = (init: {
  altKey?: boolean;
  code?: string;
  ctrlKey?: boolean;
  defaultPrevented?: boolean;
  key?: string;
  metaKey?: boolean;
  repeat?: boolean;
  shiftKey?: boolean;
  target?: unknown;
}): KeyboardEvent =>
  ({
    altKey: init.altKey ?? false,
    code: init.code ?? '',
    ctrlKey: init.ctrlKey ?? false,
    defaultPrevented: init.defaultPrevented ?? false,
    key: init.key ?? '',
    metaKey: init.metaKey ?? false,
    preventDefault: vi.fn(),
    repeat: init.repeat ?? false,
    shiftKey: init.shiftKey ?? false,
    target: init.target ?? null,
  }) as unknown as KeyboardEvent;

const SURFACE_TARGET = { tagName: 'DIV' };
const BUTTON_TARGET = { closest: () => ({}), tagName: 'BUTTON' };

const createHarness = (
  opts: {
    tools?: ToolId[];
    handleEscape?: (o: { gestureWasActive: boolean }) => boolean;
    maybeCommitModalSession?: () => boolean;
    replaying?: () => boolean;
    resolveKeyOwner?: (event: KeyboardEvent) => CanvasKeyOwner;
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
  const gestureChanges: boolean[] = [];

  const deps: PointerPipelineDeps = {
    getActiveTool: () => tool,
    getActiveToolId: () => activeToolId,
    getInputElement: () => element,
    getKeyboardRoot: () => ({ contains: (node) => node === (SURFACE_TARGET as unknown) }),
    getKeyOwnerResolver: () => opts.resolveKeyOwner ?? null,
    getToolContext: () => ctx,
    handleEscape: opts.handleEscape,
    hasTool: (id) => registered.has(id),
    isReplaying: () => opts.replaying?.() ?? false,
    maybeCommitModalSession: opts.maybeCommitModalSession,
    onGestureChange: (active) => gestureChanges.push(active),
    setTool,
    updateCursor: vi.fn(),
    viewport: createViewport(),
  };

  return { captures, deps, gestureChanges, pipeline: createPointerPipeline(deps), releases, setTool, tool };
};

afterEach(() => {
  vi.useRealTimers();
});

describe('pointer pipeline: pressure + capture', () => {
  it('captures the pointer on primary down and defaults mouse pressure to 0.5', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 7, pointerType: 'mouse', pressure: 0 }));

    expect(h.captures).toEqual([7]);
    expect(h.tool.downs).toHaveLength(1);
    expect(h.tool.downs[0]!.pressure).toBe(0.5);
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
});

describe('pointer pipeline: modal (text-edit) session commit on pointerdown', () => {
  it('commits and swallows a primary press when a modal session is open (no capture/gesture/tool routing)', () => {
    const maybeCommitModalSession = vi.fn(() => true);
    const h = createHarness({ maybeCommitModalSession });
    const event = makePointerEvent({ pointerId: 5 });

    h.pipeline.onPointerDown(event);

    expect(maybeCommitModalSession).toHaveBeenCalledTimes(1);
    expect(h.captures).toEqual([]);
    expect(h.tool.downs).toHaveLength(0);
    expect(h.pipeline.isGestureActive()).toBe(false);
    expect(event.preventDefault).toHaveBeenCalled();
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

  it('marks only a press that cancelled something, so a later handler can tell it was used', () => {
    const idle = makeKeyEvent({ key: 'Escape' });
    createHarness({ handleEscape: () => false }).pipeline.onKeyDown(idle);
    expect(idle.preventDefault).not.toHaveBeenCalled();

    const cancelling = makeKeyEvent({ key: 'Escape' });
    createHarness({ handleEscape: () => true }).pipeline.onKeyDown(cancelling);
    expect(cancelling.preventDefault).toHaveBeenCalled();

    const midGesture = makeKeyEvent({ key: 'Escape' });
    const h = createHarness({ handleEscape: () => false });
    h.pipeline.onPointerDown(makePointerEvent({}));
    h.pipeline.onKeyDown(midGesture);
    expect(midGesture.preventDefault).toHaveBeenCalled();
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
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 9 }));
    expect(h.tool.ups).toHaveLength(0);
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
});

describe('pointer pipeline: gesture-active state', () => {
  it('reports the primary-button gesture from down to up, announcing it before the tool sees the press', () => {
    const h = createHarness();
    const announcedAtToolDown: boolean[][] = [];
    h.tool.onPointerDown = () => announcedAtToolDown.push([...h.gestureChanges]);
    expect(h.pipeline.isGestureActive()).toBe(false);
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 1 }));
    expect(h.pipeline.isGestureActive()).toBe(true);
    expect(announcedAtToolDown).toEqual([[true]]);
    h.pipeline.onPointerMove(makePointerEvent({ pointerId: 1, clientX: 10 }));
    expect(h.pipeline.isGestureActive()).toBe(true);
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 1 }));
    expect(h.pipeline.isGestureActive()).toBe(false);
    expect(h.gestureChanges).toEqual([true, false]);
  });
});

describe('pointer pipeline: cancelActiveGesture + reset run onPointerCancel', () => {
  it('cancelActiveGesture releases capture and runs the tool cancel once; a no-op with no gesture', () => {
    const h = createHarness();
    h.pipeline.cancelActiveGesture();
    expect(h.tool.cancels).toBe(0);

    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 5 }));
    h.pipeline.cancelActiveGesture();
    expect(h.tool.cancels).toBe(1);
    expect(h.releases).toContain(5);
    expect(h.pipeline.isGestureActive()).toBe(false);

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
    expect(h.setTool).toHaveBeenLastCalledWith('view', { temporary: true });
    h.pipeline.onKeyUp(makeKeyEvent({ code: 'Space' }));
    expect(h.setTool).toHaveBeenLastCalledWith('brush', { temporary: true });
  });

  it('alt-hold switches to colorPicker when registered and restores on release', () => {
    const h = createHarness({ tools: ['view', 'brush', 'colorPicker'] });
    h.pipeline.onPointerEnter();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'AltLeft' }));
    expect(h.setTool).toHaveBeenLastCalledWith('colorPicker', { temporary: true });
    h.pipeline.onKeyUp(makeKeyEvent({ code: 'AltLeft' }));
    expect(h.setTool).toHaveBeenLastCalledWith('brush', { temporary: true });
  });

  it('quick-tapping C sticky-selects the bbox tool after the temporary preview switch', () => {
    const h = createHarness({ tools: ['view', 'brush', 'bbox'] });
    h.pipeline.onPointerEnter();

    h.pipeline.onKeyDown(makeKeyEvent({ code: 'KeyC', key: 'c' }));
    expect(h.setTool).toHaveBeenLastCalledWith('bbox', { temporary: true });

    h.pipeline.onKeyUp(makeKeyEvent({ code: 'KeyC', key: 'c' }));
    expect(h.setTool).toHaveBeenLastCalledWith('bbox');
    expect(h.setTool.mock.calls.slice(-2)).toEqual([['brush', { temporary: true }], ['bbox']]);
  });

  it('ignores space/alt holds when the key event targets an editable element', () => {
    const h = createHarness({ tools: ['view', 'brush', 'colorPicker'] });
    h.pipeline.onPointerEnter();

    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: { tagName: 'INPUT' } }));
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: { tagName: 'textarea' } }));
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'AltLeft', target: { isContentEditable: true } }));
    expect(h.setTool).not.toHaveBeenCalled();

    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: SURFACE_TARGET }));
    expect(h.setTool).toHaveBeenLastCalledWith('view', { temporary: true });
  });
});

describe('pointer pipeline: Alt eyedropper hold', () => {
  const pickerHarness = (opts: { resolveKeyOwner?: (event: KeyboardEvent) => CanvasKeyOwner } = {}) =>
    createHarness({ ...opts, tools: ['view', 'brush', 'colorPicker'] });
  const altDown = (target: unknown = SURFACE_TARGET) => makeKeyEvent({ altKey: true, code: 'AltLeft', target });
  const altUp = () => makeKeyEvent({ code: 'AltLeft', target: SURFACE_TARGET });

  it('prevents both halves of an owned Alt tap so the browser menu bar stays closed', () => {
    const h = pickerHarness();
    h.pipeline.onPointerEnter();
    const down = altDown();
    const up = altUp();
    h.pipeline.onKeyDown(down);
    h.pipeline.onKeyUp(up);
    expect(down.preventDefault).toHaveBeenCalledOnce();
    expect(up.preventDefault).toHaveBeenCalledOnce();

    const space = makeKeyEvent({ code: 'Space', target: SURFACE_TARGET });
    h.pipeline.onKeyUp(space);
    expect(space.preventDefault).not.toHaveBeenCalled();
  });
});

describe('pointer pipeline: injected key owner', () => {
  const ownerHarness = (owner: CanvasKeyOwner) => {
    const handleEscape = vi.fn();
    const resolveKeyOwner = vi.fn((_event: KeyboardEvent) => owner);
    const h = createHarness({ handleEscape, resolveKeyOwner, tools: ['view', 'brush', 'bbox', 'colorPicker'] });
    const commands: string[] = [];
    h.tool.onKeyCommand = (_ctx, command) => commands.push(command) > 0;
    return { ...h, commands, handleEscape, resolveKeyOwner };
  };
  const BODY = { tagName: 'BODY' };

  it.each([
    ['surface', ['apply'], 1],
    ['companion', [], 1],
    ['elsewhere', [], 0],
  ] as const)('with owner %s, Enter applies %j and Escape reaches the ladder %i time(s)', (owner, applied, escapes) => {
    const h = ownerHarness(owner);
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: BODY }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape', target: BODY }));
    expect(h.commands).toEqual(applied);
    expect(h.handleEscape).toHaveBeenCalledTimes(escapes);
  });
});

describe('pointer pipeline: canvas keyboard ownership', () => {
  const applyHarness = () => {
    const h = createHarness();
    const commands: string[] = [];
    h.tool.onKeyCommand = (_ctx, command) => commands.push(command) > 0;
    return { ...h, commands };
  };

  it('applies a session on Enter only while the surface (or no control) owns focus', () => {
    const h = applyHarness();
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: SURFACE_TARGET }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: { tagName: 'BODY' } }));
    expect(h.commands).toEqual(['apply', 'apply']);

    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: BUTTON_TARGET }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: { tagName: 'DIV' } }));
    h.pipeline.onKeyDown(makeKeyEvent({ defaultPrevented: true, key: 'Enter', target: SURFACE_TARGET }));
    for (const modifier of ['altKey', 'ctrlKey', 'metaKey', 'shiftKey'] as const) {
      h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', [modifier]: true, target: SURFACE_TARGET }));
    }
    expect(h.commands).toEqual(['apply', 'apply']);
  });

  it('takes no key beneath a modal dialog, even with focus left on the document body', () => {
    const handleEscape = vi.fn();
    const h = createHarness({ handleEscape });
    const commands: string[] = [];
    h.tool.onKeyCommand = (_ctx, command) => commands.push(command) > 0;
    h.pipeline.onPointerEnter();
    const body = { tagName: 'BODY' };
    const release = registerModalPresence();

    try {
      h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: body }));
      h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape', target: body }));
      h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', key: ' ', target: body }));
    } finally {
      release();
    }
    expect(commands).toEqual([]);
    expect(handleEscape).not.toHaveBeenCalled();
    expect(h.setTool).not.toHaveBeenCalled();

    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Enter', target: body }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Escape', target: body }));
    expect(commands).toEqual(['apply']);
    expect(handleEscape).toHaveBeenCalledOnce();
  });
});

describe('pointer pipeline: history replay', () => {
  it('starts no gesture while a replay runs, but still pans with the middle button', () => {
    let replaying = true;
    const h = createHarness({ replaying: () => replaying });
    const press = makePointerEvent({ pointerId: 1 });
    h.pipeline.onPointerDown(press);

    expect(h.tool.downs).toHaveLength(0);
    expect(h.pipeline.isGestureActive()).toBe(false);
    expect(press.preventDefault).toHaveBeenCalled();
    h.pipeline.onPointerDown(makePointerEvent({ button: 1, pointerId: 2 }));
    expect(h.captures).toEqual([2]);

    replaying = false;
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 2 }));
    expect(h.gestureChanges).toEqual([]);
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 3 }));
    expect(h.tool.downs).toHaveLength(1);
  });
});

describe('pointer pipeline: cancellation ownership', () => {
  it('a tool-switch cancel turns the rest of the pressed drag into hover until the pointer is released', () => {
    const h = createHarness();
    h.pipeline.onPointerDown(makePointerEvent({ pointerId: 4 }));
    h.pipeline.endForToolSwitch();
    expect(h.tool.cancels).toBe(1);
    expect(h.releases).toEqual([4]);

    h.pipeline.onPointerMove(makePointerEvent({ clientX: 5, pointerId: 4 }));
    h.pipeline.onPointerUp(makePointerEvent({ buttons: 0, pointerId: 4 }));
    expect(h.tool.moves.map(({ batch, input }) => [input.buttons, ...batch.map((sample) => sample.buttons)])).toEqual([
      [0, 0],
    ]);
    expect(h.tool.ups).toHaveLength(0);

    h.pipeline.onPointerMove(makePointerEvent({ buttons: 0, clientX: 6, pointerId: 4 }));
    expect(h.tool.moves).toHaveLength(2);
  });

  it('releases tool-owned modifiers on blur reset when their keyup was missed between presses', () => {
    const h = createHarness();
    const held = new Set<string>();
    h.tool.onModifierKey = (_ctx, modifier, pressed) => {
      if (pressed) {
        held.add(modifier);
      } else {
        held.delete(modifier);
      }
      return true;
    };
    const moves: string[][] = [];
    h.tool.onPointerMove = () => moves.push([...held]);
    h.pipeline.onPointerEnter();
    h.pipeline.onKeyDown(makeKeyEvent({ code: 'Space', target: SURFACE_TARGET }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Alt', code: 'AltLeft', target: SURFACE_TARGET }));
    h.pipeline.onKeyDown(makeKeyEvent({ key: 'Shift', target: SURFACE_TARGET }));
    expect(held.size).toBe(3);
    h.pipeline.reset();
    h.pipeline.onPointerEnter();
    h.pipeline.onPointerDown(makePointerEvent());
    h.pipeline.onPointerMove(makePointerEvent({ clientX: 50, clientY: 50 }));
    expect(moves).toEqual([[]]);
  });
});
