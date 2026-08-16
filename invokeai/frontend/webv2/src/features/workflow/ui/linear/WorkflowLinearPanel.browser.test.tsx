import type * as DndKitCoreModule from '@dnd-kit/core';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/react';
import type { ProjectGraphAction } from '@features/workflow/utility';

import { ChakraProvider } from '@chakra-ui/react';
import { useDroppable } from '@dnd-kit/core';
import { WorkflowUiProvider } from '@features/workflow/react';
import { createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { formEdgeDroppableId } from './formBuilderDnd';
import { FormBuilderTab } from './FormBuilderTab';
import { PanelModeToggle } from './WorkflowLinearPanel';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// This app is built with the React Compiler (`vite.config.mts`), which
// auto-memoizes plain value computations inside components based on their
// detected dependencies — so a spy on a pure helper a card's render calls
// (e.g. `isFormDescendantOrSelf`) can't be trusted as a "did this card's
// render function run" probe: the compiler can (and does) skip re-running
// such a call even when the surrounding component *was* invoked. `dnd-kit`'s
// `useDroppable` is a hook, and hook calls can never be skipped by the
// compiler (they must run unconditionally, in order, every render), so a
// real passthrough spy on it is a reliable per-card render probe instead.
vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof DndKitCoreModule>();

  return { ...actual, useDroppable: vi.fn(actual.useDroppable) };
});

describe('Workflow Linear panel mode toggle', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  const renderToggle = async () => {
    const Harness = () => {
      const [mode, setMode] = useState<'view' | 'edit'>('view');
      return <PanelModeToggle mode={mode} onChange={setMode} />;
    };

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    return [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
  };

  const selection = (tabs: HTMLButtonElement[]) => tabs.map((tab) => tab.getAttribute('aria-selected'));

  it('exposes View and Edit as a labelled tablist', async () => {
    const tabs = await renderToggle();

    expect(tabs).toHaveLength(2);
    expect(host.querySelector('[role="tablist"]')?.getAttribute('aria-label')).toBeTruthy();
    expect(selection(tabs)).toEqual(['true', 'false']);
    expect(
      tabs
        .map((tab) => tab.getAttribute('aria-controls'))
        .filter((id): id is string => id !== null)
        .map((id) => document.getElementById(id))
    ).not.toContain(null);
  });

  it('activates View and Edit with pointer and arrow keys', async () => {
    const tabs = await renderToggle();

    await act(() => userEvent.click(tabs[1]!));
    expect(selection(tabs)).toEqual(['false', 'true']);

    // Roving focus: the tablist is one tab stop and arrows move within it.
    tabs[1]?.focus();
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(selection(tabs)).toEqual(['true', 'false']);

    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(selection(tabs)).toEqual(['false', 'true']);
  });
});

/**
 * Regression coverage for the native-DnD bug: dropping a card into a
 * container reparents it, React remounts the card under its new parent, and
 * the native `dragend` event (bound to the old node) never fires — leaving
 * the module's dragging state stuck and killing every drag after it. The
 * dnd-kit port resolves moves only in `onDragEnd` at the `DndContext` level,
 * which fires regardless of node unmounts, so a second drag right after a
 * reparenting drop must still work.
 */
describe('Form builder drag and drop (dnd-kit)', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.width = '480px';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  /** root -> [heading "Field A", divider, container(column, empty)] */
  const buildInitialGraph = (): ProjectGraphState => {
    let doc = createProjectGraph('form-dnd-test');

    doc = projectGraphReducer(doc, { content: 'Field A', elementType: 'heading', type: 'addFormElement' });
    doc = projectGraphReducer(doc, { elementType: 'divider', type: 'addFormElement' });
    doc = projectGraphReducer(doc, { elementType: 'container', layout: 'column', type: 'addFormElement' });

    return doc;
  };

  const Harness = ({ initialGraph }: { initialGraph: ProjectGraphState }) => {
    const [projectGraph, setProjectGraph] = useState(initialGraph);
    const editGraph = useCallback((action: ProjectGraphAction) => {
      setProjectGraph((current) => projectGraphReducer(current, action));
    }, []);
    const adapter = useMemo(
      () =>
        ({
          commands: {
            bindLibraryWorkflow: () => undefined,
            editGraph,
            redo: () => undefined,
            replace: () => undefined,
            restoreSnapshot: () => undefined,
            saveSnapshot: () => undefined,
            undo: () => undefined,
          },
          widgets: { open: () => undefined, patchValues: () => undefined },
        }) as unknown as WorkflowUiAdapter,
      [editGraph]
    );

    return (
      <WorkflowUiProvider adapter={adapter}>
        <FormBuilderTab projectGraph={projectGraph} />
      </WorkflowUiProvider>
    );
  };

  const renderHarness = async (initialGraph: ProjectGraphState = buildInitialGraph()): Promise<void> => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Harness initialGraph={initialGraph} />
        </ChakraProvider>
      );
    });
  };

  /** The title-bar `HStack` is the drag handle: it's the direct DOM parent of its title `Text`. */
  const titleBarFor = (title: string): HTMLElement => {
    const leaf = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.children.length === 0 && element.textContent?.trim() === title
    );

    if (!leaf?.parentElement) {
      throw new Error(`title bar not found for "${title}"`);
    }

    return leaf.parentElement;
  };

  /** The card's content `Box` — the title bar's rounded-chrome parent's second (and last) child. */
  const cardContentFor = (title: string): HTMLElement => {
    const chrome = titleBarFor(title).parentElement;
    const content = chrome?.lastElementChild;

    if (!(content instanceof HTMLElement)) {
      throw new Error(`card content not found for "${title}"`);
    }

    return content;
  };

  const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
    target.dispatchEvent(
      new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
    );
  };

  const key = (target: EventTarget, code: string): void => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code }));
  };

  // dnd-kit's continuous droppable re-measuring (`MeasuringStrategy.Always`)
  // and sensor activation run on their own rAF-scheduled updates outside
  // React's synchronous event handling, so each step needs a real tick to
  // settle before the next one — a bare `act(() => pointer(...))` leaves
  // `over` stale and the drop resolves against the wrong (or no) target.
  const interact = (action: () => void): Promise<void> =>
    act(async () => {
      action();
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

  /** Drags `sourceTitle`'s title bar to `(x, y)` with a >4px jitter move first to arm the PointerSensor. */
  const dragTo = async (sourceTitle: string, x: number, y: number): Promise<void> => {
    const handle = titleBarFor(sourceTitle);
    const startRect = handle.getBoundingClientRect();
    const startX = startRect.left + startRect.width / 2;
    const startY = startRect.top + startRect.height / 2;

    await interact(() => pointer('pointerdown', handle, startX, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, startX + 8, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, x, y));
    // A `MeasuringStrategy.Always` remeasure lands one tick after the move
    // that triggered it, so the move's own `onDragMove` can still report a
    // stale `over`. A no-op settle move re-runs collision detection against
    // the now-current rects before the drop.
    await interact(() => pointer('pointermove', handle.ownerDocument, x, y + 1));
    await interact(() => pointer('pointerup', handle.ownerDocument, x, y + 1));
  };

  /**
   * Drags `sourceTitle`'s title bar to vertical position `targetCenterY` via
   * the keyboard: `Space` lifts, `ArrowDown`/`ArrowUp` steps 25px at a time
   * toward the target (dnd-kit's `defaultKeyboardCoordinateGetter`), `Space`
   * drops. Ends with a net-zero nudge in the same direction as the last real
   * step — the keyboard analogue of `dragTo`'s no-op settle move:
   * `MeasuringStrategy.Always`'s remeasure lands one tick after the arrow
   * press that triggered it, so the last press's own `onDragMove` can still
   * report a stale `over`.
   */
  const dragToWithKeyboard = async (sourceTitle: string, targetCenterY: number): Promise<void> => {
    const handle = titleBarFor(sourceTitle);
    const startRect = handle.getBoundingClientRect();
    const startCenterY = startRect.top + startRect.height / 2;
    const direction = targetCenterY >= startCenterY ? 'ArrowDown' : 'ArrowUp';
    const opposite = direction === 'ArrowDown' ? 'ArrowUp' : 'ArrowDown';
    const steps = Math.round(Math.abs(targetCenterY - startCenterY) / 25);

    handle.focus();
    expect(handle.ownerDocument.activeElement).toBe(handle);

    // Space lifts (dnd-kit's `KeyboardSensor` start code).
    await interact(() => key(handle, 'Space'));

    for (let step = 0; step < steps; step++) {
      await interact(() => key(handle, direction));
    }
    await interact(() => key(handle, direction));
    await interact(() => key(handle, opposite));

    // Space drops, resolving the move through the same `onDragEnd` ->
    // `moveFormElementTo` path the pointer drag uses.
    await interact(() => key(handle, 'Space'));
  };

  it('keeps dragging alive after a field is dropped into a container', async () => {
    await renderHarness();

    // Drag 1: "Field A" (a heading) into the empty container's drop zone.
    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();

    await dragTo('Heading', dropZoneRect.left + dropZoneRect.width / 2, dropZoneRect.top + dropZoneRect.height / 2);

    // The container's card content now holds the heading card, and the
    // empty-state hint is gone.
    const containerContent = cardContentFor('Container (column layout)');

    expect(containerContent.textContent).toContain('Heading');
    expect(containerContent.textContent).not.toContain('Empty container');

    // Drag 2: THE REGRESSION ASSERTION. "Divider" (still at the root) drops
    // onto the now-nested "Heading" card's lower edge. Under the old native
    // DnD implementation, drag 1's reparent remount lost `dragend` and left
    // `draggingElementId` stuck — this second drag would never start.
    const headingCardRect = titleBarFor('Heading').parentElement!.getBoundingClientRect();

    await dragTo('Divider', headingCardRect.left + headingCardRect.width / 2, headingCardRect.bottom - 2);

    // The divider moved into the container, next to the heading.
    const containerContentAfter = cardContentFor('Container (column layout)');

    expect(containerContentAfter.textContent).toContain('Heading');
    expect(containerContentAfter.textContent).toContain('Divider');

    // No card is left stuck at the mid-drag 40% opacity.
    const opacities = [...host.querySelectorAll<HTMLElement>('*')].map((element) => getComputedStyle(element).opacity);

    expect(opacities).not.toContain('0.4');
  });

  /**
   * Regression coverage for the "keyboard drag" finding: the drag handle
   * spreads dnd-kit's `attributes` (`role="button"`, `tabIndex=0`), which
   * promises "press space to lift, arrows to move" — a promise only true
   * once a `KeyboardSensor` is registered alongside the `PointerSensor`.
   * There's no pointer during a keyboard drag, so `KeyboardSensor` moves the
   * dragged card's tracked rect by a fixed 25px step per arrow press
   * (dnd-kit's `defaultKeyboardCoordinateGetter`) and collision detection
   * falls through to `rectIntersection` (`pointerWithin` needs real pointer
   * coordinates, which a keyboard drag never has).
   */
  it('moves a form element into a container with the keyboard', async () => {
    await renderHarness();

    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();

    await dragToWithKeyboard('Heading', dropZoneRect.top + dropZoneRect.height / 2);

    const containerContent = cardContentFor('Container (column layout)');

    expect(containerContent.textContent).toContain('Heading');
    expect(containerContent.textContent).not.toContain('Empty container');
  });

  /**
   * Coverage for the keyboard *edge* path specifically: the "into a
   * container" case above never reaches `getFormDropEdge`'s card-center
   * fallback (`handleDragMove` returns early for `into` targets before
   * `referenceY` is even computed) — a keyboard drag has no
   * `pointerCoordinates`, so this is the only path that exercises the
   * fallback dnd-kit needs for a real "arrows move the card" keyboard drag.
   * Moves "Divider" (root sibling, after "Heading") up onto the upper
   * quarter of "Heading"'s card, landing 'above' it and reordering it first.
   */
  it('reorders a form element above a sibling with the keyboard', async () => {
    await renderHarness();

    const headingCardRect = titleBarFor('Heading').parentElement!.getBoundingClientRect();

    await dragToWithKeyboard('Divider', headingCardRect.top + headingCardRect.height * 0.25);

    const leafElements = [...host.querySelectorAll<HTMLElement>('*')].filter(
      (element) => element.children.length === 0
    );
    const dividerIndex = leafElements.findIndex((element) => element.textContent?.trim() === 'Divider');
    const headingIndex = leafElements.findIndex((element) => element.textContent?.trim() === 'Heading');

    expect(dividerIndex).toBeGreaterThanOrEqual(0);
    expect(headingIndex).toBeGreaterThanOrEqual(0);
    expect(dividerIndex).toBeLessThan(headingIndex);
  });

  /**
   * Perf isolation for `BuilderDropTargetContext`: `dropTarget` (per-move
   * churn) lives apart from `BuilderDndContext`'s `activeElementId`/`form`
   * (drag start/end only), and `BuilderElement` is memoized on props that
   * don't change mid-drag — so a move that never targets "Divider" should
   * never re-invoke its `BuilderCard`. Probed via the `useDroppable`
   * passthrough spy (mocked at module scope above): `BuilderCard` calls it
   * exactly once per render with `formEdgeDroppableId(element.id)`, and hook
   * calls (unlike plain helper calls) can't be optimized away by the React
   * Compiler, so a call carrying "Divider"'s edge-droppable id is direct,
   * compiler-proof evidence its `BuilderCard` re-rendered.
   */
  it('does not re-render an unrelated card on a drag-move that only changes the drop target', async () => {
    const initialGraph = buildInitialGraph();
    const dividerId = Object.values(initialGraph.form.elements).find((element) => element.type === 'divider')!.id;
    const dividerDroppableId = formEdgeDroppableId(dividerId);

    await renderHarness(initialGraph);

    // The empty-container hint's rect, captured *before* the drag starts —
    // once a drag is active this card's `canDrop` flips true and its own
    // text changes from "Empty container..." to "Drop here", so the text
    // selector below would no longer match if read mid-drag.
    const emptyHint = [...host.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.textContent === 'Empty container — drag elements here'
    );

    expect(emptyHint).toBeDefined();

    const dropZoneRect = emptyHint!.getBoundingClientRect();
    const midX = dropZoneRect.left + dropZoneRect.width / 2;
    const midY = dropZoneRect.top + dropZoneRect.height / 2;

    const handle = titleBarFor('Heading');
    const startRect = handle.getBoundingClientRect();
    const startX = startRect.left + startRect.width / 2;
    const startY = startRect.top + startRect.height / 2;

    // Arm the `PointerSensor` (>4px activation constraint), then settle onto
    // the container's drop zone the same way `dragTo` does (a same-target
    // no-op nudge — `MeasuringStrategy.Always`'s remeasure lands one tick
    // after the move that triggered it, so a single move can still report a
    // stale `over`). This establishes a real `into` `dropTarget`; only the
    // moves *after* this point, which don't change the logical target, are
    // under test.
    await interact(() => pointer('pointerdown', handle, startX, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, startX + 8, startY));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 1));
    vi.mocked(useDroppable).mockClear();

    // Two more moves inside the same drop zone: `over` (and therefore the
    // *logical* `dropTarget`) doesn't change, but `handleDragMove` still
    // calls `setDropTarget` with a fresh object every time `onDragMove`
    // fires — that per-frame churn, not touching "Divider" at all, is
    // exactly what's under test.
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 2));
    await interact(() => pointer('pointermove', handle.ownerDocument, midX, midY + 3));

    const dividerCardRerendered = vi
      .mocked(useDroppable)
      .mock.calls.some(([options]) => options.id === dividerDroppableId);

    expect(dividerCardRerendered).toBe(false);

    // End the drag cleanly so it doesn't leak into other tests.
    await interact(() => pointer('pointerup', handle.ownerDocument, midX, midY + 3));
  });
});
