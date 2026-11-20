/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { DndContext, useDraggable, useSensor, useSensors } from '@dnd-kit/core';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { HoldToDragSensor, PrimaryMouseSensor } from './holdToDragSensor';

// Use short real-timer gates; 20px moves exceed both tolerance and immediate activation distance.
const HOLD_DELAY_MS = 120;
const HOLD_TOLERANCE_PX = 10;

// The arm wait must clear the hold delay even with interact()'s 50ms sleeps.
const HOLD_ELAPSED_MS = HOLD_DELAY_MS + 60;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const wait = (ms: number): Promise<void> =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, ms);
    });
  });

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number, init: PointerEventInit = {}) => {
  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      button: 0,
      cancelable: true,
      clientX,
      clientY,
      isPrimary: true,
      pointerId: 1,
      pointerType: 'touch',
      ...init,
    })
  );
};

const mouse = (type: string, target: EventTarget, clientX: number, clientY: number, button = 0) => {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, button, cancelable: true, clientX, clientY }));
};

const DraggableTile = ({ onClick, pannable }: { onClick: () => void; pannable: boolean }) => {
  const { listeners, setNodeRef } = useDraggable({ id: 'tile' });

  return (
    <button
      ref={setNodeRef}
      {...listeners}
      data-testid="tile"
      onClick={onClick}
      style={{ height: 100, touchAction: pannable ? 'auto' : 'none', width: 100 }}
      type="button"
    />
  );
};

const renderHarness = async (options?: { pannable?: boolean }) => {
  const pannable = options?.pannable ?? true;
  const events = {
    onClick: vi.fn(),
    onDragCancel: vi.fn(),
    onDragEnd: vi.fn(),
    onDragStart: vi.fn(),
  };

  const Harness = () => {
    const sensors = useSensors(
      useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }),
      useSensor(HoldToDragSensor, {
        activationConstraint: { delay: HOLD_DELAY_MS, tolerance: HOLD_TOLERANCE_PX },
      })
    );

    return (
      <DndContext
        sensors={sensors}
        onDragCancel={events.onDragCancel}
        onDragEnd={events.onDragEnd}
        onDragStart={events.onDragStart}
      >
        {/* The scroll container makes the tile touch-pannable; `touch-action: none` opts it out of the hold gate. */}
        <div style={{ height: 300, overflow: 'auto', width: 300 }}>
          <DraggableTile onClick={events.onClick} pannable={pannable} />
        </div>
      </DndContext>
    );
  };

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await interact(() => {
    root?.render(<Harness />);
  });

  return { events, tile: () => host!.querySelector<HTMLElement>('[data-testid="tile"]')! };
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('HoldToDragSensor touch gate', () => {
  it('gives an early move to the native scroll instead of activating a drag', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await interact(() => pointer('pointermove', document, 150, 190));

    expect(events.onDragStart).not.toHaveBeenCalled();
    expect(tile().getAttribute('data-drag-armed')).toBeNull();

    await wait(HOLD_ELAPSED_MS);
    await interact(() => pointer('pointermove', document, 150, 220));

    expect(events.onDragStart).not.toHaveBeenCalled();
    expect(events.onDragCancel).not.toHaveBeenCalled();
  });

  it('arms after a sustained hold, then drags on the follow-up move', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);

    expect(tile().getAttribute('data-drag-armed')).toBe('true');

    await interact(() => pointer('pointermove', document, 170, 150));

    expect(events.onDragStart).toHaveBeenCalledTimes(1);
    expect(tile().getAttribute('data-drag-armed')).toBeNull();

    await interact(() => pointer('pointerup', document, 190, 150));

    expect(events.onDragEnd).toHaveBeenCalledTimes(1);
  });

  it('holds the browser pan off from arming onward, but not during the hold', async () => {
    const { events, tile } = await renderHarness();

    const touchMove = () => {
      const event = new Event('touchmove', { bubbles: true, cancelable: true });
      document.dispatchEvent(event);
      return event;
    };

    await interact(() => pointer('pointerdown', tile(), 150, 150));

    expect(touchMove().defaultPrevented).toBe(false);

    await wait(HOLD_ELAPSED_MS);
    expect(tile().getAttribute('data-drag-armed')).toBe('true');

    expect(touchMove().defaultPrevented).toBe(true);

    await interact(() => pointer('pointermove', document, 170, 150));
    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    expect(touchMove().defaultPrevented).toBe(true);

    // End the gesture so the sensor cannot leak into the next test in this page.
    await interact(() => pointer('pointerup', document, 170, 150));
  });

  it('treats a motionless armed hold that lifts as an ordinary tap', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);

    await interact(() => {
      pointer('pointerup', document, 150, 150);
      tile().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });

    expect(events.onDragStart).not.toHaveBeenCalled();
    expect(tile().getAttribute('data-drag-armed')).toBeNull();
    expect(events.onClick).toHaveBeenCalledOnce();
  });

  it('swallows the trailing click of a drag that lifted on its own tile', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);
    await interact(() => pointer('pointermove', document, 170, 150));
    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    // The click must land within the 50ms post-lift window the swallow covers, exactly as a browser's compat click
    // after a touch lift would.
    await interact(() => {
      pointer('pointerup', document, 170, 150);
      tile().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });

    expect(events.onClick).not.toHaveBeenCalled();
    expect(events.onDragEnd).toHaveBeenCalledTimes(1);
  });

  it('releases the gesture on pointercancel and accepts a new one', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);
    await interact(() => pointer('pointermove', document, 170, 150));
    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    await interact(() => pointer('pointercancel', document, 170, 150));
    expect(events.onDragCancel).toHaveBeenCalledTimes(1);

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);
    await interact(() => pointer('pointermove', document, 170, 150));

    expect(events.onDragStart).toHaveBeenCalledTimes(2);

    // End the gesture: a sensor left mid-drag outlives the unmounted harness (its listeners are on the shared
    // window/document) and would leak into the next test in this page.
    await interact(() => pointer('pointerup', document, 170, 150));
  });

  it('cancels the hold when a second finger lands', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await interact(() => pointer('pointerdown', document, 100, 100, { isPrimary: false, pointerId: 2 }));

    await wait(HOLD_ELAPSED_MS);
    expect(tile().getAttribute('data-drag-armed')).toBeNull();

    await interact(() => pointer('pointermove', document, 170, 150));

    expect(events.onDragStart).not.toHaveBeenCalled();
  });

  it('lets the context menu through when armed, and suppresses it mid-drag', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);
    expect(tile().getAttribute('data-drag-armed')).toBe('true');

    const menuWhileArmed = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    await interact(() => window.dispatchEvent(menuWhileArmed));

    expect(menuWhileArmed.defaultPrevented).toBe(false);
    expect(tile().getAttribute('data-drag-armed')).toBeNull();

    await interact(() => pointer('pointermove', document, 170, 150));
    expect(events.onDragStart).not.toHaveBeenCalled();

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await wait(HOLD_ELAPSED_MS);
    await interact(() => pointer('pointermove', document, 170, 150));
    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    const menuWhileDragging = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    await interact(() => window.dispatchEvent(menuWhileDragging));

    expect(menuWhileDragging.defaultPrevented).toBe(true);

    // End the gesture so the sensor cannot leak into the next test in this page.
    await interact(() => pointer('pointerup', document, 170, 150));
  });

  it('activates immediately by distance on a surface that cannot pan', async () => {
    const { events, tile } = await renderHarness({ pannable: false });

    await interact(() => pointer('pointerdown', tile(), 150, 150));
    await interact(() => pointer('pointermove', document, 170, 150));

    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    await interact(() => pointer('pointerup', document, 170, 150));

    expect(events.onDragEnd).toHaveBeenCalledTimes(1);
  });

  it('activates immediately by distance for pen pointers on a pannable surface', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => pointer('pointerdown', tile(), 150, 150, { pointerType: 'pen' }));
    await interact(() => pointer('pointermove', document, 170, 150, { pointerType: 'pen' }));

    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    await interact(() => pointer('pointerup', document, 170, 150, { pointerType: 'pen' }));

    expect(events.onDragEnd).toHaveBeenCalledTimes(1);
  });
});

describe('PrimaryMouseSensor button guard', () => {
  it('starts drags for the primary button only', async () => {
    const { events, tile } = await renderHarness();

    await interact(() => mouse('mousedown', tile(), 150, 150, 1));
    await interact(() => mouse('mousemove', document, 170, 150));

    expect(events.onDragStart).not.toHaveBeenCalled();

    await interact(() => mouse('mousedown', tile(), 150, 150));
    await interact(() => mouse('mousemove', document, 170, 150));

    expect(events.onDragStart).toHaveBeenCalledTimes(1);

    await interact(() => mouse('mouseup', document, 170, 150));

    expect(events.onDragEnd).toHaveBeenCalledTimes(1);
  });
});
