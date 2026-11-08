import { act } from 'react';
import { vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const FRAME_MS = 16;
const STABLE_FRAMES = 3;

const nextFrame = (): Promise<void> =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, FRAME_MS);
    });
  });

let pointerTargets = 0;

/**
 * A momentary 2px target on top of everything: the real pointer can only be moved by hovering an element. Hovering
 * hands the browser a selector, which is positional unless the element has an id; a hint card's portal unmounting
 * between the two would shift a positional one off the target, and the hover would wait on nothing.
 */
const createPointerTarget = (): HTMLDivElement => {
  const target = document.createElement('div');

  pointerTargets += 1;
  target.id = `feature-hint-pointer-target-${pointerTargets}`;

  return target;
};

/** Move the real pointer to the page corner, so the next hover enters its trigger afresh and emits pointerenter. */
export const parkPointer = async (): Promise<void> => {
  const parking = createPointerTarget();

  parking.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;z-index:2147483647';
  document.body.append(parking);
  await act(() => userEvent.hover(parking));
  parking.remove();
};

/** The open card, which portals out of the test host. */
export const getHintCard = (): HTMLElement | null =>
  document.querySelector<HTMLElement>('[data-scope="hover-card"][data-part="content"]');

/** Hover a hint trigger and wait until its card has opened and stopped moving; resolves to the card's rect. */
export const openHintCard = async (trigger: Element, timeoutMs = 5000): Promise<DOMRect> => {
  await parkPointer();
  await act(() => userEvent.hover(trigger));

  const deadline = Date.now() + timeoutMs;
  let last: DOMRect | null = null;
  let stableFrames = 0;

  while (stableFrames < STABLE_FRAMES) {
    if (Date.now() > deadline) {
      throw new Error(`Timed out after ${timeoutMs}ms waiting for the hint card to open and settle`);
    }

    await nextFrame();

    const rect = getHintCard()?.getBoundingClientRect() ?? null;
    const isSame = rect !== null && last !== null && rect.x === last.x && rect.y === last.y;

    stableFrames = isSame ? stableFrames + 1 : 0;
    last = rect;
  }

  return last!;
};

type Point = { x: number; y: number };

/**
 * The probe sits on top of everything, so hovering it is also a leave from whatever lies beneath (the card, say). The
 * browser recomputes hover in the next frame and re-enters that element; wait for it, or a close the leave scheduled
 * can fire first. Bounded by a second frame: by then the recomputation has run, and a step that never put the
 * pointer over the probe has nothing to re-enter.
 */
const pointerUncovered = (): Promise<void> =>
  new Promise<void>((resolve) => {
    let frame = 0;
    const done = () => {
      document.removeEventListener('pointerover', done, true);
      cancelAnimationFrame(frame);
      resolve();
    };

    document.addEventListener('pointerover', done, true);
    frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(done);
    });
  });

/**
 * Walk the real pointer from `from` to `to` in straight steps, one pointermove per step: a single hover jumps, which
 * never crosses the space between a label and its card.
 *
 * Under fake timers each step advances the clock by exactly `stepDelayMs`, so the walk's timing no longer depends on
 * how long the browser takes to move the pointer.
 */
export const movePointerAlong = async (
  from: Point,
  to: Point,
  { stepDelayMs, steps }: { stepDelayMs: number; steps: number }
): Promise<void> => {
  for (let step = 0; step <= steps; step += 1) {
    const x = from.x + ((to.x - from.x) * step) / steps;
    const y = from.y + ((to.y - from.y) * step) / steps;
    const probe = createPointerTarget();

    probe.style.cssText = `position:fixed;left:${x - 1}px;top:${y - 1}px;width:2px;height:2px;z-index:2147483647`;
    document.body.append(probe);
    await act(() => userEvent.hover(probe));

    probe.remove();
    await act(() => pointerUncovered());

    if (vi.isFakeTimers()) {
      await act(() => {
        vi.advanceTimersByTime(stepDelayMs);
      });
    } else {
      await act(async () => {
        await new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, stepDelayMs);
        });
      });
    }
  }
};
