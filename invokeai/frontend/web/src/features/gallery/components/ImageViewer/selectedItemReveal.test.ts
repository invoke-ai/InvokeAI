import { createAutoSwitchedSelectionMarker } from 'features/gallery/store/autoSwitchedImages';
import { describe, expect, it } from 'vitest';

import { createSelectedItemRevealController } from './selectedItemReveal';

/**
 * Drives the controller through the same call sequences the preview components' effects produce,
 * with a hand-cranked timer so reveal expiry is explicit. The marker is the real implementation —
 * the reveal/auto-switch interplay is exactly what these tests exist to pin.
 */
const createHarness = () => {
  let revealed = false;
  const lastRenderedItemNameRef = { current: null as string | null };
  const marker = createAutoSwitchedSelectionMarker();
  const timers = new Map<number, () => void>();
  let nextTimerId = 1;

  const controller = createSelectedItemRevealController({
    lastRenderedItemNameRef,
    marker,
    setRevealed: (value) => {
      revealed = value;
    },
    durationMs: 2000,
    schedule: (fn) => {
      const id = nextTimerId++;
      timers.set(id, fn);
      return id;
    },
    cancel: (id) => {
      timers.delete(id);
    },
  });

  return {
    controller,
    marker,
    lastRenderedItemNameRef,
    isRevealed: () => revealed,
    // The components' unmount handler lowers the shared flag directly, without going through the
    // controller — StrictMode runs it between the doubled mount effects.
    lowerExternally: () => {
      revealed = false;
    },
    fireTimers: () => {
      for (const [id, fn] of [...timers]) {
        timers.delete(id);
        fn();
      }
    },
    pendingTimerCount: () => timers.size,
  };
};

// An auto-switch as onInvocationComplete + the selection listener produce it: record, then the
// dispatched selection lands and settles the marker.
const autoSwitchTo = (marker: ReturnType<typeof createAutoSwitchedSelectionMarker>, itemName: string) => {
  marker.record(itemName);
  marker.settle(itemName);
};

const inputs = (overrides: Partial<Parameters<ReturnType<typeof createHarness>['controller']['run']>[0]> = {}) => ({
  shouldShowProgressInViewer: true,
  hasProgressImage: true,
  isProgressImageResolving: false,
  renderedItemName: null as string | null,
  selectedItemName: null as string | null,
  ...overrides,
});

const rendering = (itemName: string) => inputs({ renderedItemName: itemName, selectedItemName: itemName });

describe('createSelectedItemRevealController', () => {
  it('reveals a mid-render selection change, then lowers when the timer fires', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(true);
    h.fireTimers();
    expect(h.isRevealed()).toBe(false);
  });

  it('does not reveal the first render after the viewer opens', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    expect(h.isRevealed()).toBe(false);
  });

  it('does not re-reveal when a later run finds the same item and no reveal in flight', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    h.fireTimers();
    // e.g. the resolving flag flapping, or any other dependency change with the item unchanged.
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(false);
  });

  it('does not reveal an auto-switched item', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    autoSwitchTo(h.marker, 'b.png');
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(false);
  });

  it('consumes the marker even when no progress is showing, so a later click on that item reveals', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    autoSwitchTo(h.marker, 'b.png');
    // The auto-switched item renders with the overlay down — the common case.
    h.controller.run(inputs({ renderedItemName: 'b.png', selectedItemName: 'b.png', hasProgressImage: false }));
    expect(h.isRevealed()).toBe(false);
    // A new render starts; the user clicks away and back to the once-auto-switched item.
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(true);
  });

  it('keeps the reveal through a StrictMode double-invoked mount effect', () => {
    // React StrictMode mounts run effect -> cleanup -> effect. The cleanup cancels the reveal
    // timer and the unmount handler lowers the shared flag; the second run then finds the shared
    // ref already holding the new name. Without the re-arm, every cross-media first reveal dies
    // in development.
    const h = createHarness();
    h.controller.run(rendering('previous-image.png'));
    h.controller.run(rendering('clicked-video.mp4'));
    expect(h.isRevealed()).toBe(true);
    h.controller.clearTimer();
    h.lowerExternally();
    h.controller.run(rendering('clicked-video.mp4'));
    expect(h.isRevealed()).toBe(true);
    expect(h.pendingTimerCount()).toBe(1);
    h.fireTimers();
    expect(h.isRevealed()).toBe(false);
  });

  it('reveals a click that landed inside a resolve window once the window ends', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    // The user clicks a video while a finished render's preview is resolving; the next render's
    // progress then resumes before the resolve ends.
    h.controller.run(inputs({ isProgressImageResolving: true, renderedItemName: 'b.mp4', selectedItemName: 'b.mp4' }));
    expect(h.isRevealed()).toBe(false);
    // The ref must not have advanced past the click — that is what kept this click dead before.
    expect(h.lastRenderedItemNameRef.current).toBe('a.png');
    h.controller.run(rendering('b.mp4'));
    expect(h.isRevealed()).toBe(true);
  });

  it('still suppresses an auto-switch whose render landed inside a resolve window', () => {
    // The marker must not be consumed by a run that cannot reveal, or the post-resolve run would
    // read the auto-switch as a user click.
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    autoSwitchTo(h.marker, 'b.mp4');
    h.controller.run(inputs({ isProgressImageResolving: true, renderedItemName: 'b.mp4', selectedItemName: 'b.mp4' }));
    expect(h.isRevealed()).toBe(false);
    h.controller.run(rendering('b.mp4'));
    expect(h.isRevealed()).toBe(false);
  });

  it('reveals the selection made after the selection was cleared — including the same item', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    // Clearing the selection empties the viewer; the ref must not fall back to the "nothing has
    // rendered yet" state or the next click is treated as the viewer's first render and stays
    // hidden under the progress overlay.
    h.controller.run(inputs({ renderedItemName: null, selectedItemName: null }));
    h.controller.run(rendering('b.mp4'));
    expect(h.isRevealed()).toBe(true);

    const h2 = createHarness();
    h2.controller.run(rendering('a.png'));
    h2.controller.run(inputs({ renderedItemName: null, selectedItemName: null }));
    h2.controller.run(rendering('a.png'));
    expect(h2.isRevealed()).toBe(true);
  });

  it('keeps the previous item while a selection exists but its render has not landed', () => {
    // Image preloads and DTO fetches make the rendered item lag the selection; the in-between run
    // must neither reveal nor erase the fact of what was on screen before.
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(inputs({ renderedItemName: null, selectedItemName: 'b.png' }));
    expect(h.isRevealed()).toBe(false);
    expect(h.lastRenderedItemNameRef.current).toBe('a.png');
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(true);
  });

  it('does not reveal while the rendered item lags the selection', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(inputs({ renderedItemName: 'a.png', selectedItemName: 'b.png' }));
    expect(h.isRevealed()).toBe(false);
  });

  it('lowers the reveal when the overlay is not showing at all', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(inputs({ renderedItemName: 'b.png', selectedItemName: 'b.png', hasProgressImage: false }));
    expect(h.isRevealed()).toBe(false);
    const h2 = createHarness();
    h2.controller.run(rendering('a.png'));
    h2.controller.run(
      inputs({ renderedItemName: 'b.png', selectedItemName: 'b.png', shouldShowProgressInViewer: false })
    );
    expect(h2.isRevealed()).toBe(false);
  });

  it('keeps a reveal the user already earned when a generation elsewhere starts resolving', () => {
    // The clicked item is on screen with its two seconds running. Another session finishing is no
    // reason to slam the opaque overlay back over it.
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(true);

    h.controller.run({ ...rendering('b.png'), isProgressImageResolving: true });
    expect(h.isRevealed(), 'the granted reveal survives the resolve window').toBe(true);

    // ...and it still ends on its own rather than sticking there.
    h.fireTimers();
    expect(h.isRevealed()).toBe(false);
  });

  it('lowers during a resolve window when the in-flight reveal is for a different item', () => {
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    expect(h.isRevealed()).toBe(true);
    h.controller.run({ ...rendering('c.png'), isProgressImageResolving: true });
    expect(h.isRevealed()).toBe(false);
  });

  it('leaves no stale timer behind when a run supersedes a reveal', () => {
    // Two timers alive at once means the older one lowers the newer one's reveal early.
    const h = createHarness();
    h.controller.run(rendering('a.png'));
    h.controller.run(rendering('b.png'));
    h.controller.run(rendering('c.png'));
    expect(h.pendingTimerCount()).toBe(1);

    h.controller.run({ ...rendering('c.png'), isProgressImageResolving: true });
    expect(h.pendingTimerCount(), 'the resolve-window re-arm replaces the timer, not adds one').toBe(1);
  });
});
