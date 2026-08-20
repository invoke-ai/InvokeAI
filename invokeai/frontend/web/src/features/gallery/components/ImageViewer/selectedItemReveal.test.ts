import type { GallerySelectionDescriptor } from 'features/gallery/store/gallerySelectionSource';
import { describe, expect, it } from 'vitest';

import { createSelectedItemRevealMachine } from './selectedItemReveal';

const DURATION_MS = 2000;
const MEDIA_GRACE_MS = 1000;

/** Drives the machine the way the preview components' effects do, with hand-cranked timers. */
const createHarness = () => {
  let revealed = false;
  let generation = 0;
  let selection: GallerySelectionDescriptor = { name: null, generation: 0, isAutoSwitch: false };
  const timers = new Map<number, { fn: () => void; ms: number }>();
  let nextTimerId = 1;

  const machine = createSelectedItemRevealMachine({
    setRevealed: (value) => {
      revealed = value;
    },
    durationMs: DURATION_MS,
    mediaGraceMs: MEDIA_GRACE_MS,
    schedule: (fn, ms) => {
      const id = nextTimerId++;
      timers.set(id, { fn, ms });
      return id;
    },
    cancel: (id) => {
      timers.delete(id);
    },
  });

  const base = {
    renderedItemName: null as string | null,
    isMediaReady: false,
    shouldShowProgressInViewer: true,
    hasProgressImage: true,
    isProgressImageResolving: false,
  };

  return {
    machine,
    isRevealed: () => revealed,
    /** The provider's $gallerySelection subscription. */
    noteSelection: () => machine.noteSelection(selection),
    pendingTimerCount: () => timers.size,
    /** A selection dispatch, as the gallery source listener would publish it. */
    select: (name: string | null, options: { isAutoSwitch?: boolean } = {}) => {
      generation += 1;
      selection = { name, generation, isAutoSwitch: options.isAutoSwitch ?? false };
    },
    /** One effect run. */
    sync: (overrides: Partial<typeof base> = {}) => {
      machine.sync({ selection, ...base, ...overrides });
    },
    /** The viewer showing an item whose media has painted. */
    syncRendered: (itemName: string, overrides: Partial<typeof base> = {}) => {
      machine.sync({ selection, ...base, renderedItemName: itemName, isMediaReady: true, ...overrides });
    },
    fireTimers: () => {
      for (const [id, timer] of [...timers]) {
        timers.delete(id);
        timer.fn();
      }
    },
  };
};

describe('createSelectedItemRevealMachine', () => {
  it('reveals a mid-render click once its media has painted, then lowers on the timer', () => {
    const h = createHarness();
    h.select('b.png');
    h.sync();
    expect(h.isRevealed(), 'not before the media is ready').toBe(false);

    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(true);

    h.fireTimers();
    expect(h.isRevealed()).toBe(false);
  });

  it('does not lift the overlay onto an element that has not painted yet', () => {
    // The video element mounts immediately but shows black until it decodes a frame; revealing
    // then would replace the live preview with a black rectangle.
    const h = createHarness();
    h.select('b.mp4');
    h.sync({ renderedItemName: 'b.mp4', isMediaReady: false });
    expect(h.isRevealed()).toBe(false);
  });

  it('reveals anyway when the media never becomes ready', () => {
    // A failed load or an undecodable codec must not swallow the click for the whole render.
    const h = createHarness();
    h.select('broken.mp4');
    h.sync({ renderedItemName: 'broken.mp4', isMediaReady: false });
    h.fireTimers();
    expect(h.isRevealed()).toBe(true);
  });

  it('never reveals an auto-switch', () => {
    const h = createHarness();
    h.select('finished.png', { isAutoSwitch: true });
    h.syncRendered('finished.png');
    expect(h.isRevealed()).toBe(false);
  });

  it('reveals the item already on screen when the user picks it again', () => {
    // Nothing about the rendered item changes, so the previous-name comparison this replaces could
    // not see it — the click was simply dead.
    const h = createHarness();
    h.select('a.png');
    h.syncRendered('a.png');
    h.fireTimers();
    expect(h.isRevealed()).toBe(false);

    h.select('a.png');
    h.syncRendered('a.png');
    expect(h.isRevealed()).toBe(true);
  });

  it('does not reveal when the viewer simply opens on an existing selection', () => {
    // No selection was dispatched, so nothing is owed a reveal.
    const h = createHarness();
    h.syncRendered('a.png');
    expect(h.isRevealed()).toBe(false);
  });

  it('does not reveal a selection made while no progress was showing', () => {
    // The click was already visible; a later generation starting must not flash it.
    const h = createHarness();
    h.select('b.png');
    h.sync({ hasProgressImage: false });
    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(false);
  });

  it('holds a reveal owed during a resolve window until the window ends', () => {
    const h = createHarness();
    h.select('b.png');
    h.sync({ isProgressImageResolving: true });
    h.syncRendered('b.png', { isProgressImageResolving: true });
    expect(h.isRevealed(), 'the hand-off owns the viewer while it runs').toBe(false);

    h.syncRendered('b.png');
    expect(h.isRevealed(), 'and the click is honoured once it ends').toBe(true);
  });

  it('keeps a reveal already granted when a resolve window starts under it', () => {
    const h = createHarness();
    h.select('b.png');
    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(true);

    h.syncRendered('b.png', { isProgressImageResolving: true });
    expect(h.isRevealed()).toBe(true);
  });

  it('drops the reveal when the progress overlay goes away', () => {
    const h = createHarness();
    h.select('b.png');
    h.syncRendered('b.png');
    h.syncRendered('b.png', { hasProgressImage: false });
    expect(h.isRevealed()).toBe(false);
    expect(h.pendingTimerCount(), 'and takes its timer with it').toBe(0);
  });

  it('drops the reveal when the selection is cleared', () => {
    const h = createHarness();
    h.select('b.png');
    h.syncRendered('b.png');
    h.select(null);
    h.sync();
    expect(h.isRevealed()).toBe(false);
  });

  it('keeps at most one timer alive across a run of selections', () => {
    // Two live timers means the older one lowers the newer reveal early.
    const h = createHarness();
    h.select('a.png');
    h.syncRendered('a.png');
    h.select('b.png');
    h.syncRendered('b.png');
    h.select('c.png');
    h.syncRendered('c.png');
    expect(h.pendingTimerCount()).toBe(1);
  });

  it('is idempotent across repeated syncs of the same inputs (StrictMode double-invoke)', () => {
    const h = createHarness();
    h.select('b.png');
    h.syncRendered('b.png');
    const revealedAfterFirst = h.isRevealed();
    h.syncRendered('b.png');
    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(revealedAfterFirst);
    expect(h.pendingTimerCount()).toBe(1);
  });

  it('holds a claim that has not been shown when a hand-off begins', () => {
    // The reveal is owed but not yet visible: dropping it here would lose the click entirely,
    // since nothing else remembers it.
    const h = createHarness();
    h.select('b.png');
    h.sync({ renderedItemName: 'b.png', isMediaReady: false });
    h.sync({ renderedItemName: 'b.png', isMediaReady: false, isProgressImageResolving: true });
    expect(h.isRevealed()).toBe(false);

    // The media arrives during the window — still not shown, the hand-off owns the viewer.
    h.syncRendered('b.png', { isProgressImageResolving: true });
    expect(h.isRevealed()).toBe(false);

    // ...and it is honoured once the window ends.
    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(true);
  });

  it('does not reveal a superseded item when its media finally arrives', () => {
    // The slow video's frame lands after the user has moved on. Revealing it then would show them
    // an item they are no longer looking at, over a live preview.
    const h = createHarness();
    h.select('slow.mp4');
    h.sync({ renderedItemName: 'slow.mp4', isMediaReady: false });

    h.select('next.png');
    h.sync({ renderedItemName: 'slow.mp4', isMediaReady: false });

    h.syncRendered('slow.mp4');
    expect(h.isRevealed()).toBe(false);
  });

  it('settles a selection that lands while no preview is mounted, rather than replaying it', () => {
    // Comparison mode keeps the provider (and the progress preview) alive while unmounting both
    // preview components, so nothing syncs. Returning must not fire a reveal for a click made
    // while no overlay was covering anything.
    const h = createHarness();
    h.select('a.png');
    h.syncRendered('a.png');
    h.fireTimers();

    const detach = h.machine.attach();
    detach();
    h.select('picked-while-comparing.png');
    h.noteSelection();

    h.syncRendered('picked-while-comparing.png');
    expect(h.isRevealed()).toBe(false);
  });

  it('ignores noteSelection while a preview is mounted, leaving the decision to sync', () => {
    const h = createHarness();
    const detach = h.machine.attach();
    h.select('b.png');
    h.noteSelection();
    h.syncRendered('b.png');
    expect(h.isRevealed()).toBe(true);
    detach();
  });
});
