import type { GallerySelectionDescriptor } from 'features/gallery/store/gallerySelectionSource';

/**
 * The viewer's "reveal the selected item" mechanism, as one item-owned state machine.
 *
 * A generation covers the viewer with an opaque progress overlay, which would otherwise swallow
 * every gallery click for the whole render: the selection changes underneath, but nothing visibly
 * happens. The reveal lifts the overlay for `durationMs` when the *user* picks something, then
 * drops back to the live preview.
 *
 * The machine is owned by the viewer context, so both preview components drive one instance and a
 * click that switches media type is just another selection. What used to carry that across the
 * component swap — a shared mutable ref holding the previously rendered name, a sentinel value for
 * "the selection was cleared", and one controller per component — is gone: a selection is identified
 * by `generation`, so "the user picked X", "X has been on screen for a while" and "the user picked X
 * again" are three distinguishable events rather than three readings of one string comparison.
 *
 * A reveal is owed from the moment the selection lands, but it is only *shown* once that item's
 * media can actually be seen. Lifting the overlay onto an element that has not decoded a frame yet
 * shows the user a black rectangle where their click should be — so the machine waits for the item
 * to be rendered and for `isMediaReady`. Once the item is rendered, the wait for its media is
 * bounded by `mediaGraceMs`, so that media which never becomes ready (a failed load, a codec the
 * browser will not decode) still makes the click land rather than swallowing it. Until the item
 * renders there is nothing sensible to lift the overlay onto — only the previous item or a blank
 * element — so the claim stays outstanding instead of running against a deadline.
 */
type SelectedItemRevealInputs = {
  /** Who selected what, and which selection it is (see gallerySelectionSource). */
  selection: GallerySelectionDescriptor;
  /** The item the component is rendering right now, or null when it has nothing to show. */
  renderedItemName: string | null;
  /** Whether the rendered item has a frame on screen: an image decoded, a video's first frame
   * painted. False while the element is still black. */
  isMediaReady: boolean;
  /** The user's "show progress in viewer" setting. */
  shouldShowProgressInViewer: boolean;
  /** Whether a live progress preview is covering the viewer at all. */
  hasProgressImage: boolean;
  /** Whether a finished generation's preview is mid-handoff to its final frame. */
  isProgressImageResolving: boolean;
};

type MachineState =
  /** Nothing revealed, and nothing owed: this selection has had its turn, or never earned one. */
  | { kind: 'idle'; generation: number }
  /** Owed a reveal, but a finished generation is resolving into its final frame and owns the
   * viewer for the moment. Resumes when that window ends. */
  | { kind: 'deferred'; generation: number; itemName: string }
  /** Owed a reveal, waiting for the item to render and its media to paint. The grace deadline
   * runs only while the item is rendered. */
  | { kind: 'awaiting-media'; generation: number; itemName: string }
  /** Showing the selected item; the duration timer is running. */
  | { kind: 'revealing'; generation: number; itemName: string };

export type SelectedItemRevealMachine = {
  /** Fold the current inputs in. Called from the preview components' effects. */
  sync: (inputs: SelectedItemRevealInputs) => void;
  /**
   * Registers a mounted preview component; the returned function unregisters it. While none is
   * registered nothing calls `sync`, so selections have to be settled through `noteSelection`
   * instead.
   */
  attach: () => () => void;
  /**
   * A selection landed. Ignored while a preview is attached — its next `sync` carries the same
   * descriptor and decides properly, with the progress state this cannot see. While none is
   * attached (the viewer is in comparison mode, say) this settles the generation, so returning to
   * the viewer does not fire a reveal for a click made when no overlay was covering anything.
   */
  noteSelection: (selection: GallerySelectionDescriptor) => void;
  /** Drop any reveal and cancel timers. For provider teardown. */
  reset: () => void;
  /** The current state's kind, for tests and debugging. */
  peek: () => MachineState['kind'];
};

export const createSelectedItemRevealMachine = (deps: {
  setRevealed: (revealed: boolean) => void;
  durationMs: number;
  /** How long a reveal waits for its rendered item's media before showing anyway. */
  mediaGraceMs: number;
  schedule?: (fn: () => void, ms: number) => number;
  cancel?: (id: number) => void;
}): SelectedItemRevealMachine => {
  const {
    setRevealed,
    durationMs,
    mediaGraceMs,
    schedule = (fn, ms) => window.setTimeout(fn, ms),
    cancel = (id) => window.clearTimeout(id),
  } = deps;

  let state: MachineState = { kind: 'idle', generation: 0 };
  let timerId = 0;
  let attachedPreviews = 0;

  const clearTimer = () => {
    cancel(timerId);
    timerId = 0;
  };

  const enter = (next: MachineState) => {
    state = next;
    setRevealed(next.kind === 'revealing');
  };

  const goIdle = (generation: number) => {
    clearTimer();
    enter({ kind: 'idle', generation });
  };

  const goDeferred = (generation: number, itemName: string) => {
    clearTimer();
    enter({ kind: 'deferred', generation, itemName });
  };

  const startRevealing = (generation: number, itemName: string) => {
    clearTimer();
    enter({ kind: 'revealing', generation, itemName });
    timerId = schedule(() => {
      timerId = 0;
      goIdle(generation);
    }, durationMs);
  };

  const startAwaitingMedia = (generation: number, itemName: string) => {
    clearTimer();
    enter({ kind: 'awaiting-media', generation, itemName });
  };

  const sync: SelectedItemRevealMachine['sync'] = (inputs) => {
    const { selection, renderedItemName, isMediaReady } = inputs;
    const { shouldShowProgressInViewer, hasProgressImage, isProgressImageResolving } = inputs;

    // Nothing is covering the viewer, so there is nothing to reveal from under — and no reveal to
    // keep alive. Settling the generation here is what keeps a selection made while the overlay was
    // down from firing a reveal when the next generation raises it.
    if (!shouldShowProgressInViewer || !hasProgressImage) {
      goIdle(selection.generation);
      return;
    }

    if (selection.generation !== state.generation) {
      // A selection landed.
      if (selection.name === null || selection.isAutoSwitch) {
        // An auto-switch is the gallery handing over a finished item, not the user asking to see
        // something: revealing it would flash that item over the next render's live preview.
        goIdle(selection.generation);
      } else if (isProgressImageResolving) {
        goDeferred(selection.generation, selection.name);
      } else {
        startAwaitingMedia(selection.generation, selection.name);
      }
    } else if (state.kind === 'deferred' && !isProgressImageResolving) {
      // The hand-off finished; the reveal this selection is owed can start.
      startAwaitingMedia(state.generation, state.itemName);
    } else if (state.kind === 'awaiting-media' && isProgressImageResolving) {
      // A hand-off began while we were waiting for the media. Hold the claim — dropping it would
      // lose a click that has not been shown yet.
      goDeferred(state.generation, state.itemName);
    }

    // Show it the moment the owed item's media is actually on screen. Evaluated in the same pass
    // that entered the state, so a click on an item already rendered and decoded reveals at once.
    if (state.kind === 'awaiting-media' && renderedItemName === state.itemName) {
      if (isMediaReady) {
        startRevealing(state.generation, state.itemName);
      } else if (timerId === 0) {
        // The item is rendered but has not painted a frame. Media that never will (a failed load,
        // a codec the browser cannot decode) must not swallow the click, so the wait is bounded —
        // but the deadline starts here, not at the selection: counted from the click, it would
        // lift the overlay onto whatever was still showing while the item itself was slow to
        // arrive, and then leave the real item covered once it landed. No generation check is
        // needed in the callback — every transition clears this timer first, so it can only fire
        // while this exact claim is still the one outstanding. It is deliberately not cancelled
        // if `renderedItemName` later disagrees: with both previews mounted mid-swap, the stale
        // component's sync must not kill the deadline the matching one armed.
        const { generation, itemName } = state;
        timerId = schedule(() => {
          timerId = 0;
          if (state.kind === 'awaiting-media') {
            startRevealing(generation, itemName);
          }
        }, mediaGraceMs);
      }
    }

    // A reveal already granted runs out its timer even if a hand-off starts under it: that would
    // slam the overlay back over a click the user has already earned.
  };

  return {
    sync,
    attach: () => {
      attachedPreviews += 1;
      return () => {
        attachedPreviews -= 1;
      };
    },
    noteSelection: (selection) => {
      if (attachedPreviews > 0) {
        return;
      }
      goIdle(selection.generation);
    },
    reset: () => {
      clearTimer();
      enter({ kind: 'idle', generation: state.generation });
    },
    peek: () => state.kind,
  };
};
