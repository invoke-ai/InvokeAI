/**
 * The viewer's "reveal selected item" mechanism, extracted from the preview components so its
 * sequencing is testable without rendering.
 *
 * A generation covers the viewer with an opaque progress overlay, which would otherwise swallow
 * every gallery click for the whole render: the selection changes underneath, but nothing visibly
 * happens. The reveal lifts the overlay for `durationMs` when the rendered item changes because
 * the *user* picked a new one, then drops back to the live preview.
 *
 * CurrentImagePreview and CurrentVideoPreview each own one controller and call `run` from the
 * effect that reacts to their rendered item; `lastRenderedItemNameRef` is shared between them via
 * the viewer context so a click that switches media type still reads as a selection change.
 *
 * Sequencing rules the controller encodes (each one has a failure mode without it):
 *
 * - The previous-item ref is NOT advanced, and the auto-switch marker NOT consumed, while a
 *   finished render's preview is resolving into its final frame. A click landing inside that
 *   window keeps its identity until the window ends, so the next run can still classify it —
 *   revealing a user click that resumes progress would otherwise be impossible, and an
 *   auto-switch would consume its marker on a run that can never reveal, then read as a user
 *   click afterwards.
 * - The marker IS consumed on every other change of the rendered item, even when no progress is
 *   showing: it must not outlive the render it was recorded for.
 * - Clearing the selection moves the ref to a sentinel rather than null: null means "nothing has
 *   rendered since the viewer opened" and suppresses the reveal (that first render is not a
 *   click), but the selection made after a clear IS a click and must reveal — including a
 *   re-selection of the very item that was cleared.
 * - A run that finds the rendered item unchanged while this controller's own reveal is the one
 *   in flight re-arms it instead of killing it. React StrictMode double-invokes a mount's
 *   effects (run → cleanup → run), so without this every cross-media first reveal dies in
 *   development: the second run sees the name the first run wrote into the shared ref.
 * - Every other path lowers the reveal. `run` cancels the running reveal's timer before
 *   deciding, so an outcome that left the flag raised would have nothing left to lower it.
 */

type SelectedItemRevealInputs = {
  /** The user's "show progress in viewer" setting. */
  shouldShowProgressInViewer: boolean;
  /** Whether a live progress preview exists to be covering the viewer at all. */
  hasProgressImage: boolean;
  /** Whether a finished generation's preview is mid-handoff to its final frame. */
  isProgressImageResolving: boolean;
  /** The item the component is rendering right now, or null when it has nothing to show. */
  renderedItemName: string | null;
  /** The item the gallery selection points at, or null when the selection is empty. */
  selectedItemName: string | null;
};

type SelectedItemRevealController = {
  /** Decide and apply the reveal state for the current render. Call from the effect body. */
  run: (inputs: SelectedItemRevealInputs) => void;
  /** Cancel the running reveal's timer without changing the revealed state. Call from the
   * effect's cleanup — the next `run` (or the component's unmount handler) owns the state. */
  clearTimer: () => void;
};

// Written into the shared ref when the selection is cleared after something had rendered. Never
// equals a real item name (names are filenames), never equals null (the nothing-rendered-yet
// state), so the next rendered item compares as a change and gets its reveal.
const SELECTION_CLEARED = '';

export const createSelectedItemRevealController = (deps: {
  /** Shared with the other preview component via the viewer context. */
  lastRenderedItemNameRef: { current: string | null };
  /** The auto-switch marker (see features/gallery/store/autoSwitchedImages). */
  marker: { consume: (itemName: string) => boolean };
  /** Writes the shared "temporarily showing selected item" flag. */
  setRevealed: (revealed: boolean) => void;
  durationMs: number;
  schedule?: (fn: () => void, ms: number) => number;
  cancel?: (id: number) => void;
}): SelectedItemRevealController => {
  const {
    lastRenderedItemNameRef,
    marker,
    setRevealed,
    durationMs,
    schedule = (fn, ms) => window.setTimeout(fn, ms),
    cancel = (id) => window.clearTimeout(id),
  } = deps;

  // The item this controller's currently-armed reveal is showing, null when none is in flight.
  // This is what lets a StrictMode re-run tell "my own reveal, restarted" from "nothing changed".
  let activeRevealItemName: string | null = null;
  let timerId = 0;

  const lower = () => {
    activeRevealItemName = null;
    setRevealed(false);
  };

  const run: SelectedItemRevealController['run'] = ({
    shouldShowProgressInViewer,
    hasProgressImage,
    isProgressImageResolving,
    renderedItemName,
    selectedItemName,
  }) => {
    cancel(timerId);

    // Resolve window: leave the ref and the marker exactly as they are (see the module docblock)
    // so whatever lands during it can still be classified — and revealed — when it ends.
    if (isProgressImageResolving) {
      if (activeRevealItemName !== null && activeRevealItemName === renderedItemName) {
        // A reveal the user already earned is in flight over this very item. A generation
        // finishing elsewhere is no reason to slam the overlay back over their click, so re-arm
        // (the timer was cancelled above) instead of lowering.
        timerId = schedule(lower, durationMs);
        return;
      }
      lower();
      return;
    }

    const previousRenderedItemName = lastRenderedItemNameRef.current;
    if (renderedItemName !== null) {
      lastRenderedItemNameRef.current = renderedItemName;
    } else if (selectedItemName === null && previousRenderedItemName !== null) {
      // Selection cleared. While a selection exists but its render hasn't landed yet (preload or
      // DTO fetch pending), the ref must keep the previous item — nulling it would erase the
      // "previous item" fact and swallow the reveal the successor run would fire.
      lastRenderedItemNameRef.current = SELECTION_CLEARED;
    }

    // Consumed before the visibility guards below: in the common case the auto-switched item
    // renders with no progress showing, and the marker must not outlive the render it was
    // recorded for.
    const wasAutoSwitchedTo =
      renderedItemName !== null && renderedItemName !== previousRenderedItemName && marker.consume(renderedItemName);

    if (
      !shouldShowProgressInViewer ||
      !hasProgressImage ||
      renderedItemName === null ||
      // Render lagging the selection: whatever is on screen is not what the user picked, so
      // showing it would not make their click land.
      renderedItemName !== selectedItemName
    ) {
      lower();
      return;
    }

    if (previousRenderedItemName === null || previousRenderedItemName === renderedItemName) {
      if (activeRevealItemName !== renderedItemName) {
        // Not a change of displayed item — nothing happened that needs to be made visible. The
        // first render after the viewer opens is not a click either.
        lower();
        return;
      }
      // This controller's own reveal of this exact item is in flight: a StrictMode re-run of the
      // mount effect (its cleanup already cancelled the timer, and the unmount handler lowered
      // the flag). Fall through and re-arm rather than killing the reveal the user just earned.
    } else if (wasAutoSwitchedTo) {
      // The reveal exists for *user* selections. The auto-switch selection lands after an async
      // DTO fetch, so the next render's first progress event can slot in ahead of it and reset
      // the resolving flag — by the time it reaches this effect, timing cannot tell it from a
      // click, but the marker can. Revealing it would flash the finished item over the new
      // render's live preview.
      lower();
      return;
    }

    activeRevealItemName = renderedItemName;
    setRevealed(true);
    timerId = schedule(lower, durationMs);
  };

  return {
    run,
    clearTimer: () => {
      cancel(timerId);
    },
  };
};
