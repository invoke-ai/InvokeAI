import type { S } from 'services/api/types';

/**
 * Backstop for the deferred progress-image clear.
 *
 * A completed queue item hands responsibility for clearing the viewer's progress preview to the
 * final image's load callback, so the preview appears to resolve into the finished image instead of
 * flickering through the previously-selected one. That callback is not guaranteed to fire:
 * - the image request can fail, which Chakra reports as `onError`, not `onLoad`;
 * - the finished image can already be the one on screen, and Chakra's `useImage` only re-runs when
 *   `src` changes;
 * - the load can beat the terminal queue event, leaving nothing to trigger the clear afterwards;
 * - the item's outputs can all be intermediate, so no selection change ever happens.
 *
 * Any of those wedges the opaque overlay over the finished image until the page is reloaded, so the
 * armed state needs a deadline.
 *
 * Deliberately long. This is a backstop against a state that would otherwise be permanent, not a
 * latency target: the reveal is gated on the thumbnail, so the normal path resolves in well under a
 * second. Firing early is its own regression — it replaces the preview with the previously-selected
 * gallery image, or with nothing at all — so the deadline sits well beyond how long a ~20KB
 * thumbnail can plausibly take, even on a bad connection.
 */
export const PROGRESS_IMAGE_RESOLVE_TIMEOUT_MS = 30_000;

type TerminalProgressAction =
  /** Clear the progress preview now. */
  | 'clear'
  /** Defer the clear until the final image loads, or until the backstop above fires. */
  | 'arm'
  /** This event does not own the shared progress preview — leave it alone. */
  | 'ignore';

/** The fields of a terminal `queue_item_status_changed` event the decision depends on. */
type TerminalQueueItemEvent = Pick<S['QueueItemStatusChangedEvent'], 'item_id' | 'status' | 'origin' | 'destination'>;

type TerminalProgressActionOptions = {
  /** Whether the gallery auto-switches to the finished image. */
  autoSwitch: boolean;
  /** The item id currently owning the shared progress atoms, or null when they are unset. */
  globalProgressItemId: number | null;
};

/**
 * Decides what a terminal `queue_item_status_changed` event should do to the viewer's progress
 * preview. Pure, so the branchy policy is testable without a socket or a React tree — the caller
 * owns the side effects.
 */
export const getTerminalProgressAction = (
  data: TerminalQueueItemEvent,
  { autoSwitch, globalProgressItemId }: TerminalProgressActionOptions
): TerminalProgressAction => {
  // The shared progress atoms may currently hold a DIFFERENT session's latest preview (multi-GPU).
  // Only the item that owns them may clear them — otherwise canceling item A would blank item B's
  // still-running preview until B's next image event.
  if (globalProgressItemId !== null && globalProgressItemId !== data.item_id) {
    return 'ignore';
  }

  // Nothing is going to load in place of the preview, so there is no resolve illusion to create.
  if (data.status === 'canceled' || data.status === 'failed') {
    return 'clear';
  }

  // Auto-switch off means the viewer is not going to show the finished image at all.
  if (!autoSwitch) {
    return 'clear';
  }

  // Origin 'canvas' with a destination that is not 'canvas' (i.e. without a ':<session id>' suffix)
  // means the image is bound for the staging area rather than this viewer, so nothing will ever
  // load here to clear the preview.
  if (data.origin === 'canvas' && data.destination !== 'canvas') {
    return 'clear';
  }

  return 'arm';
};

type DeferredClear = {
  /**
   * Arms the deferred clear, replacing any deadline already pending. `onDeadline` runs only if
   * nothing disarms first.
   */
  arm: (onDeadline: () => void) => void;
  /** Cancels a pending deadline. Safe to call when nothing is armed. */
  disarm: () => void;
  /** True between `arm` and the next `disarm`, or until the deadline fires. */
  isArmed: () => boolean;
};

/**
 * Owns the armed flag and its backstop timer as one unit.
 *
 * Keeping them together is the point: when they were two independent pieces of state, a path that
 * reset the flag but forgot the timer left the deadline running past the generation that armed it,
 * so it fired later and blanked a subsequent generation's live preview. Arming always supersedes
 * the previous deadline rather than stacking, and disarming always cancels it.
 *
 * Uses bare `setTimeout`/`clearTimeout` rather than the `window` members so this stays usable — and
 * testable with fake timers — outside a DOM.
 */
export const createDeferredClear = (timeoutMs: number = PROGRESS_IMAGE_RESOLVE_TIMEOUT_MS): DeferredClear => {
  let armed = false;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const disarm = () => {
    armed = false;
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  const arm = (onDeadline: () => void) => {
    disarm();
    armed = true;
    timeoutId = setTimeout(() => {
      timeoutId = null;
      armed = false;
      onDeadline();
    }, timeoutMs);
  };

  return { arm, disarm, isArmed: () => armed };
};
