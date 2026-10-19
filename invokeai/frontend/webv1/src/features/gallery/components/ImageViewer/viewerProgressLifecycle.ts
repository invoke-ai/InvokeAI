import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import type { MapStore, WritableAtom } from 'nanostores';
import type { S } from 'services/api/types';

/** Live progress for a single in-flight session (queue item). Used to tile the viewer when several
 * sessions run concurrently (multi-GPU). Only items that have produced a preview image are tracked.
 * `seq` orders data by most recent update, so the shared single-image preview can be handed to the
 * freshest remaining session when its current owner terminates. */
export type ViewerProgressDatum = {
  itemId: number;
  seq: number;
  progressEvent: S['InvocationProgressEvent'];
  progressImage: ProgressImageType;
};

export type ViewerProgressDataMap = Record<number, ViewerProgressDatum | undefined>;

/** The subset of LRUCache the lifecycle needs — kept minimal so tests can pass a plain Map. */
type FinishedQueueItemIds = {
  has: (itemId: number) => boolean;
  set: (itemId: number, value: boolean) => unknown;
};

/** The subset of LRUCache the lifecycle needs — kept minimal so tests can pass a plain Map. */
type ItemIdBySessionId = {
  get: (sessionId: string) => number | undefined;
  set: (sessionId: string, itemId: number) => unknown;
  clear: () => unknown;
};

/**
 * How long a completed session's retained preview may wait for its final image to load before it
 * is cleared anyway. The "resolve" illusion normally ends on that image's load event, but nothing
 * guarantees the event arrives: the load can fail (the viewer's <Image> reports errors through
 * onError, not onLoad), or auto-switch can end up selecting a concurrently-completed session's
 * image instead, in which case the retained session's image is never rendered at all. Without this
 * bound the preview — an opaque overlay — could cover the viewer until the next generation.
 */
export const RESOLVE_TIMEOUT_MS = 3000;

export type ViewerProgressStores = {
  $progressEvent: WritableAtom<S['InvocationProgressEvent'] | null>;
  $progressImage: WritableAtom<ProgressImageType | null>;
  /** Per-session progress, keyed by queue item id. Drives the tiled multi-session preview. */
  $progressData: MapStore<ViewerProgressDataMap>;
  $isProgressImageResolving: WritableAtom<boolean>;
  /** Finished queue items, tracked so trailing progress events cannot repopulate the preview. */
  finishedQueueItemIds: FinishedQueueItemIds;
  /** Queue item id of each session we have seen progress for, keyed by session id. Outlives the
   * item's terminal event so a late final-image load can be attributed to the session that
   * produced it (see onFinalImageLoaded). */
  itemIdBySessionId: ItemIdBySessionId;
};

const pickLatestDatum = (data: ViewerProgressDataMap): ViewerProgressDatum | null => {
  let latest: ViewerProgressDatum | null = null;
  for (const datum of Object.values(data)) {
    if (datum !== undefined && (latest === null || datum.seq > latest.seq)) {
      latest = datum;
    }
  }
  return latest;
};

/**
 * The store-side lifecycle of the image viewer's live-preview state, factored out of the React
 * provider so it can be unit tested. The provider owns the socket subscriptions and the
 * ownership/scope checks on incoming events; every store mutation happens here.
 *
 * The state it manages:
 * - `$progressData`: one entry per session with a preview image (the tiled multi-session view).
 * - `$progressEvent` / `$progressImage`: the shared single-image preview, owned by the session
 *   that most recently reported progress.
 */
export const createViewerProgressLifecycle = (stores: ViewerProgressStores) => {
  const {
    $progressEvent,
    $progressImage,
    $progressData,
    $isProgressImageResolving,
    finishedQueueItemIds,
    itemIdBySessionId,
  } = stores;
  let seq = 0;
  // The queue item whose retained preview the final gallery image's onLoad should clear — the tail
  // end of the "resolve" illusion for a completed session (see onTerminal / onFinalImageLoaded).
  // Null when no illusion is pending. Always written through setPendingResolve, which keeps the
  // safety timeout in sync.
  let pendingResolveItemId: number | null = null;
  // Every item we have seen progress for and not yet seen terminate, including items that have not
  // produced a preview image (those are absent from $progressData). A queue clear deletes items
  // without emitting a per-item terminal event, so this is the set that must be marked finished
  // there — otherwise an image-less session could later emit an image and resurrect the preview.
  const unfinishedItemIds = new Set<number>();
  let resolveTimeoutId: ReturnType<typeof setTimeout> | null = null;

  const clearRetainedPreview = (): void => {
    $isProgressImageResolving.set(false);
    $progressEvent.set(null);
    $progressImage.set(null);
  };

  /**
   * Arm (or, with null, disarm) the pending "resolve" illusion. Every write to
   * `pendingResolveItemId` goes through here, so the safety timeout exists exactly while the
   * illusion is pending: anything that ends the illusion — a load, a takeover by another session,
   * a reset — also cancels the timeout, and it can never clear a preview that some other session
   * has since taken over.
   *
   * The timeout bounds the illusion only. A preview left standing by a path that does not arm one
   * (a progress event that carries no image replaces $progressEvent but not $progressImage, so the
   * previous session's frame stays up while the next queue item spins up) is unbounded here, as it
   * is today — that frame is taken down by the next preview image rather than by this timeout.
   */
  const setPendingResolve = (itemId: number | null): void => {
    if (resolveTimeoutId !== null) {
      clearTimeout(resolveTimeoutId);
      resolveTimeoutId = null;
    }
    pendingResolveItemId = itemId;
    if (itemId === null) {
      return;
    }
    resolveTimeoutId = setTimeout(() => {
      resolveTimeoutId = null;
      pendingResolveItemId = null;
      clearRetainedPreview();
    }, RESOLVE_TIMEOUT_MS);
  };

  const clearAll = (): void => {
    setPendingResolve(null);
    unfinishedItemIds.clear();
    // Session attributions describe state this reset just dropped. Keeping them would make later
    // loads of those images look like another session's, suppressing clears they should perform.
    itemIdBySessionId.clear();
    clearRetainedPreview();
    $progressData.set({});
  };

  /**
   * A worker claimed this queue item (`in_progress`). Tracked so a queue clear can mark it
   * finished: the clear cancels the items that were already running before it deletes the rows,
   * but a worker that claims an item in between gets no terminal event at all — its row is gone —
   * and its first progress event would otherwise appear seconds after the clear and put a preview
   * for a deleted item on screen, with nothing left to ever take it down.
   *
   * Returns false if the item already finished (event ignored).
   */
  const onItemStarted = (itemId: number): boolean => {
    if (finishedQueueItemIds.has(itemId)) {
      return false;
    }
    unfinishedItemIds.add(itemId);
    return true;
  };

  /** Record a progress event. Returns false if the item already finished (event ignored). */
  const recordProgress = (data: S['InvocationProgressEvent']): boolean => {
    if (finishedQueueItemIds.has(data.item_id)) {
      return false;
    }
    unfinishedItemIds.add(data.item_id);
    itemIdBySessionId.set(data.session_id, data.item_id);
    setPendingResolve(null);
    $isProgressImageResolving.set(false);
    $progressEvent.set(data);
    if (data.image) {
      $progressImage.set(data.image);
      // Track per-session so the viewer can tile concurrent sessions (multi-GPU).
      $progressData.setKey(data.item_id, {
        itemId: data.item_id,
        seq: ++seq,
        progressEvent: data,
        progressImage: data.image,
      });
    }
    return true;
  };

  /** Handle a terminal status for a queue item. Returns false if it already finished (ignored). */
  const onTerminal = (data: S['QueueItemStatusChangedEvent'], autoSwitch: boolean): boolean => {
    if (finishedQueueItemIds.has(data.item_id)) {
      return false;
    }
    finishedQueueItemIds.set(data.item_id, true);
    unfinishedItemIds.delete(data.item_id);
    // Remove this session's tile from the multi-session preview as soon as it reaches a terminal
    // state. The single-image "resolve" illusion below is handled separately via onLoadImage.
    $progressData.setKey(data.item_id, undefined);
    // The shared $progressEvent/$progressImage globals may currently hold a DIFFERENT session's
    // latest preview (multi-GPU). Only the item that owns them may replace or clear them —
    // otherwise canceling item A would blank item B's still-running preview until B's next image
    // event.
    const globalProgressEvent = $progressEvent.get();
    if (globalProgressEvent !== null && globalProgressEvent.item_id !== data.item_id) {
      return true;
    }
    const successor = pickLatestDatum($progressData.get());
    if (successor !== null) {
      // The terminated item owned the shared preview, but other sessions are still generating:
      // hand the preview to the most recently updated one immediately. The tiled view only renders
      // with more than one active session, so once a single session remains it is displayed
      // through these globals — leaving them cleared (or parked on the finished session's stale
      // frame via the resolve illusion) would hide a still-running preview. This applies to every
      // terminal status, including successful completion with auto-switch.
      setPendingResolve(null);
      $isProgressImageResolving.set(false);
      $progressEvent.set(successor.progressEvent);
      $progressImage.set(successor.progressImage);
      return true;
    }
    if (globalProgressEvent === null) {
      // Nothing is retained (this item never reported progress), so there is no illusion to run —
      // arming one would leave $isProgressImageResolving stuck on until the next generation.
      setPendingResolve(null);
      $isProgressImageResolving.set(false);
      return true;
    }
    // Completed queue items have the progress event cleared by the onLoadImage callback. This allows the viewer to
    // create the illusion of the progress image "resolving" into the final image. If we cleared the progress image
    // now, there would be a flicker where the progress image disappears before the final image appears, and the
    // last-selected gallery image should be shown for a brief moment.
    //
    // When gallery auto-switch is disabled, we do not need to create this illusion, because we are not going to
    // switch to the final image automatically. In this case, we clear the progress image immediately.
    //
    // We also clear the progress image if the queue item is canceled or failed, as there is no final image to show.
    if (
      data.status === 'canceled' ||
      data.status === 'failed' ||
      !autoSwitch ||
      // When the origin is 'canvas' and destination is 'canvas' (without a ':<session id>' suffix), that means the
      // image is going to be added to the staging area. In this case, we need to clear the progress image else it
      // will be stuck on the viewer.
      (data.origin === 'canvas' && data.destination !== 'canvas')
    ) {
      setPendingResolve(null);
      clearRetainedPreview();
    } else {
      setPendingResolve(data.item_id);
      $isProgressImageResolving.set(true);
    }
    return true;
  };

  /**
   * The final gallery image (or video) finished loading. If a completed session's "resolve"
   * illusion is pending, this is its tail end: the retained preview is cleared so the final image
   * shows. A no-op otherwise (e.g. when the preview was handed to a still-running session).
   *
   * `sessionId` identifies the item that was loaded (ImageDTO/VideoDTO `session_id`). Several
   * sessions run concurrently under multi-GPU and auto-switch, so a load can arrive late, after a
   * *different* session took over the retained preview: session A completes and hands the preview
   * to B, B then completes and starts its own resolve illusion, and only then does A's final image
   * finish loading. Clearing on A's load would cut B's illusion short — exactly the flicker the
   * illusion exists to hide — so a load is ignored when it can be positively attributed to another
   * session we tracked. Loads we cannot attribute (uploads, images from before this viewer
   * mounted) still clear.
   *
   * Ignoring a load must never be the difference between the preview clearing and not clearing:
   * the retained session's own image may never load at all (see RESOLVE_TIMEOUT_MS), and the
   * ignored load may have been the last one coming. The timeout armed alongside the illusion is
   * what bounds it — this check only decides whether the illusion ends early.
   */
  const onFinalImageLoaded = (sessionId: string | null): void => {
    if (pendingResolveItemId === null) {
      return;
    }
    if (sessionId !== null) {
      const loadedItemId = itemIdBySessionId.get(sessionId);
      if (loadedItemId !== undefined && loadedItemId !== pendingResolveItemId) {
        return;
      }
    }
    setPendingResolve(null);
    clearRetainedPreview();
  };

  /**
   * Handle a queue-cleared event. A clear deletes queue items without emitting a per-item terminal
   * status event for every one of them (a worker claimed mid-clear is stopped only by this event),
   * so the tracked previews must be dropped here. Which items were deleted depends on the event's
   * scope (mirroring workflowExecutionCoordinator.onQueueCleared): an unscoped clear (user_id=null
   * — an admin or single-user clear) deleted every item; a clear scoped to the current user
   * deleted all of this client's items; another user's scoped clear — received in full by admins
   * or as the sanitized user_id="redacted" broadcast by everyone else — deleted none of this
   * client's items, and this store only ever tracks the client's own items.
   *
   * Returns whether the clear applied to this client's previews.
   */
  const onQueueCleared = (data: S['QueueClearedEvent'], currentUserId: string | null): boolean => {
    const clearedUserId = data.user_id ?? null;
    if (clearedUserId !== null && clearedUserId !== currentUserId) {
      return false;
    }
    // Mark every session we have seen progress for and not yet seen terminate as finished, so a
    // trailing invocation_progress event from a worker that the clear is still stopping cannot
    // repopulate the preview. This must cover sessions that have not produced a preview image yet
    // — they are absent from $progressData and may not own the shared globals, but their first
    // image event would otherwise resurrect the preview after the clear.
    for (const itemId of unfinishedItemIds) {
      finishedQueueItemIds.set(itemId, true);
    }
    clearAll();
    return true;
  };

  /**
   * Drop all preview state without marking items finished. For socket disconnection and socket
   * replacement (auth-token/user change): the tracked sessions belong to the old connection and
   * will never emit another terminal event on this one.
   */
  const reset = (): void => {
    clearAll();
  };

  return { onFinalImageLoaded, onItemStarted, onQueueCleared, onTerminal, recordProgress, reset };
};
