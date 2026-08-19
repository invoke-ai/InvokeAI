import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import type {
  ViewerProgressDataMap,
  ViewerProgressDatum,
} from 'features/gallery/components/ImageViewer/viewerProgressLifecycle';
import { createViewerProgressLifecycle } from 'features/gallery/components/ImageViewer/viewerProgressLifecycle';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { LRUCache } from 'lru-cache';
import { type Atom, atom, computed, map, type MapStore, type WritableAtom } from 'nanostores';
import type { MutableRefObject, PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { S } from 'services/api/types';
import { getEventScope } from 'services/events/eventScope';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

export type { ViewerProgressDatum } from 'features/gallery/components/ImageViewer/viewerProgressLifecycle';

type ImageViewerContextValue = {
  $progressEvent: Atom<S['InvocationProgressEvent'] | null>;
  $progressImage: Atom<ProgressImageType | null>;
  $hasProgressImage: Atom<boolean>;
  /** Per-session progress, keyed by queue item id. Drives the tiled multi-session preview. */
  $progressData: MapStore<ViewerProgressDataMap>;
  /** Active sessions (those with a preview image), sorted by item id for a stable tile order. */
  $activeProgressData: Atom<ViewerProgressDatum[]>;
  $isProgressImageResolving: Atom<boolean>;
  $isTemporarilyShowingSelectedImage: WritableAtom<boolean>;
  /** Name of the item most recently rendered by either preview component (image or video). Shared
   * across the two components so a click that switches media type (image -> video or back) still
   * reads as a selection change to the temporary-reveal logic — a per-component ref resets on the
   * swap and would silently swallow the first reveal after every type switch. */
  lastRenderedItemNameRef: MutableRefObject<string | null>;
  /**
   * The viewer finished loading the final image/video for the given session (its DTO's
   * `session_id`, or null when it has none). Ends the completed session's "resolve" illusion.
   */
  onLoadImage: (sessionId: string | null) => void;
};

/** How long a mid-generation gallery click shows the clicked item before the live preview returns. */
export const SELECTED_ITEM_REVEAL_DURATION_MS = 2000;

const ImageViewerContext = createContext<ImageViewerContextValue | null>(null);

const log = logger('events');

export const ImageViewerContextProvider = memo((props: PropsWithChildren) => {
  const socket = useStore($socket);
  const store = useAppStore();
  const autoSwitch = useAppSelector(selectAutoSwitch);
  const $progressEvent = useState(() => atom<S['InvocationProgressEvent'] | null>(null))[0];
  const $progressImage = useState(() => atom<ProgressImageType | null>(null))[0];
  const $hasProgressImage = useState(() => computed($progressImage, (progressImage) => progressImage !== null))[0];
  // Per-session progress, keyed by queue item id, for the tiled multi-session preview (multi-GPU).
  const $progressData = useState(() => map<ViewerProgressDataMap>({}))[0];
  const $activeProgressData = useState(() =>
    computed($progressData, (progressData) =>
      Object.values(progressData)
        .filter((datum): datum is ViewerProgressDatum => datum !== undefined)
        .sort((a, b) => a.itemId - b.itemId)
    )
  )[0];
  const $isProgressImageResolving = useState(() => atom(false))[0];
  const $isTemporarilyShowingSelectedImage = useState(() => atom(false))[0];
  const lastRenderedItemNameRef = useRef<string | null>(null);
  // We can have race conditions where we receive a progress event for a queue item that has already finished. Easiest
  // way to handle this is to keep track of finished queue items in a cache and ignore progress events for those.
  const [finishedQueueItemIds] = useState(() => new LRUCache<number, boolean>({ max: 200 }));
  // Session id -> queue item id, learned from progress events. Outlives the item's terminal event
  // so a late final-image load can be attributed to the session that produced it.
  const [itemIdBySessionId] = useState(() => new LRUCache<string, number>({ max: 200 }));
  // All store mutations live in the lifecycle (extracted for unit testing); the effects below own
  // the socket subscriptions and the ownership/scope checks on incoming events.
  const lifecycle = useState(() =>
    createViewerProgressLifecycle({
      $progressEvent,
      $progressImage,
      $progressData,
      $isProgressImageResolving,
      finishedQueueItemIds,
      itemIdBySessionId,
    })
  )[0];

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onInvocationProgress = (data: S['InvocationProgressEvent']) => {
      // The backend routes progress events to the owner's room only; this check is defense in
      // depth, mirroring the invocation_progress listener in setEventListeners.
      if (getEventScope(store.getState, data) !== 'own') {
        return;
      }
      if (!lifecycle.recordProgress(data)) {
        log.trace(
          { data } as JsonObject,
          `Received InvocationProgressEvent event for already-finished queue item ${data.item_id}`
        );
      }
    };

    socket.on('invocation_progress', onInvocationProgress);

    return () => {
      socket.off('invocation_progress', onInvocationProgress);
    };
  }, [lifecycle, socket, store]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      // Other users' terminal status changes must not clear this client's live progress
      // preview. Both the sanitized companion (user_id="redacted", broadcast to every queue
      // subscriber) and foreign full events (received by admins via the admin room) carry a
      // real top-level item_id and terminal status, so without this guard they would drive
      // the terminal branch below and blank the viewer mid-generation.
      if (getEventScope(store.getState, data) !== 'own') {
        return;
      }
      if (data.status === 'in_progress') {
        // Track the claim itself, not just the progress events that follow it: a queue clear
        // cancels the items already running before it deletes the rows, so a worker that claims an
        // item in between never gets a terminal event and its first progress event lands after the
        // clear (see the lifecycle's onItemStarted).
        lifecycle.onItemStarted(data.item_id);
        return;
      }
      if (data.status !== 'completed' && data.status !== 'canceled' && data.status !== 'failed') {
        return;
      }
      if (!lifecycle.onTerminal(data, autoSwitch)) {
        log.trace(
          { data } as JsonObject,
          `Received QueueItemStatusChangedEvent event for already-finished queue item ${data.item_id}`
        );
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [autoSwitch, lifecycle, socket, store]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueCleared = (data: S['QueueClearedEvent']) => {
      // Scope is decided inside the lifecycle: it needs the current user id to tell whether the
      // clear could have deleted this client's items (see onQueueCleared's docstring).
      const currentUserId = selectCurrentUser(store.getState())?.user_id ?? null;
      lifecycle.onQueueCleared(data, currentUserId);
    };

    socket.on('queue_cleared', onQueueCleared);

    return () => {
      socket.off('queue_cleared', onQueueCleared);
    };
  }, [lifecycle, socket, store]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onDisconnect = () => {
      // Mirrors the app-wide progress stores (see setEventListeners): a disconnected socket may
      // miss terminal events, so previews from before the gap cannot be trusted to ever resolve.
      lifecycle.reset();
    };

    socket.on('disconnect', onDisconnect);

    return () => {
      socket.off('disconnect', onDisconnect);
      // The socket is being replaced (e.g. an auth-token change swapped $socket for a different
      // user's connection) or the viewer is unmounting: any tracked sessions belong to the old
      // connection and will never emit another terminal event here.
      lifecycle.reset();
    };
  }, [lifecycle, socket]);

  const onLoadImage = useCallback(
    (sessionId: string | null) => {
      lifecycle.onFinalImageLoaded(sessionId);
    },
    [lifecycle]
  );

  const value = useMemo(
    () => ({
      $progressEvent,
      $progressImage,
      $hasProgressImage,
      $progressData,
      $activeProgressData,
      $isProgressImageResolving,
      $isTemporarilyShowingSelectedImage,
      lastRenderedItemNameRef,
      onLoadImage,
    }),
    [
      $hasProgressImage,
      $progressData,
      $activeProgressData,
      $isProgressImageResolving,
      $isTemporarilyShowingSelectedImage,
      $progressEvent,
      $progressImage,
      onLoadImage,
    ]
  );

  return <ImageViewerContext.Provider value={value}>{props.children}</ImageViewerContext.Provider>;
});
ImageViewerContextProvider.displayName = 'ImageViewerContextProvider';

export const useImageViewerContext = () => {
  const value = useContext(ImageViewerContext);
  assert(value !== null, 'useImageViewerContext must be used within a ImageViewerContextProvider');
  return value;
};
