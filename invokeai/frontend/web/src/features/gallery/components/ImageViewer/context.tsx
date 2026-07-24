import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { LRUCache } from 'lru-cache';
import { type Atom, atom, computed, map, type MapStore, type WritableAtom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { S } from 'services/api/types';
import { getEventScope } from 'services/events/eventScope';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

/** Live progress for a single in-flight session (queue item). Used to tile the viewer when several
 * sessions run concurrently (multi-GPU). Only items that have produced a preview image are tracked. */
export type ViewerProgressDatum = {
  itemId: number;
  progressEvent: S['InvocationProgressEvent'];
  progressImage: ProgressImageType;
};

type ViewerProgressDataMap = Record<number, ViewerProgressDatum | undefined>;

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
  onLoadImage: () => void;
};

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
  const shouldClearProgressImageOnLoadRef = useRef(false);
  // We can have race conditions where we receive a progress event for a queue item that has already finished. Easiest
  // way to handle this is to keep track of finished queue items in a cache and ignore progress events for those.
  const [finishedQueueItemIds] = useState(() => new LRUCache<number, boolean>({ max: 200 }));

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
      if (finishedQueueItemIds.has(data.item_id)) {
        log.trace(
          { data } as JsonObject,
          `Received InvocationProgressEvent event for already-finished queue item ${data.item_id}`
        );
        return;
      }
      shouldClearProgressImageOnLoadRef.current = false;
      $isProgressImageResolving.set(false);
      $progressEvent.set(data);
      if (data.image) {
        $progressImage.set(data.image);
        // Track per-session so the viewer can tile concurrent sessions (multi-GPU).
        $progressData.setKey(data.item_id, {
          itemId: data.item_id,
          progressEvent: data,
          progressImage: data.image,
        });
      }
    };

    socket.on('invocation_progress', onInvocationProgress);

    return () => {
      socket.off('invocation_progress', onInvocationProgress);
    };
  }, [$isProgressImageResolving, $progressData, $progressEvent, $progressImage, finishedQueueItemIds, socket, store]);

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
      if (finishedQueueItemIds.has(data.item_id)) {
        log.trace(
          { data } as JsonObject,
          `Received QueueItemStatusChangedEvent event for already-finished queue item ${data.item_id}`
        );
        return;
      }
      if (data.status === 'completed' || data.status === 'canceled' || data.status === 'failed') {
        finishedQueueItemIds.set(data.item_id, true);
        // Remove this session's tile from the multi-session preview as soon as it reaches a terminal
        // state. The single-image "resolve" illusion below is handled separately via onLoadImage.
        $progressData.setKey(data.item_id, undefined);
        // The shared $progressEvent/$progressImage globals may currently hold a DIFFERENT session's
        // latest preview (multi-GPU). Only the item that owns them may clear them — otherwise
        // canceling item A would blank item B's still-running preview until B's next image event.
        const globalProgressEvent = $progressEvent.get();
        if (globalProgressEvent !== null && globalProgressEvent.item_id !== data.item_id) {
          return;
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
          shouldClearProgressImageOnLoadRef.current = false;
          $isProgressImageResolving.set(false);
          $progressEvent.set(null);
          $progressImage.set(null);
        } else {
          shouldClearProgressImageOnLoadRef.current = true;
          $isProgressImageResolving.set(true);
        }
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [
    $isProgressImageResolving,
    $progressData,
    $progressEvent,
    $progressImage,
    autoSwitch,
    finishedQueueItemIds,
    socket,
    store,
  ]);

  const onLoadImage = useCallback(() => {
    if (!shouldClearProgressImageOnLoadRef.current) {
      return;
    }

    shouldClearProgressImageOnLoadRef.current = false;
    $isProgressImageResolving.set(false);
    $progressEvent.set(null);
    $progressImage.set(null);
  }, [$isProgressImageResolving, $progressEvent, $progressImage]);

  const value = useMemo(
    () => ({
      $progressEvent,
      $progressImage,
      $hasProgressImage,
      $progressData,
      $activeProgressData,
      $isProgressImageResolving,
      $isTemporarilyShowingSelectedImage,
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
