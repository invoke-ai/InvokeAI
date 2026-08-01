import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { LRUCache } from 'lru-cache';
import { type Atom, atom, computed, map, type MapStore, type WritableAtom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { S } from 'services/api/types';
import { getEventScope } from 'services/events/eventScope';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { createDeferredClear, getTerminalProgressAction } from './progressImageResolution';

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
  // Owns both the "clear on load" flag and its backstop timer, so no path can reset one and leak
  // the other. See createDeferredClear.
  const [deferredClear] = useState(() => createDeferredClear());
  // We can have race conditions where we receive a progress event for a queue item that has already finished. Easiest
  // way to handle this is to keep track of finished queue items in a cache and ignore progress events for those.
  const [finishedQueueItemIds] = useState(() => new LRUCache<number, boolean>({ max: 200 }));

  // Cancels a pending deferred clear without touching the preview itself. Every path that takes
  // responsibility for the preview away from the armed onLoadImage must call this, or the backstop
  // outlives the generation that armed it and blanks a later one's live preview.
  const disarmDeferredClear = useCallback(() => {
    deferredClear.disarm();
    $isProgressImageResolving.set(false);
  }, [$isProgressImageResolving, deferredClear]);

  const clearProgressImage = useCallback(() => {
    disarmDeferredClear();
    $progressEvent.set(null);
    $progressImage.set(null);
  }, [disarmDeferredClear, $progressEvent, $progressImage]);

  // Nulling $progressImage tears down the whole overlay, tiles included — $activeProgressData only
  // renders while it is set. So when other sessions are still producing previews (multi-GPU), the
  // backstop must not clear: the overlay has already stopped being this item's to own. Disarming is
  // enough; those sessions clear it via their own terminal events.
  const onResolveDeadline = useCallback(() => {
    if ($activeProgressData.get().length > 0) {
      disarmDeferredClear();
      return;
    }
    clearProgressImage();
  }, [$activeProgressData, clearProgressImage, disarmDeferredClear]);

  useEffect(() => {
    return () => {
      deferredClear.disarm();
    };
  }, [deferredClear]);

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
      // A new preview supersedes any deferred clear still armed by the previous queue item, whose
      // final image may never have loaded. Leaving its backstop running would blank this preview
      // mid-generation.
      disarmDeferredClear();
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
  }, [$progressData, $progressEvent, $progressImage, disarmDeferredClear, finishedQueueItemIds, socket, store]);

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

        // See getTerminalProgressAction for why each outcome is chosen. 'arm' defers the clear to
        // onLoadImage so the viewer can create the illusion of the progress image "resolving" into
        // the final image — clearing it here instead would flicker through the previously-selected
        // gallery image before the final one appears.
        const action = getTerminalProgressAction(data, {
          autoSwitch,
          globalProgressItemId: $progressEvent.get()?.item_id ?? null,
        });

        if (action === 'ignore') {
          return;
        }

        if (action === 'clear') {
          clearProgressImage();
          return;
        }

        $isProgressImageResolving.set(true);
        // onLoadImage is not guaranteed to fire — see PROGRESS_IMAGE_RESOLVE_TIMEOUT_MS. Without
        // this deadline the overlay can cover the finished image until the page is reloaded.
        deferredClear.arm(onResolveDeadline);
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
    autoSwitch,
    clearProgressImage,
    deferredClear,
    finishedQueueItemIds,
    onResolveDeadline,
    socket,
    store,
  ]);

  // The viewer's progress atoms are separate stores from the global ones in services/events/stores,
  // which setEventListeners already resets on every socket lifecycle transition. Without the same
  // reset here the two diverge: socket.io has no event replay, so a drop spanning the terminal
  // queue_item_status_changed loses that event permanently and nothing is left to clear the opaque
  // overlay covering the finished image. Backgrounding a tab long enough for the connection to be
  // torn down is the common way to hit this.
  //
  // Clearing on disconnect — not just on reconnect — matches the progress *bars*, which already
  // vanish then. If the generation is in fact still running, the next invocation_progress event
  // repopulates the preview within a step.
  useEffect(() => {
    if (!socket) {
      return;
    }

    const onSocketLifecycleChange = () => {
      clearProgressImage();
      // connect_error fires once per reconnection attempt, i.e. roughly once a second while the
      // server is down. `set` compares by reference, so an unconditional `set({})` would notify
      // every subscriber on every attempt; only replace the map when it actually holds something.
      if (Object.keys($progressData.get()).length > 0) {
        $progressData.set({});
      }
    };

    socket.on('connect', onSocketLifecycleChange);
    socket.on('connect_error', onSocketLifecycleChange);
    socket.on('disconnect', onSocketLifecycleChange);

    return () => {
      socket.off('connect', onSocketLifecycleChange);
      socket.off('connect_error', onSocketLifecycleChange);
      socket.off('disconnect', onSocketLifecycleChange);
    };
  }, [$progressData, clearProgressImage, socket]);

  const onLoadImage = useCallback(() => {
    if (!deferredClear.isArmed()) {
      return;
    }

    clearProgressImage();
  }, [clearProgressImage, deferredClear]);

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
