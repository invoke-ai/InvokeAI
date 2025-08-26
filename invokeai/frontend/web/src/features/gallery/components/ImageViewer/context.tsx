import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { LRUCache } from 'lru-cache';
import { type Atom, atom, computed } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import type { JsonObject } from 'type-fest';

type ImageViewerContextValue = {
  $progressEvent: Atom<S['InvocationProgressEvent'] | null>;
  $progressImage: Atom<ProgressImageType | null>;
  $hasProgressImage: Atom<boolean>;
  onLoadImage: () => void;
};

const ImageViewerContext = createContext<ImageViewerContextValue | null>(null);

const log = logger('events');

export const ImageViewerContextProvider = memo((props: PropsWithChildren) => {
  const socket = useStore($socket);
  const autoSwitch = useAppSelector(selectAutoSwitch);
  const $progressEvent = useState(() => atom<S['InvocationProgressEvent'] | null>(null))[0];
  const $progressImage = useState(() => atom<ProgressImageType | null>(null))[0];
  const $hasProgressImage = useState(() => computed($progressImage, (progressImage) => progressImage !== null))[0];
  // We can have race conditions where we receive a progress event for a queue item that has already finished. Easiest
  // way to handle this is to keep track of finished queue items in a cache and ignore progress events for those.
  const [finishedQueueItemIds] = useState(() => new LRUCache<number, boolean>({ max: 200 }));

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onInvocationProgress = (data: S['InvocationProgressEvent']) => {
      if (finishedQueueItemIds.has(data.item_id)) {
        log.trace(
          { data } as JsonObject,
          `Received InvocationProgressEvent event for already-finished queue item ${data.item_id}`
        );
        return;
      }
      $progressEvent.set(data);
      if (data.image) {
        $progressImage.set(data.image);
      }
    };

    socket.on('invocation_progress', onInvocationProgress);

    return () => {
      socket.off('invocation_progress', onInvocationProgress);
    };
  }, [$progressEvent, $progressImage, finishedQueueItemIds, socket]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      if (finishedQueueItemIds.has(data.item_id)) {
        log.trace(
          { data } as JsonObject,
          `Received QueueItemStatusChangedEvent event for already-finished queue item ${data.item_id}`
        );
        return;
      }
      if (data.status === 'completed' || data.status === 'canceled' || data.status === 'failed') {
        finishedQueueItemIds.set(data.item_id, true);
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
          $progressEvent.set(null);
          $progressImage.set(null);
        }
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [$progressEvent, $progressImage, autoSwitch, finishedQueueItemIds, socket]);

  const onLoadImage = useCallback(() => {
    $progressEvent.set(null);
    $progressImage.set(null);
  }, [$progressEvent, $progressImage]);

  const value = useMemo(
    () => ({ $progressEvent, $progressImage, $hasProgressImage, onLoadImage }),
    [$hasProgressImage, $progressEvent, $progressImage, onLoadImage]
  );

  return <ImageViewerContext.Provider value={value}>{props.children}</ImageViewerContext.Provider>;
});
ImageViewerContextProvider.displayName = 'ImageViewerContextProvider';

export const useImageViewerContext = () => {
  const value = useContext(ImageViewerContext);
  assert(value !== null, 'useImageViewerContext must be used within a ImageViewerContextProvider');
  return value;
};
