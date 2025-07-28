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
        if (data.status === 'canceled' || data.status === 'failed') {
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
