import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { type Atom, atom, computed } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';

type ImageViewerContextValue = {
  $progressEvent: Atom<S['InvocationProgressEvent'] | null>;
  $progressImage: Atom<ProgressImageType | null>;
  $hasProgressImage: Atom<boolean>;
  onLoadImage: () => void;
};

const ImageViewerContext = createContext<ImageViewerContextValue | null>(null);

export const ImageViewerContextProvider = memo((props: PropsWithChildren) => {
  const socket = useStore($socket);
  const autoSwitch = useAppSelector(selectAutoSwitch);
  const $progressEvent = useState(() => atom<S['InvocationProgressEvent'] | null>(null))[0];
  const $progressImage = useState(() => atom<ProgressImageType | null>(null))[0];
  const $hasProgressImage = useState(() => computed($progressImage, (progressImage) => progressImage !== null))[0];

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onInvocationProgress = (data: S['InvocationProgressEvent']) => {
      $progressEvent.set(data);
      if (data.image) {
        $progressImage.set(data.image);
      }
    };

    socket.on('invocation_progress', onInvocationProgress);

    return () => {
      socket.off('invocation_progress', onInvocationProgress);
    };
  }, [$progressEvent, $progressImage, socket]);

  useEffect(() => {
    if (!socket) {
      return;
    }

    const onQueueItemStatusChanged = (data: S['QueueItemStatusChangedEvent']) => {
      // When auto-switch is enabled, we will get a load event as we switch to the new image. This in turn clears the progress image,
      // creating the illusion of the progress image turning into the new image.
      // But when auto-switch is disabled, we won't get that load event, so we need to clear the progress image manually.
      if (data.origin === 'canvas' || !autoSwitch) {
        $progressEvent.set(null);
        $progressImage.set(null);
      }
    };

    socket.on('queue_item_status_changed', onQueueItemStatusChanged);

    return () => {
      socket.off('queue_item_status_changed', onQueueItemStatusChanged);
    };
  }, [$progressEvent, $progressImage, autoSwitch, socket]);

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
