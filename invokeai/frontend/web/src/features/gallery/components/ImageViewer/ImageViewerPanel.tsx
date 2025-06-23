import { Divider, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { type Atom, atom, computed } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ImageDTO, S } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';

export const ImageViewerPanel = memo(() => {
  return (
    <ImageViewerContextProvider>
      <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
        <ViewerToolbar />
        <Divider />
        <ImageViewer />
      </Flex>
    </ImageViewerContextProvider>
  );
});
ImageViewerPanel.displayName = 'ImageViewerPanel';

type ImageViewerContextValue = {
  $progressEvent: Atom<S['InvocationProgressEvent'] | null>;
  $progressImage: Atom<ProgressImageType | null>;
  $hasProgressImage: Atom<boolean>;
  onLoadImage: (imageDTO: ImageDTO) => void;
};

const ImageViewerContext = createContext<ImageViewerContextValue | null>(null);

const ImageViewerContextProvider = memo((props: PropsWithChildren) => {
  const socket = useStore($socket);
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

  const onLoadImage = useCallback(
    (imageDTO: ImageDTO) => {
      const progressEvent = $progressEvent.get();
      if (!progressEvent || !imageDTO) {
        return;
      }
      if (progressEvent.session_id === imageDTO.session_id) {
        $progressImage.set(null);
      }
    },
    [$progressEvent, $progressImage]
  );

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
