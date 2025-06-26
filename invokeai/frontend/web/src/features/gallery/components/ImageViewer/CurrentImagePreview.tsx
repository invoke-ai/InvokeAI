import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectAutoSwitch } from 'features/gallery/store/gallerySelectors';
import type { ProgressImage as ProgressImageType } from 'features/nodes/types/common';
import { selectShouldShowImageDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { atom } from 'nanostores';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { ImageDTO, S } from 'services/api/types';
import { $socket } from 'services/events/stores';

import { ImageMetadataMini } from './ImageMetadataMini';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';

export const CurrentImagePreview = memo(({ imageDTO }: { imageDTO: ImageDTO | null }) => {
  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const autoSwitch = useAppSelector(selectAutoSwitch);

  const socket = useStore($socket);
  const $progressEvent = useState(() => atom<S['InvocationProgressEvent'] | null>(null))[0];
  const $progressImage = useState(() => atom<ProgressImageType | null>(null))[0];
  const progressImage = useStore($progressImage);
  const progressEvent = useStore($progressEvent);

  // Show and hide the next/prev buttons on mouse move
  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] = useState<boolean>(false);
  const timeoutId = useRef(0);
  const onMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
    window.clearTimeout(timeoutId.current);
  }, []);
  const onMouseOut = useCallback(() => {
    timeoutId.current = window.setTimeout(() => {
      setShouldShowNextPrevButtons(false);
    }, 500);
  }, []);

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

    if (autoSwitch) {
      return;
    }
    // When auto-switch is enabled, we will get a load event as we switch to the new image. This in turn clears the progress image,
    // creating the illusion of the progress image turning into the new image.
    // But when auto-switch is disabled, we won't get that load event, so we need to clear the progress image manually.
    const onQueueItemStatusChanged = () => {
      $progressEvent.set(null);
      $progressImage.set(null);
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

  const withProgress = shouldShowProgressInViewer && progressImage !== null;

  return (
    <Flex
      onMouseOver={onMouseOver}
      onMouseOut={onMouseOut}
      width="full"
      height="full"
      alignItems="center"
      justifyContent="center"
      position="relative"
    >
      {imageDTO && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center">
          <DndImage imageDTO={imageDTO} onLoad={onLoadImage} />
        </Flex>
      )}
      {!imageDTO && <NoContentForViewer />}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          <ProgressImage progressImage={progressImage} />
          {progressEvent && (
            <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
          )}
        </Flex>
      )}
      <Flex flexDir="column" gap={2} position="absolute" top={0} insetInlineStart={0} alignItems="flex-start">
        <CanvasAlertsInvocationProgress />
        {imageDTO && !withProgress && <ImageMetadataMini imageName={imageDTO.image_name} />}
      </Flex>
      {shouldShowImageDetails && imageDTO && !withProgress && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <ImageMetadataViewer image={imageDTO} />
        </Box>
      )}
      <AnimatePresence>
        {shouldShowNextPrevButtons && imageDTO && (
          <Box
            as={motion.div}
            key="nextPrevButtons"
            initial={initial}
            animate={animateArrows}
            exit={exit}
            position="absolute"
            top={0}
            right={0}
            bottom={0}
            left={0}
            pointerEvents="none"
          >
            <NextPrevImageButtons />
          </Box>
        )}
      </AnimatePresence>
    </Flex>
  );
});
CurrentImagePreview.displayName = 'CurrentImagePreview';

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animateArrows: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.07 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.07 },
};
