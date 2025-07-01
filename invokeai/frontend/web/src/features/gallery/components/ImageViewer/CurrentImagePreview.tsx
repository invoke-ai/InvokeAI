import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectShouldShowImageDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useRef, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';

export const CurrentImagePreview = memo(({ imageDTO }: { imageDTO: ImageDTO | null }) => {
  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { onLoadImage, $progressEvent, $progressImage } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);

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
